# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVlabs/DiffusionNFT/scripts/train_nft_sd3.py.
# Modified for Sana and single-GPU execution; see THIRD_PARTY_NOTICES.md.

from collections import defaultdict
from contextlib import nullcontext
import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

# Make this checkout authoritative even if another flow_grpo is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from absl import app, flags
from ml_collections import config_flags
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from peft import LoraConfig, get_peft_model, PeftModel

from flow_grpo.diffusers_patch.pipeline_sana import (
    load_pipeline, encode_prompt_conditions, sample_images, predict_guided_velocity,
)
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.rewards import multi_score
from flow_grpo.stat_tracking import PerPromptStatTracker


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", str(PROJECT_ROOT / "config/sana.py"), "Sana NFT config.")
logger = logging.getLogger("sana_nft")
# The unchanged upstream scorer returns raw PickScore / 26 for optimization.
# Convert only reporting values back to the paper's raw score units.
PICKSCORE_REPORT_MULTIPLIER = 26.0


class TextPromptDataset(Dataset):
    """Original NFT plain-text prompt dataset contract."""

    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines()]
        if not self.prompts or any(not prompt for prompt in self.prompts):
            raise ValueError(f"Empty dataset/prompt: {self.file_path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        return [example["prompt"] for example in examples], [example["metadata"] for example in examples]


class DistributedKRepeatSampler(Sampler):
    """Original NFT grouping/shuffle order; batch_size is a logical batch."""

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, "Candidate groups must fit a logical sampling batch"
        self.m = self.total_samples // self.k
        if self.m > len(dataset):
            raise ValueError("Logical batch requests more distinct prompts than the dataset contains")
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def return_decay(step, decay_type):
    """Unchanged original NFT old-policy refresh schedule."""
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    return 0.0 if step < flat else min((step - flat) * uprate, uphold)


def autocast_context(config):
    if config.mixed_precision == "no":
        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[config.mixed_precision]
    return torch.autocast("cuda", dtype=dtype)


def validate_config(config, world_size):
    if world_size != 1:
        raise ValueError("This initial Sana entry supports --nproc_per_node=1 only")
    if set(config.reward_fn) != {"pickscore"}:
        raise ValueError("This initial port is scoped to PickScore; optional inherited scorers are not validated")
    if not config.use_lora:
        raise ValueError("This entry point supports LoRA, not full-model finetuning")
    if not math.isfinite(config.sample.guidance_scale) or config.sample.guidance_scale < 1.0:
        raise ValueError("Training guidance_scale must be finite and at least 1.0")
    if not config.sample.deterministic:
        raise ValueError("This Sana port requires deterministic NFT sampling")
    if config.sample.solver not in ("dpm1", "dpm2"):
        raise ValueError("Use dpm1 or dpm2 with the unchanged NFT solver")
    if config.mixed_precision not in ("no", "fp16", "bf16"):
        raise ValueError("mixed_precision must be no/fp16/bf16")
    if config.beta <= 0 or config.train.adv_clip_max <= 0:
        raise ValueError("NFT beta and advantage clipping scale must be positive")
    positive = [config.num_epochs, config.max_train_steps, config.sample.num_steps,
                config.sample.train_batch_size, config.sample.num_image_per_prompt,
                config.sample.num_batches_per_epoch, config.sample.generation_batch_size,
                config.train.batch_size, config.train.gradient_accumulation_steps,
                config.train.num_inner_epochs, config.sample.test_batch_size]
    if any(int(value) != value or value < 1 for value in positive):
        raise ValueError("Epochs, steps and all batch sizes must be positive integers")
    if not 0 < config.train.timestep_fraction <= 1:
        raise ValueError("timestep_fraction must lie in (0,1]")
    if int(config.sample.num_steps * config.train.timestep_fraction) < 1:
        raise ValueError("Configuration selects zero training timesteps")
    if config.sample.num_image_per_prompt < 2 or not config.per_prompt_stat_tracking:
        raise ValueError("Use at least two candidates and the original prompt stat tracker")
    if config.sample.train_batch_size % config.sample.num_image_per_prompt:
        raise ValueError("Logical sampling batch must contain complete candidate groups")
    samples = config.sample.train_batch_size * config.sample.num_batches_per_epoch
    update_batch = config.train.batch_size * config.train.gradient_accumulation_steps
    if samples % update_batch:
        raise ValueError("Collected samples must divide exactly into accumulated optimizer batches")
    updates_per_round = samples // update_batch * config.train.num_inner_epochs
    if config.max_train_steps % updates_per_round:
        raise ValueError("max_train_steps must end at a complete outer round for exact round-boundary saves")
    if config.save_freq < 1 or config.eval_freq < 0:
        raise ValueError("save_freq must be positive and eval_freq non-negative")
    if config.resume_from and config.train.lora_path:
        raise ValueError("Choose resume_from OR an initial lora_path")
    return updates_per_round


def nft_policy_predictions(model, xt, timesteps, batch, *, guidance_scale):
    """Use the same guided field for current, detached old and frozen reference.

    Both current CFG branches remain differentiable. One concatenated DDP
    forward per policy avoids multiple current forwards before one backward.
    The default adapter is restored before gradient-checkpoint recomputation.
    """
    transformer = model.module if hasattr(model, "module") else model
    conditions = dict(
        guidance_scale=guidance_scale, concatenate=True,
        negative_prompt_embeds=batch.get("negative_prompt_embeds"),
        negative_prompt_attention_mask=batch.get("negative_prompt_attention_mask"),
    )

    def predict():
        return predict_guided_velocity(
            model, xt, timesteps, batch["prompt_embeds"], batch["prompt_attention_mask"],
            **conditions,
        )

    try:
        transformer.set_adapter("old")
        with torch.no_grad():
            old_prediction = predict().detach()
        transformer.set_adapter("default")
        forward_prediction = predict()
        with torch.no_grad(), transformer.disable_adapter():
            ref_prediction = predict()
        return forward_prediction, old_prediction, ref_prediction
    finally:
        transformer.set_adapter("default")


def nft_loss(config, x0, xt, t_expanded, forward_prediction, old_prediction,
             ref_forward_prediction, advantages):
    """Original NFT adaptive loss, over conditional or consistently guided fields."""
    advantages_clip = torch.clamp(advantages, -config.train.adv_clip_max, config.train.adv_clip_max)
    mode = config.train.adv_mode
    if mode == "positive_only":
        advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
    elif mode == "negative_only":
        advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
    elif mode == "one_only":
        advantages_clip = torch.where(advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip))
    elif mode == "binary":
        advantages_clip = torch.sign(advantages_clip)
    elif mode != "all":
        raise ValueError(f"Unknown adv_mode: {mode}")
    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
    r = torch.clamp(normalized_advantages_clip, 0, 1)
    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
    implicit_negative_prediction = (1.0 + config.beta) * old_prediction.detach() - config.beta * forward_prediction
    x0_prediction = xt - t_expanded * positive_prediction
    with torch.no_grad():
        weight_factor = (torch.abs(x0_prediction.double() - x0.double())
                         .mean(dim=tuple(range(1, x0.ndim)), keepdim=True).clip(min=0.00001))
    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
    with torch.no_grad():
        negative_weight_factor = (torch.abs(negative_x0_prediction.double() - x0.double())
                                  .mean(dim=tuple(range(1, x0.ndim)), keepdim=True).clip(min=0.00001))
    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(dim=tuple(range(1, x0.ndim)))
    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()
    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim))).mean()
    loss = policy_loss + config.train.beta * kl_div_loss
    return loss, {
        "loss": loss.detach(), "policy_loss": policy_loss.detach(),
        "unweighted_policy_loss": ori_policy_loss.mean().detach(), "kl_loss": kl_div_loss.detach(),
        "x0_norm": (x0 ** 2).mean().detach(),
        "old_deviation": ((forward_prediction - old_prediction) ** 2).mean().detach(),
    }


def pickscore_record(prompt, normalized_reward, image):
    """Keep both units explicit in new evaluation and preview JSON records."""
    normalized_reward = float(normalized_reward)
    return {
        "prompt": prompt, "reward": normalized_reward * PICKSCORE_REPORT_MULTIPLIER,
        "reward_normalized": normalized_reward, "reward_scale": "pickscore_raw",
        "image": image,
    }


def summarize_pickscore_rewards(normalized_rewards):
    """Reporting only: retain existing normalized fields and add raw PickScore."""
    rewards = np.asarray(normalized_rewards, dtype=np.float64)
    metrics = {
        "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()), "reward_max": float(rewards.max()),
        "reward_scale": "pickscore_div26", "pickscore_scale": "raw",
        "pickscore_report_multiplier": PICKSCORE_REPORT_MULTIPLIER,
    }
    for statistic in ("mean", "std", "min", "max"):
        metrics[f"pickscore_{statistic}"] = metrics[f"reward_{statistic}"] * PICKSCORE_REPORT_MULTIPLIER
    return metrics


def summarize_prompt_groups(prompts, normalized_rewards, advantages, *, adv_clip_max):
    """Reporting only: one reward/advantage per image, before clipping or shuffling."""
    prompts = np.asarray(prompts)
    rewards = np.asarray(normalized_rewards, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    if rewards.ndim != 1 or prompts.shape != rewards.shape or advantages.shape != rewards.shape or not rewards.size:
        raise ValueError("Group reporting requires matching nonempty one-dimensional inputs")
    if not math.isfinite(adv_clip_max) or adv_clip_max <= 0:
        raise ValueError("Group reporting requires a positive finite advantage clip boundary")
    if not np.isfinite(rewards).all() or not np.isfinite(advantages).all():
        raise ValueError("Cannot report nonfinite rewards or advantages")

    def advantage_stats(values):
        return {
            "advantage_mean": float(values.mean()), "advantage_std": float(values.std()),
            "advantage_abs_mean": float(np.abs(values).mean()),
            "advantage_min": float(values.min()), "advantage_max": float(values.max()),
            "advantage_clip_fraction": float((np.abs(values) > adv_clip_max).mean()),
            "advantage_clip_max": float(adv_clip_max),
        }

    groups = []
    for group_index, prompt in enumerate(dict.fromkeys(prompts.tolist()), start=1):
        selected = prompts == prompt
        group_rewards, group_advantages = rewards[selected], advantages[selected]
        record = summarize_pickscore_rewards(group_rewards)
        record.update(advantage_stats(group_advantages))
        record.update(
            group=group_index, prompt=prompt, count=int(selected.sum()),
            statistics_scope="collection_round",
            pickscore_values=(group_rewards * PICKSCORE_REPORT_MULTIPLIER).tolist(),
            advantage_values=group_advantages.tolist(),
        )
        groups.append(record)
    summary = advantage_stats(advantages)
    summary.update(prompt_group_count=len(groups), statistics_scope="collection_round")
    return summary, groups


def write_group_metrics(run_dir, groups, *, step, epoch):
    """Keep full group records on disk and one short terminal line per prompt."""
    with (run_dir / "group_metrics.jsonl").open("a", encoding="utf-8") as f:
        for group in groups:
            record = dict(step=step, epoch=epoch, **group)
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            logger.info(
                "GROUP step=%d group=%d n=%d pickscore=%.6f std=%.6f adv_std=%.6f adv_abs=%.6f clip=%.2f%%",
                step, group["group"], group["count"], group["pickscore_mean"], group["pickscore_std"],
                group["advantage_std"], group["advantage_abs_mean"],
                100.0 * group["advantage_clip_fraction"],
            )


def write_metric(run_dir, metrics):
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, allow_nan=False) + "\n")
    logger.info(
        "METRIC step=%d loss=%.6f grad_norm=%.6f reward=%.6f std=%.6f reward_norm=%.6f delta=%.3e cfg=%.2f time=%.2fs score_scale=raw",
        metrics["step"], metrics["loss"], metrics["grad_norm"], metrics["pickscore_mean"],
        metrics["pickscore_std"], metrics["reward_mean"], metrics["lora_update_max_abs"],
        metrics["guidance_scale"], metrics["iteration_seconds"],
    )


def save_ckpt(run_dir, model, optimizer, scaler, ema, params, old_params,
              global_step, next_epoch, config):
    """Save raw default weights with matching optimizer, old policy, EMA and RNG."""
    model.set_adapter("default")
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{global_step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint / "lora", selected_adapters=["default"])
    state = {
        "global_step": global_step, "next_epoch": next_epoch,
        "old_params": [p.detach().cpu().clone() for p in old_params],
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
        "config": config.to_dict(),
    }
    torch.save(state, checkpoint / "training_state.pt")
    if ema is not None:
        ema.copy_ema_to(params, store_temp=True)
        try:
            model.save_pretrained(checkpoint / "lora_ema", selected_adapters=["default"])
        finally:
            ema.copy_temp_to(params)
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    logger.info("Checkpoint saved: %s", checkpoint)


@torch.no_grad()
def evaluate(pipe, dataset, reward_fn, config, device, output_dir, params=None, ema=None,
             *, guidance_scale=None, sampling_backend="nft", adapter_label="current"):
    """Small, fixed-seed held-out evaluation; reuse the same scorer instance."""
    limit = config.eval_max_samples
    if limit < 1:
        raise ValueError("eval_max_samples must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    guidance_scale = config.sample.guidance_scale if guidance_scale is None else guidance_scale
    count = min(limit, len(dataset))
    settings = {
        "model": str(config.pretrained.model), "prompt_split": "test", "count": count,
        "seed": config.seed, "initial_noise_seed": config.seed + 10000,
        "guidance_scale": guidance_scale, "sampler": sampling_backend,
        "adapter": adapter_label, "ema_applied": ema is not None,
        "nft_solver": config.sample.solver,
        "native_scheduler": type(pipe.scheduler).__name__,
        "native_scheduler_config": {
            key: pipe.scheduler.config.get(key) for key in (
                "algorithm_type", "solver_order", "solver_type", "prediction_type",
                "use_flow_sigmas", "flow_shift", "num_train_timesteps",
                "lower_order_final", "euler_at_final", "timestep_spacing", "final_sigmas_type",
            )
        },
        "steps": config.sample.eval_num_steps, "resolution": config.resolution,
        "generation_batch_size": config.sample.generation_batch_size,
        "max_prompt_length": config.max_prompt_length,
        "mixed_precision": config.mixed_precision, "base_weight_dtype": "float32",
        "scores_schema_version": 2, "reward_scale": "pickscore_raw",
        "reward_normalized_scale": "pickscore_div26",
        "pickscore_report_multiplier": PICKSCORE_REPORT_MULTIPLIER,
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    logger.info("EVAL start count=%d sampler=%s cfg=%.2f steps=%d output=%s",
                count, sampling_backend, guidance_scale, config.sample.eval_num_steps, output_dir)
    if ema is not None:
        ema.copy_ema_to(params, store_temp=True)
    try:
        pipe.transformer.eval()
        generator = torch.Generator(device=device).manual_seed(config.seed + 10000)
        records = []
        for start in range(0, min(limit, len(dataset)), config.sample.test_batch_size):
            examples = [dataset[i] for i in range(start, min(start + config.sample.test_batch_size, limit, len(dataset)))]
            prompts, metadata = dataset.collate_fn(examples)
            logger.info("EVAL encode prompts=%d:%d/%d", start, start + len(prompts), count)
            embeds, masks, negative_embeds, negative_masks = encode_prompt_conditions(
                pipe, prompts, device, config.max_prompt_length, config.model_offload,
                guidance_scale=guidance_scale,
            )
            # Split even eval batches to avoid an accidental memory increase.
            for offset in range(0, len(prompts), config.sample.generation_batch_size):
                end = min(offset + config.sample.generation_batch_size, len(prompts))
                logger.info("EVAL generate images=%d:%d/%d sampler=%s cfg=%.2f",
                            start + offset, start + end, count, sampling_backend, guidance_scale)
                with autocast_context(config):
                    images, _, _ = sample_images(
                        pipe, embeds[offset:end], masks[offset:end], height=config.resolution,
                        width=config.resolution, steps=config.sample.eval_num_steps,
                        solver=config.sample.solver, generator=generator, offload=config.model_offload,
                        guidance_scale=guidance_scale, sampling_backend=sampling_backend,
                        negative_prompt_embeds=negative_embeds[offset:end] if negative_embeds is not None else None,
                        negative_prompt_attention_mask=negative_masks[offset:end] if negative_masks is not None else None,
                    )
                scores, _ = reward_fn(images, prompts[offset:end], metadata[offset:end])
                for j, (prompt, score) in enumerate(zip(prompts[offset:end], scores["avg"])):
                    filename = f"{start + offset + j:05d}.png"
                    pipe.image_processor.postprocess(
                        images[j:j+1], output_type="pil", do_denormalize=[False],
                    )[0].save(output_dir / filename)
                    record = pickscore_record(prompt, score, filename)
                    records.append(record)
                    logger.info("EVAL image=%d/%d reward=%.6f reward_norm=%.6f saved=%s score_scale=raw",
                                len(records), count, record["reward"], record["reward_normalized"], filename)
        with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        mean = float(np.mean([r["reward"] for r in records]))
        logger.info("EVAL count=%d reward_mean=%.6f reward_mean_normalized=%.6f images=%s score_scale=raw",
                    len(records), mean, mean / PICKSCORE_REPORT_MULTIPLIER, output_dir)
        return mean
    finally:
        if ema is not None:
            ema.copy_temp_to(params)


def main(_):
    config = FLAGS.config
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    updates_per_round = validate_config(config, world_size)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        train(config, device, updates_per_round)
    finally:
        dist.destroy_process_group()


def train(config, device, updates_per_round):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(config.save_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
    ])
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    logger.info("Run directory: %s; source: %s", run_dir, PROJECT_ROOT)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support BF16")
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "fp16")
    pipe = load_pipeline(config.pretrained.model, device, config.model_offload)
    lora_config = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, init_lora_weights="gaussian",
        target_modules=[f"{attn}.{projection}" for attn in ("attn1", "attn2")
                        for projection in ("to_q", "to_k", "to_v", "to_out.0")],
    )
    resume_state = None
    if config.resume_from:
        checkpoint = Path(config.resume_from).expanduser().resolve()
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError(f"Checkpoint is incomplete or not this port's format: {checkpoint}")
        # Only load self-produced, trusted training state; it contains RNG objects.
        resume_state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint / "lora", is_trainable=True, local_files_only=True)
    elif config.train.lora_path:
        transformer = PeftModel.from_pretrained(pipe.transformer, config.train.lora_path, is_trainable=True, local_files_only=True)
    else:
        transformer = get_peft_model(pipe.transformer, lora_config)
    loaded_lora = transformer.peft_config["default"]
    if loaded_lora.r != config.lora_rank or loaded_lora.lora_alpha != config.lora_alpha:
        raise ValueError("Loaded adapter rank/alpha differ from config; record the actual adapter settings")
    transformer.add_adapter("old", transformer.peft_config["default"])
    pipe.transformer = transformer
    transformer.set_adapter("default")
    params = [p for p in transformer.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in params}
    transformer.set_adapter("old")
    old_params = [p for p in transformer.parameters() if p.requires_grad]
    transformer.set_adapter("default")
    if not params or len(params) != len(old_params):
        raise ValueError("Default/old adapters must have matching parameter sets")
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer_ddp = DDP(transformer, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        params, lr=config.train.learning_rate, betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon,
    )
    ema = EMAModuleWrapper(params, decay=0.9, update_step_interval=1, device=device) if config.train.ema else None
    train_dataset, test_dataset = TextPromptDataset(config.dataset), TextPromptDataset(config.dataset, "test")
    sampler = DistributedKRepeatSampler(train_dataset, config.sample.train_batch_size,
                                        config.sample.num_image_per_prompt, 1, 0, config.seed)
    dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0, collate_fn=train_dataset.collate_fn)
    # DataLoader iterator initialization may consume RNG; do it before restore.
    train_iter = iter(dataloader)
    reward_fn = multi_score(
        device, config.reward_fn, pickscore_model_path=config.reward_model_path,
        pickscore_processor_path=config.reward_processor_path,
    )
    stat_tracker = PerPromptStatTracker(config.sample.global_std)
    global_step, first_epoch = 0, 0
    if resume_state is not None:
        saved_config = resume_state["config"]
        # Reject changes that invalidate old-policy state, sampling or accumulation.
        current = config.to_dict()
        for key in ("sample", "train", "beta", "decay_type", "lora_rank", "lora_alpha", "resolution",
                    "mixed_precision", "max_prompt_length", "seed", "dataset", "pretrained", "reward_fn",
                    "reward_model_path", "reward_processor_path", "eval_freq", "eval_max_samples",
                    "model_offload", "gradient_checkpointing", "allow_tf32"):
            if current[key] != saved_config[key]:
                raise ValueError(f"Resume requires unchanged {key}; use train.lora_path for a new experiment")
        global_step, first_epoch = resume_state["global_step"], resume_state["next_epoch"]
        if len(resume_state["old_params"]) != len(old_params):
            raise ValueError("Saved old adapter parameter count differs")
        with torch.no_grad():
            for saved, target in zip(resume_state["old_params"], old_params, strict=True):
                target.copy_(saved.to(target.device))
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        if ema is not None:
            ema.load_state_dict(resume_state["ema"])
        random.setstate(resume_state["rng_python"])
        np.random.set_state(resume_state["rng_numpy"])
        torch.set_rng_state(resume_state["rng_torch"])
        torch.cuda.set_rng_state_all(resume_state["rng_cuda"])
        logger.info("Resumed round boundary: step=%d next_epoch=%d", global_step, first_epoch)
    else:
        with torch.no_grad():
            for source, target in zip(params, old_params, strict=True):
                target.copy_(source.detach())
    logger.info("Trainable default LoRA parameters: %d; logical batch=%d; generation microbatch=%d",
                sum(p.numel() for p in params), config.sample.train_batch_size, config.sample.generation_batch_size)
    logger.info("NFT setup: cfg=%.2f for rollout/current/old/reference; solver=%s; training microbatch=%d",
                config.sample.guidance_scale, config.sample.solver, config.train.batch_size)
    if config.sample.guidance_scale > 1:
        logger.info("Guided NFT extension: both current CFG branches receive gradients; paired training batch=%d",
                    2 * config.train.batch_size)
    if first_epoch >= config.num_epochs or global_step >= config.max_train_steps:
        raise ValueError("No remaining rounds/steps; increase num_epochs and max_train_steps to resume")
    needed_rounds = (config.max_train_steps - global_step) // updates_per_round
    if config.num_epochs - first_epoch < needed_rounds:
        raise ValueError("num_epochs is too small to reach max_train_steps")
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(first_epoch, first_epoch + needed_rounds):
        # COLLECT: frozen old policy, logical groups intact across microbatches.
        transformer.eval()
        transformer.set_adapter("old")
        samples, all_prompts, all_rewards, sample_images_preview = [], [], [], []
        for batch_index in range(config.sample.num_batches_per_epoch):
            sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + batch_index)
            prompts, metadata = next(train_iter)
            embeds, masks, negative_embeds, negative_masks = encode_prompt_conditions(
                pipe, prompts, device, config.max_prompt_length, config.model_offload,
                guidance_scale=config.sample.guidance_scale,
            )
            for start in range(0, len(prompts), config.sample.generation_batch_size):
                end = min(start + config.sample.generation_batch_size, len(prompts))
                logger.info("SAMPLE round=%d batch=%d candidates=%d:%d/%d", epoch, batch_index, start, end, len(prompts))
                with autocast_context(config):
                    images, clean, timesteps = sample_images(
                        pipe, embeds[start:end], masks[start:end], height=config.resolution,
                        width=config.resolution, steps=config.sample.num_steps,
                        solver=config.sample.solver, offload=config.model_offload,
                        guidance_scale=config.sample.guidance_scale,
                        negative_prompt_embeds=negative_embeds[start:end] if negative_embeds is not None else None,
                        negative_prompt_attention_mask=negative_masks[start:end] if negative_masks is not None else None,
                    )
                rewards, _ = reward_fn(images, prompts[start:end], metadata[start:end])
                # Upstream multi_score returns a list of scalar CUDA tensors.
                # Convert explicitly; NumPy cannot directly read CUDA tensors.
                values = np.asarray([float(score) for score in rewards["avg"]], dtype=np.float64)
                if len(values) != end - start or not np.isfinite(values).all():
                    raise ValueError("Reward count mismatch or nonfinite PickScore")
                sample = {"latents_clean": clean, "prompt_embeds": embeds[start:end],
                          "prompt_attention_mask": masks[start:end],
                          "timesteps": timesteps.unsqueeze(0).repeat(end-start, 1)}
                if negative_embeds is not None:
                    # Keep unconditional inputs paired through cat/shuffle/microbatch slicing.
                    sample["negative_prompt_embeds"] = negative_embeds[start:end]
                    sample["negative_prompt_attention_mask"] = negative_masks[start:end]
                samples.append(sample)
                all_prompts.extend(prompts[start:end])
                all_rewards.extend(values.tolist())
                if len(sample_images_preview) < 4:
                    sample_images_preview.append((prompts[start], float(values[0]), images[0:1].cpu()))
                del images
        transformer.set_adapter("default")
        collated = {key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]}
        reward_array = np.asarray(all_rewards)
        # Preserve original per-prompt centering and optional GLOBAL std exactly.
        repeated_rewards = np.repeat(reward_array[:, None], num_train_timesteps, axis=1)
        advantages = stat_tracker.update(all_prompts, repeated_rewards)
        stat_tracker.clear()
        collated["advantages"] = torch.from_numpy(advantages).to(device)
        reward_metrics = summarize_pickscore_rewards(reward_array)
        # Each image has the same advantage at all repeated training times.
        # Read one column for diagnostics; never modify the training arrays.
        advantage_metrics, group_metrics = summarize_prompt_groups(
            all_prompts, reward_array, advantages[:, 0], adv_clip_max=config.train.adv_clip_max,
        )
        reward_metrics.update(advantage_metrics)
        preview_dir = run_dir / "samples" / f"round-{epoch}"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, (prompt, score, image) in enumerate(sample_images_preview):
            pipe.image_processor.postprocess(
                image, output_type="pil", do_denormalize=[False],
            )[0].save(preview_dir / f"{i:02d}.png")
        (preview_dir / "prompts.json").write_text(json.dumps([
            pickscore_record(p, s, f"{i:02d}.png")
            for i, (p, s, _) in enumerate(sample_images_preview)
        ], ensure_ascii=False, indent=2), encoding="utf-8")
        del samples, sample, embeds, masks, negative_embeds, negative_masks, sample_images_preview

        # TRAIN: same image shuffle, full-time shuffle, random K times, NFT loss.
        transformer_ddp.train()
        total_samples = len(all_prompts)
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        accumulated_steps, info_accumulated = 0, defaultdict(list)
        update_start = time.perf_counter()
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_samples, device=device)
            shuffled = {key: value[perm] for key, value in collated.items()}
            time_perm = torch.stack([torch.randperm(config.sample.num_steps, device=device) for _ in range(total_samples)])
            shuffled["timesteps"] = shuffled["timesteps"][torch.arange(total_samples, device=device)[:, None], time_perm]
            for start in range(0, total_samples, config.train.batch_size):
                batch = {key: value[start:start + config.train.batch_size] for key, value in shuffled.items()}
                for j in range(num_train_timesteps):
                    x0 = batch["latents_clean"]
                    t = batch["timesteps"][:, j].float() / 1000.0
                    t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
                    noise = torch.randn_like(x0.float())
                    xt = (1 - t_expanded) * x0 + t_expanded * noise
                    with autocast_context(config):
                        forward_prediction, old_prediction, ref_prediction = nft_policy_predictions(
                            transformer_ddp, xt, batch["timesteps"][:, j], batch,
                            guidance_scale=config.sample.guidance_scale,
                        )
                    # Checkpoint recomputation during backward must use default.
                    transformer.set_adapter("default")
                    if any(p.requires_grad for p in transformer.parameters() if id(p) not in trainable_ids):
                        raise RuntimeError("A non-default/base parameter unexpectedly became trainable")
                    loss, terms = nft_loss(config, x0, xt, t_expanded, forward_prediction,
                                           old_prediction, ref_prediction, batch["advantages"][:, j])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite NFT loss before backward")
                    scaler.scale(loss / effective_grad_accum_steps).backward()
                    accumulated_steps += 1
                    for key, value in terms.items():
                        info_accumulated[key].append(value)
                    if accumulated_steps % effective_grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(params, config.train.max_grad_norm, error_if_nonfinite=True)
                        before = [p.detach().clone() for p in params]
                        scaler.step(optimizer)
                        scaler.update()
                        delta = max((p.detach() - old).abs().max().item() for p, old in zip(params, before, strict=True))
                        del before
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if ema is not None:
                            ema.step(params, global_step)
                        torch.cuda.synchronize(device)
                        metrics = {key: float(torch.stack(values).mean()) for key, values in info_accumulated.items()}
                        metrics.update(reward_metrics)
                        metrics.update(step=global_step, epoch=epoch, inner_epoch=inner_epoch,
                                       grad_norm=float(norm), lora_update_max_abs=delta,
                                       iteration_seconds=time.perf_counter()-update_start,
                                       learning_rate=optimizer.param_groups[0]["lr"],
                                        guidance_scale=config.sample.guidance_scale,
                                       peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
                        write_metric(run_dir, metrics)
                        write_group_metrics(run_dir, group_metrics, step=global_step, epoch=epoch)
                        if delta == 0:
                            logger.warning("Optimizer step produced no measurable LoRA change")
                        info_accumulated.clear()
                        update_start = time.perf_counter()
        if accumulated_steps % effective_grad_accum_steps:
            raise RuntimeError("Unconsumed gradients at round end")

        # Refresh old ONCE after all inner epochs, as in the original script.
        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for source, target in zip(params, old_params, strict=True):
                target.copy_(target.detach() * decay + source.detach().clone() * (1.0-decay))
        if config.eval_freq and (epoch + 1) % config.eval_freq == 0:
            evaluate(pipe, test_dataset, reward_fn, config, device, run_dir / "eval" / f"step-{global_step}", params, ema)
        if (epoch + 1) % config.save_freq == 0 or global_step == config.max_train_steps:
            save_ckpt(run_dir, transformer, optimizer, scaler, ema, params, old_params, global_step, epoch+1, config)
    logger.info("DONE steps=%d output=%s", global_step, run_dir)


if __name__ == "__main__":
    app.run(main)
