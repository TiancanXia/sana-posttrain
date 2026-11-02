# SPDX-License-Identifier: Apache-2.0
"""Single-GPU Sana ART; frozen Euler replay and upstream ART tensor kernels.

See README.md and THIRD_PARTY_NOTICES.md for setup, scope and provenance.
The existing NFT entry point and its checkpoints are independent of this file.
"""

import argparse
from contextlib import nullcontext
import datetime
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from flow_grpo.art import art_loss, art_scalar_covariance
from flow_grpo.art.sampling import sample_art_trajectories
from flow_grpo.art.windows import prompt_disjoint_windows
from flow_grpo.diffusers_patch.pipeline_sana import (
    encode_prompt_conditions, load_pipeline, predict_guided_velocity,
)
from flow_grpo.rewards import multi_score

logger = logging.getLogger("sana_art")
TARGET_MODULES = [f"{attn}.{projection}" for attn in ("attn1", "attn2")
                  for projection in ("to_q", "to_k", "to_v", "to_out.0")]
FORMAT_VERSION = 1


def load_config(value):
    # A Windows drive colon is part of the path, not a preset separator.
    path, separator, preset = value.rpartition(":")
    if value.lower().endswith(".py") or not separator:
        path, preset = value, "smoke"
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("sana_art_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config(preset)


def read_prompts(path):
    prompts = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    prompts = [prompt.strip() for prompt in prompts]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError(f"Expected one nonempty prompt per line: {path}")
    return prompts


def autocast_context(config):
    if config.training.precision == "no":
        return nullcontext()
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[
        config.training.precision
    ])


def validate_config(config):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This ART entry supports one GPU/process only")
    if config.algorithm.name != "diffusionart" or config.algorithm.restore_zero_signal:
        raise ValueError("Use scalar ART with restore_zero_signal=False")
    if config.training.optimizer != "adamw":
        raise ValueError("This ART entry implements AdamW only")
    if config.training.precision not in ("no", "fp16", "bf16"):
        raise ValueError("precision must be no/fp16/bf16")
    positive = [config.sampling.steps, config.sampling.prompts_per_rollout,
                config.sampling.samples_per_prompt, config.sampling.microbatch,
                config.sampling.decode_microbatch, config.sampling.height, config.sampling.width,
                config.training.microbatch, config.training.max_updates,
                config.model.lora_rank, config.model.max_prompt_length, config.reward.microbatch,
                config.runtime.checkpoint_every, config.runtime.max_no_signal_rollouts]
    if any(type(value) is not int or value < 1 for value in positive):
        raise ValueError("Counts, dimensions and microbatches must be positive integers")
    if config.sampling.samples_per_prompt % 2:
        raise ValueError("Each prompt needs an even K for two disjoint ART windows")
    if config.training.max_updates % 2 or config.runtime.checkpoint_every % 2:
        raise ValueError("max_updates and checkpoint_every must be even: save complete rollouts")
    slots = list(config.sampling.train_timesteps)
    if not slots or any(type(slot) is not int or slot < 0 or slot >= config.sampling.steps for slot in slots):
        raise ValueError("train_timesteps must select valid zero-based sampling slots")
    if len(set(slots)) != len(slots):
        raise ValueError("train_timesteps must not repeat slots")
    if not math.isfinite(config.sampling.guidance_scale) or config.sampling.guidance_scale < 1:
        raise ValueError("guidance_scale must be finite and at least 1")
    for label, value in (("learning_rate", config.training.learning_rate),
                         ("epsilon", config.training.epsilon), ("grad_clip", config.training.grad_clip),
                         ("lora_alpha", config.model.lora_alpha)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    if not math.isfinite(config.training.weight_decay) or config.training.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if len(config.training.betas) != 2 or any(not 0 <= beta < 1 for beta in config.training.betas):
        raise ValueError("AdamW betas must be in [0,1)")
    if type(config.runtime.preview_images) is not int or config.runtime.preview_images < 0:
        raise ValueError("preview_images must be a nonnegative integer")
    if config.runtime.resume_from and config.runtime.warm_start_lora:
        raise ValueError("Choose resume_from or warm_start_lora, not both")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resume_contract(config, pipe):
    """Guard algorithm/data/schedule identity while allowing a longer step budget."""
    values = config.to_dict()
    training = dict(values["training"])
    training.pop("max_updates")
    model_root = Path(config.model.checkpoint)
    model_metadata = {}
    for relative in ("model_index.json", "transformer/config.json", "vae/config.json",
                     "text_encoder/config.json", "scheduler/scheduler_config.json"):
        model_metadata[relative] = file_sha256(model_root / relative)
    reward_metadata = {}
    for label, directory in (("model", config.reward.model_path), ("processor", config.reward.processor_path)):
        for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                         "special_tokens_map.json", "tokenizer.json", "vocab.json", "merges.txt"):
            asset = Path(directory) / filename
            if asset.is_file():
                reward_metadata[f"{label}/{filename}"] = file_sha256(asset)
    return {
        "algorithm": values["algorithm"], "model": values["model"],
        "reward": values["reward"], "data": values["data"], "sampling": values["sampling"],
        "training": training, "seed": config.seed,
        "train_sha256": file_sha256(config.data.train_file),
        "test_sha256": file_sha256(config.data.test_file), "model_metadata": model_metadata,
        "reward_metadata": reward_metadata,
        "scheduler_class": type(pipe.scheduler).__name__,
        "sigmas": pipe.scheduler.sigmas.detach().cpu().float().tolist(),
    }


def append_json(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def make_scorer(config, device):
    return multi_score(device, {"pickscore": 1.0},
                       pickscore_model_path=config.reward.model_path,
                       pickscore_processor_path=config.reward.processor_path,
                       pickscore_batch_size=config.reward.microbatch)


def score_images(scorer, images, prompts):
    scores, _ = scorer(images, prompts, [{} for _ in prompts])
    values = torch.tensor([float(value) for value in scores["avg"]], dtype=torch.float32)
    if values.shape != (len(prompts),) or not bool(torch.isfinite(values).all()):
        raise ValueError("PickScore must return one finite normalized score per image")
    return values


def save_checkpoint(run_dir, model, optimizer, scaler, config, contract, step, next_round):
    if step % 2:
        raise ValueError("ART checkpoints require a complete pair of updates")
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint / "lora", selected_adapters=["default"])
    torch.save({
        "algorithm": "art", "format_version": FORMAT_VERSION,
        "global_step": step, "next_round": next_round,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
        "contract": contract, "config": config.to_dict(),
    }, checkpoint / "training_state.pt")
    (checkpoint / "COMPLETE").write_text("art\n", encoding="utf-8")
    logger.info("Checkpoint saved: %s", checkpoint)


def collect_rollout(pipe, scorer, all_prompts, config, device, round_index, run_dir):
    """Keep complete prompt-instance groups; generation microbatches are independent."""
    selection_rng = torch.Generator(device="cpu").manual_seed(config.seed + round_index)
    chosen = torch.randperm(len(all_prompts), generator=selection_rng)[
        :config.sampling.prompts_per_rollout
    ].tolist()
    group_prompts = [all_prompts[index] for index in chosen]
    k = config.sampling.samples_per_prompt
    prompts = [prompt for prompt in group_prompts for _ in range(k)]
    groups = torch.arange(len(prompts), dtype=torch.long) // k
    embeds, masks, negatives, negative_masks = encode_prompt_conditions(
        pipe, prompts, device, config.model.max_prompt_length, config.model.offload,
        guidance_scale=config.sampling.guidance_scale,
    )
    conditioning = {"embeds": embeds.cpu(), "masks": masks.cpu()}
    if negatives is not None:
        conditioning.update(negatives=negatives.cpu(), negative_masks=negative_masks.cpu())
    del embeds, masks, negatives, negative_masks
    pieces, rewards = [], []
    pipe.transformer.eval()
    preview_count = 0
    for start in range(0, len(prompts), config.sampling.microbatch):
        stop = min(start + config.sampling.microbatch, len(prompts))
        logger.info("SAMPLE round=%d candidates=%d:%d/%d sampler=art_euler",
                    round_index, start, stop, len(prompts))
        with autocast_context(config):
            piece = sample_art_trajectories(
                pipe, conditioning["embeds"][start:stop], conditioning["masks"][start:stop],
                height=config.sampling.height, width=config.sampling.width,
                steps=config.sampling.steps, offload=config.model.offload,
                guidance_scale=config.sampling.guidance_scale,
                negative_prompt_embeds=conditioning.get("negatives", None)[start:stop]
                if "negatives" in conditioning else None,
                negative_prompt_attention_mask=conditioning.get("negative_masks", None)[start:stop]
                if "negative_masks" in conditioning else None,
                decode_batch_size=config.sampling.decode_microbatch,
            )
        images = piece.pop("images")
        values = score_images(scorer, images, prompts[start:stop])
        rewards.append(values)
        # Preview limit affects disk images only; ALL trajectories participate in training.
        if preview_count < config.runtime.preview_images:
            preview_dir = run_dir / "samples" / f"round-{round_index}"
            preview_dir.mkdir(parents=True, exist_ok=True)
            for offset in range(min(len(images), config.runtime.preview_images - preview_count)):
                filename = f"{preview_count:02d}.png"
                pixels = (images[offset].permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8)
                Image.fromarray(pixels.numpy()).save(preview_dir / filename)
                append_json(preview_dir / "scores.jsonl", {
                    "row": start + offset, "prompt": prompts[start + offset],
                    "pickscore": float(values[offset] * 26), "image": filename,
                })
                preview_count += 1
        del images
        pieces.append(piece)
    buffer = {name: torch.cat([piece[name] for piece in pieces], dim=0)
              for name in ("states", "old_velocities", "endpoints")}
    for name in ("sigmas", "dt", "timesteps"):
        if any(not torch.equal(piece[name], pieces[0][name]) for piece in pieces):
            raise ValueError("Sampling schedule changed within one ART rollout")
        buffer[name] = pieces[0][name]
    buffer.update(conditioning)
    buffer.update(groups=groups, prompts=prompts, group_prompts=group_prompts,
                  rewards=torch.cat(rewards))
    return buffer


def write_groups(run_dir, buffer, covariance, round_index, step):
    for group, prompt in enumerate(buffer["group_prompts"]):
        values = buffer["rewards"][buffer["groups"] == group]
        raw = values * 26.0
        record = {
            "round": round_index, "step_before": step, "group": group, "prompt": prompt,
            "count": len(values), "reward_mean": float(values.mean()),
            "pickscore_mean": float(raw.mean()), "pickscore_std": float(raw.std(unbiased=False)),
            "pickscore_values": raw.tolist(), "reward_normalized_values": values.tolist(),
            "covariance_rms": float(covariance["group_covariance_rms"][group]),
            "guidance_magnitude": float(covariance["group_guidance_magnitudes"][group]),
            "active": bool(covariance["group_active"][group]),
        }
        append_json(run_dir / "group_metrics.jsonl", record)
        logger.info("GROUP round=%d group=%d n=%d pickscore=%.6f std=%.6f active=%s",
                    round_index, group, len(values), record["pickscore_mean"],
                    record["pickscore_std"], record["active"])


def update_window(model, optimizer, scaler, parameters, buffer, covariance, items, config, device):
    """Mean over ALL window (image,time) items, including inactive/masked rows."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in parameters]
    sums = dict(loss=0.0, unmasked_loss=0.0, masked_fraction=0.0, kept_fraction=0.0,
                movement_score=0.0)
    for start in range(0, len(items), config.training.microbatch):
        batch = items[start:start + config.training.microbatch]
        rows = torch.tensor([item[0] for item in batch], dtype=torch.long)
        slots = torch.tensor([item[1] for item in batch], dtype=torch.long)
        group = buffer["groups"][rows]
        state = buffer["states"][rows, slots].to(device)
        old = buffer["old_velocities"][rows, slots].to(device)
        with autocast_context(config):
            current = predict_guided_velocity(
                model, state, buffer["timesteps"][slots].to(device),
                buffer["embeds"][rows].to(device), buffer["masks"][rows].to(device),
                guidance_scale=config.sampling.guidance_scale,
                negative_prompt_embeds=buffer["negatives"][rows].to(device)
                if "negatives" in buffer else None,
                negative_prompt_attention_mask=buffer["negative_masks"][rows].to(device)
                if "negative_masks" in buffer else None,
                concatenate=True,
            )
        result = art_loss(
            current, old, buffer["sigmas"][slots].to(device),
            covariance["group_directions"][group].to(device),
            covariance["group_guidance_magnitudes"][group].to(device), state=state,
            ode_velocity_coefficient=buffer["dt"][slots].to(device),
            **config.algorithm.loss.to_dict(),
        )
        losses = result["per_item_loss"]
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError("Nonfinite ART loss; optimizer was not advanced")
        scaler.scale(losses.sum() / len(items)).backward()
        for key, value in (("loss", losses), ("unmasked_loss", result["per_item_unmasked_loss"]),
                           ("masked_fraction", result["masked"].float()),
                           ("kept_fraction", result["keep_mask"].float()),
                           ("movement_score", result["movement_score"])):
            sums[key] += float(value.detach().double().sum())
        del result, current, losses, state, old
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.grad_clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    delta = max(float((parameter.detach() - previous).abs().max())
                for parameter, previous in zip(parameters, before, strict=True))
    if not math.isfinite(delta):
        raise FloatingPointError("Nonfinite updated LoRA weights; no checkpoint will be saved")
    optimizer.zero_grad(set_to_none=True)
    return dict({key: value / len(items) for key, value in sums.items()},
                grad_norm=float(norm), lora_update_max_abs=delta, item_count=len(items))


def train(config, device):
    validate_config(config)
    prompts = read_prompts(config.data.train_file)
    if config.sampling.prompts_per_rollout > len(prompts):
        raise ValueError("prompts_per_rollout exceeds the training dataset length")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
    if config.training.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support BF16")
    pipe = load_pipeline(config.model.checkpoint, device, config.model.offload)
    pipe.scheduler.set_timesteps(config.sampling.steps, device=device)
    contract = resume_contract(config, pipe)
    state = None
    adapter = config.runtime.warm_start_lora
    if config.runtime.resume_from:
        checkpoint = Path(config.runtime.resume_from).expanduser().resolve()
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError(f"Incomplete ART checkpoint: {checkpoint}")
        # Only use trusted self-produced training states: they contain Python RNG objects.
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if state.get("algorithm") != "art" or state.get("format_version") != FORMAT_VERSION:
            raise ValueError("Not a compatible ART checkpoint; NFT states cannot be resumed here")
        if state["contract"] != contract:
            raise ValueError("Resume changes model, data, sampling or ART/optimizer settings")
        adapter = str(checkpoint / "lora")
    if adapter:
        model = PeftModel.from_pretrained(pipe.transformer, adapter, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(pipe.transformer, LoraConfig(
            r=config.model.lora_rank, lora_alpha=config.model.lora_alpha,
            init_lora_weights="gaussian", target_modules=TARGET_MODULES,
        ))
    loaded = model.peft_config["default"]
    if (loaded.r != config.model.lora_rank or loaded.lora_alpha != config.model.lora_alpha
            or set(loaded.target_modules) != set(TARGET_MODULES)):
        raise ValueError("Loaded adapter rank/alpha/target modules differ from ART config")
    model.set_adapter("default")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any("lora_" not in name for name, value in model.named_parameters() if value.requires_grad):
        raise ValueError("Only a nonempty LoRA parameter set may receive gradients")
    if config.model.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    pipe.transformer = model
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate,
                                 betas=tuple(config.training.betas), eps=config.training.epsilon,
                                 weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.precision == "fp16")
    scorer = make_scorer(config, device)
    step, round_index = 0, 0
    if state is not None:
        step, round_index = state["global_step"], state["next_round"]
        if step < 0 or step % 2 or round_index < step // 2 or step >= config.training.max_updates:
            raise ValueError("Invalid/finished round-boundary checkpoint; increase max_updates to continue")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        # Restore after all model/scorer initialization, which may consume RNG.
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(config.runtime.output_dir).expanduser().resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
    ])
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "resume_contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("Run directory: %s", run_dir)
    logger.info("ART setup: P=%d K=%d images/rollout=%d updates/rollout=2 train_times=%d CFG=%.2f",
                config.sampling.prompts_per_rollout, config.sampling.samples_per_prompt,
                config.sampling.prompts_per_rollout * config.sampling.samples_per_prompt,
                len(config.sampling.train_timesteps), config.sampling.guidance_scale)
    logger.info("Trainable LoRA=%d; generation microbatch=%d; replay item microbatch=%d; decode microbatch=%d",
                sum(parameter.numel() for parameter in parameters), config.sampling.microbatch,
                config.training.microbatch, config.sampling.decode_microbatch)
    started = time.perf_counter()
    no_signal = 0
    while step < config.training.max_updates:
        torch.cuda.synchronize(device)
        round_started = time.perf_counter()
        buffer = collect_rollout(pipe, scorer, prompts, config, device, round_index, run_dir)
        covariance = art_scalar_covariance(
            buffer["endpoints"], buffer["rewards"], buffer["groups"],
            config.sampling.prompts_per_rollout, **config.algorithm.covariance.to_dict(),
        )
        write_groups(run_dir, buffer, covariance, round_index, step)
        if not bool(covariance["group_active"].any()):
            no_signal += 1
            logger.warning("NO_SIGNAL round=%d step=%d: no AdamW step", round_index, step)
            round_index += 1
            if no_signal >= config.runtime.max_no_signal_rollouts:
                raise RuntimeError("Repeated zero-signal rollouts; inspect rewards before spending more GPU time")
            del buffer, covariance
            continue
        no_signal = 0
        slots = list(config.sampling.train_timesteps)
        order = torch.randperm(len(buffer["prompts"])).tolist()
        windows = prompt_disjoint_windows(buffer["groups"], len(slots), order=order)
        for window_index, window in enumerate(windows):
            # The upstream splitter returns slot POSITIONS; map them to selected sampling slots.
            items = tuple((row, slots[slot]) for row, slot in window)
            torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            metrics = update_window(model, optimizer, scaler, parameters, buffer, covariance,
                                    items, config, device)
            torch.cuda.synchronize(device)
            step += 1
            selected_rows = torch.tensor(sorted({row for row, _ in items}), dtype=torch.long)
            rewards = buffer["rewards"][selected_rows]
            metrics.update(step=step, round=round_index, window=window_index,
                           reward_mean=float(rewards.mean()), pickscore_mean=float(rewards.mean() * 26),
                           reward_std=float(rewards.std(unbiased=False)),
                           pickscore_std=float(rewards.std(unbiased=False) * 26),
                           update_seconds=time.perf_counter() - update_started,
                           round_elapsed_seconds=time.perf_counter() - round_started,
                           guidance_scale=config.sampling.guidance_scale,
                           images_per_update=len(selected_rows), train_times=len(slots))
            append_json(run_dir / "metrics.jsonl", metrics)
            logger.info("METRIC step=%d round=%d window=%d loss=%.6f grad_norm=%.6f reward=%.6f masked=%.3f delta=%.3e time=%.2fs",
                        step, round_index, window_index, metrics["loss"], metrics["grad_norm"],
                        metrics["pickscore_mean"], metrics["masked_fraction"],
                        metrics["lora_update_max_abs"], metrics["update_seconds"])
        round_index += 1
        # A checkpoint's LoRA, optimizer and RNG always describe the same complete rollout boundary.
        if step % config.runtime.checkpoint_every == 0 or step == config.training.max_updates:
            save_checkpoint(run_dir, model, optimizer, scaler, config, contract, step, round_index)
        del buffer, covariance
    torch.cuda.synchronize(device)
    logger.info("DONE steps=%d next_round=%d training_seconds=%.2f output=%s",
                step, round_index, time.perf_counter() - started, run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/art.py") + ":smoke")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    train(config, torch.device(f"cuda:{local_rank}"))


if __name__ == "__main__":
    main()
