# SPDX-License-Identifier: Apache-2.0
"""Fixed-seed evaluation using the same Euler sampler as Sana ART collection."""

import argparse
import json
import logging
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from PIL import Image
import torch
from peft import PeftModel

from flow_grpo.art.sampling import sample_art_trajectories
from flow_grpo.diffusers_patch.pipeline_sana import encode_prompt_conditions, load_pipeline
from train_art_sana import append_json, load_config, make_scorer, read_prompts, score_images


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/art.py") + ":smoke")
    parser.add_argument("--adapter", type=Path, help="Standard PEFT checkpoint-N/lora directory")
    parser.add_argument("--output", type=Path, required=True, help="A NEW output directory")
    parser.add_argument("--num-images", type=int, help="Defaults to config.evaluation.count")
    parser.add_argument("--seed", type=int, help="Defaults to config.seed")
    parser.add_argument("--steps", type=int, help="Defaults to config.evaluation.steps")
    parser.add_argument("--guidance-scale", type=float, help="Defaults to config.evaluation.guidance_scale")
    args = parser.parse_args()
    config = load_config(args.config)
    args.num_images = config.evaluation.count if args.num_images is None else args.num_images
    args.seed = config.seed if args.seed is None else args.seed
    args.steps = config.evaluation.steps if args.steps is None else args.steps
    args.guidance_scale = config.evaluation.guidance_scale if args.guidance_scale is None else args.guidance_scale
    if args.num_images < 1 or args.steps < 1:
        parser.error("num-images and steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error("guidance-scale must be finite and at least 1")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Choose a new output directory to preserve previous comparisons")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prompts = read_prompts(config.data.test_file)[:args.num_images]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
    device = torch.device("cuda:0")
    pipe = load_pipeline(config.model.checkpoint, device, config.model.offload)
    if args.adapter:
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, str(args.adapter.expanduser().resolve()),
            is_trainable=False, local_files_only=True,
        )
    pipe.transformer.eval()
    scorer = make_scorer(config, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 10000)
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "model": config.model.checkpoint,
        "adapter": str(args.adapter.expanduser().resolve()) if args.adapter else "base",
        "sampler": "art_euler", "sigma_grid": "Sana checkpoint scheduler",
        "scheduler_class": type(pipe.scheduler).__name__,
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "seed": args.seed, "initial_noise_seed": args.seed + 10000,
        "count": len(prompts), "prompt_file": config.data.test_file,
        "height": config.sampling.height, "width": config.sampling.width,
        "generation_microbatch": 1, "decode_microbatch": 1,
        "base_weight_dtype": "float32", "reward_scale": "pickscore_raw",
        "reward_normalized_scale": "pickscore_div26",
    }
    pipe.scheduler.set_timesteps(args.steps, device=device)
    settings["sigmas"] = pipe.scheduler.sigmas.detach().cpu().float().tolist()
    (output / "evaluation_config.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    raw_scores = []
    for index, prompt in enumerate(prompts):
        logging.info("EVAL image=%d/%d sampler=art_euler cfg=%.2f", index + 1, len(prompts), args.guidance_scale)
        embeds, mask, negatives, negative_mask = encode_prompt_conditions(
            pipe, [prompt], device, config.model.max_prompt_length, config.model.offload,
            guidance_scale=args.guidance_scale,
        )
        result = sample_art_trajectories(
            pipe, embeds, mask, height=config.sampling.height, width=config.sampling.width,
            steps=args.steps, generator=generator, offload=config.model.offload,
            guidance_scale=args.guidance_scale, negative_prompt_embeds=negatives,
            negative_prompt_attention_mask=negative_mask, decode_batch_size=1, cache_trajectory=False,
        )
        normalized = float(score_images(scorer, result["images"], [prompt])[0])
        raw = normalized * 26.0
        filename = f"{index:05d}.png"
        pixels = (result["images"][0].permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8)
        Image.fromarray(pixels.numpy()).save(output / filename)
        append_json(output / "scores.jsonl", {
            "prompt": prompt, "reward": raw, "reward_normalized": normalized,
            "image": filename, "sampler": "art_euler", "score_scale": "raw",
        })
        raw_scores.append(raw)
        logging.info("EVAL image=%d/%d reward=%.6f saved=%s", index + 1, len(prompts), raw, filename)
        del result, embeds, mask, negatives, negative_mask
    mean = sum(raw_scores) / len(raw_scores)
    summary = dict(count=len(raw_scores), reward_mean=mean, reward_mean_normalized=mean / 26,
                   sampler="art_euler", score_scale="raw")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    logging.info("EVAL count=%d reward_mean=%.6f reward_mean_normalized=%.6f images=%s sampler=art_euler score_scale=raw",
                 len(raw_scores), mean, mean / 26, output)


if __name__ == "__main__":
    main()
