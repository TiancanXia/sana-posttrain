"""Evaluate a local Sana base/standard PEFT LoRA with fixed prompts and seeds."""

import argparse
import importlib.util
import logging
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from peft import PeftModel
from flow_grpo.diffusers_patch.pipeline_sana import load_pipeline
from flow_grpo.rewards import multi_score
# Reuse the exact training evaluation path, not a second model-time convention.
from train_nft_sana import TextPromptDataset, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, help="This port's checkpoint/lora or checkpoint/lora_ema directory")
    parser.add_argument("--output", type=Path, required=True, help="A NEW output directory")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="Evaluation CFG; >=1. Pass 4.5 for the *_cfg45 training presets.")
    parser.add_argument("--sampler", choices=("nft", "native"), default="nft",
                        help="NFT solver or the checkpoint's native scheduler.step loop.")
    args = parser.parse_args()
    if args.num_images < 1 or args.steps < 1:
        parser.error("num-images and steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error("guidance-scale must be finite and at least 1")
    if args.output.exists():
        parser.error("Choose a new output directory to preserve previous comparisons")
    spec = importlib.util.spec_from_file_location("sana_config", ROOT / "config/sana.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.smoke()
    config.eval_max_samples = args.num_images
    config.seed = args.seed
    config.sample.eval_num_steps = args.steps
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda:0")
    pipe = load_pipeline(config.pretrained.model, device, config.model_offload)
    if args.adapter:
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, str(args.adapter.resolve()), is_trainable=False, local_files_only=True,
        )
    scorer = multi_score(device, config.reward_fn, pickscore_model_path=config.reward_model_path,
                         pickscore_processor_path=config.reward_processor_path)
    mean = evaluate(
        pipe, TextPromptDataset(config.dataset, "test"), scorer, config, device, args.output.resolve(),
        guidance_scale=args.guidance_scale, sampling_backend=args.sampler,
        adapter_label=str(args.adapter.resolve()) if args.adapter else "base",
    )
    logging.info("EVAL adapter=%s sampler=%s cfg=%.2f reward_mean=%.6f score_scale=raw",
                 str(args.adapter.resolve()) if args.adapter else "base", args.sampler,
                 args.guidance_scale, mean)


if __name__ == "__main__":
    main()
