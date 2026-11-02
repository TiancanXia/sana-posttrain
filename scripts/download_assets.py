# SPDX-License-Identifier: Apache-2.0
"""Download external weights and verified prompt files; no model code is run."""

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.paths import (
    SANA_MODEL_DIR, PICKSCORE_MODEL_DIR, PICKSCORE_PROCESSOR_DIR, PROMPT_DATA_DIR,
)

PROMPT_REVISION = "cbb14f84b8312b620390dfcbe2fab69c1383b104"
PROMPT_HASHES = {
    "train.txt": "2f425adcbb9a6d0feee7c6c448e1cb83a1f71deca9434b803907e2578eca837f",
    "test.txt": "d75c025fa451866e70f7e36243f7cdd93095523cb51f5f9651f2ce3236d20262",
}


def download_prompts():
    """Require the same prompt bytes as the reference experiments."""
    PROMPT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected in PROMPT_HASHES.items():
        target = PROMPT_DATA_DIR / filename
        if target.exists():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Refusing to replace different prompts: {target}")
            print(f"Verified existing prompts: {target}")
            continue
        url = f"https://raw.githubusercontent.com/NVlabs/DiffusionNFT/{PROMPT_REVISION}/dataset/pickscore/{filename}"
        request = Request(url, headers={"User-Agent": "sana-posttrain-asset-download"})
        with urlopen(request, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"Upstream prompt checksum changed; inspect before use: {url}")
        # Exclusive creation avoids overwriting a concurrently supplied prompt file.
        with target.open("xb") as handle:
            handle.write(content)
        print(f"Downloaded and verified prompts: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "model", "reward", "prompts"), default="all")
    parser.add_argument("--sana-revision", default="main", help="Optional pinned Sana Hub commit")
    parser.add_argument("--pickscore-revision", default="main", help="Optional pinned PickScore Hub commit")
    parser.add_argument("--processor-revision", default="main", help="Optional pinned processor Hub commit")
    args = parser.parse_args()
    if args.only in ("all", "model", "reward"):
        from huggingface_hub import snapshot_download

        if args.only in ("all", "model"):
            snapshot_download(
                repo_id="Efficient-Large-Model/Sana_600M_1024px_diffusers",
                revision=args.sana_revision, local_dir=str(SANA_MODEL_DIR),
                allow_patterns=[
                    "model_index.json", "scheduler/*.json", "tokenizer/*",
                    "text_encoder/config.json", "text_encoder/model-*.safetensors",
                    "text_encoder/model.safetensors.index.json",
                    "transformer/config.json", "transformer/diffusion_pytorch_model.fp16.safetensors",
                    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
                    "LICENSE*", "README.md",
                ],
            )
        if args.only in ("all", "reward"):
            snapshot_download(
                repo_id="yuvalkirstain/PickScore_v1", revision=args.pickscore_revision,
                local_dir=str(PICKSCORE_MODEL_DIR),
                allow_patterns=["config.json", "model.safetensors", "LICENSE*", "README.md"],
            )
            snapshot_download(
                repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", revision=args.processor_revision,
                local_dir=str(PICKSCORE_PROCESSOR_DIR),
                allow_patterns=[
                    "config.json", "preprocessor_config.json", "special_tokens_map.json",
                    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "LICENSE*", "README.md",
                ],
            )
    if args.only in ("all", "prompts"):
        download_prompts()
    print("Assets prepared. Review their upstream licenses and keep weights fixed during an experiment.")


if __name__ == "__main__":
    main()
