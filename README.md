# Sana PostTrain

Single-GPU post-training of **Sana-600M** with **DiffusionNFT** and **DiffusionART**, using **PickScore** as an in-process reward.

This repository provides a compact research implementation for collecting candidate images, scoring them, updating attention LoRA adapters, and evaluating saved checkpoints. It targets a single CUDA GPU with a 32 GB reference configuration. It is an independent adaptation, not an official NVIDIA or Kling AI release or a reproduction of the papers' large-scale experiments.

**License:** Apache-2.0 for the project adaptations, subject to retained third-party terms. In particular, the MixGRPO-derived NFT solver has academic-use restrictions. Read [Licensing and acknowledgements](#licensing-and-acknowledgements) before use.

## What is included

- **NFT:** group-relative rewards, forward-process optimization, a frozen old-policy adapter, and an optional EMA adapter.
- **ART:** explicit Euler rollouts, cached states and predictions, reward-dependent velocity targets, and two disjoint optimizer updates per rollout.
- **Shared components:** local Sana/PickScore loading, separate generation and training microbatches, checkpoint/resume state, per-prompt statistics, and fixed-seed evaluation.

The supported recipe is Sana-600M at 1024 x 1024 with PickScore and attention LoRA. Other base models, reward models and distributed training are outside the supported release scope. Model weights, prompt files, dependency source trees and experiment outputs are downloaded or generated locally; they are not included in Git.

## Installation

Use Linux, Python 3.10 or later, an NVIDIA GPU, and a CUDA-enabled PyTorch installation. Install a matching PyTorch/torchvision pair using the [official PyTorch instructions](https://pytorch.org/get-started/locally/); the scripts require PyTorch 2.6 or later. Keep sufficient CPU RAM for the offloaded Gemma text encoder, DC-AE and trajectory caches.

```bash
git clone https://github.com/TiancanXia/sana-posttrain.git
cd sana-posttrain

# Activate an environment with CUDA-enabled PyTorch and torchvision installed.
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` installs Diffusers 0.36.0 as a normal Python dependency. A local Diffusers source checkout, the native Sana repository, Cosmos-RL, Redis and a separate reward server are not required. The `flow_grpo` package is imported directly from this checkout by the entry scripts; it does not need a separate editable installation.

If editable Diffusers source is needed for development, install a separate checkout of the same version instead:

```bash
git clone --depth 1 --branch v0.36.0 \
  https://github.com/huggingface/diffusers.git /path/to/diffusers-0.36.0
python -m pip install -e /path/to/diffusers-0.36.0
```

No change to Diffusers' Sana implementation is required. This project does not use torchao quantization. If an unrelated, incompatible torchao installation causes an import error, use a clean environment without that optional package rather than copying dependency source patches into this repository.

## Download weights and prompts

Download these assets from their original sources:

| Asset | Source | Default local directory |
| --- | --- | --- |
| Sana-600M pipeline | [Efficient-Large-Model/Sana_600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers) | `models/Sana_600M_1024px_diffusers/` |
| PickScore weights | [yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1) | `reward_models/pickscore/model/` |
| PickScore processor | [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | `reward_models/pickscore/processor/` |
| PickScore train/test prompts | [DiffusionNFT prompt files](https://github.com/NVlabs/DiffusionNFT/tree/cbb14f84b8312b620390dfcbe2fab69c1383b104/dataset/pickscore) | `dataset/pickscore/` |

Review the upstream model and data terms before downloading. The helper below downloads only the selected Sana precision files, PickScore safetensors, processor files and checksum-verified prompts. It does not download the LAION CLIP model weights, duplicate precision variants, or run a model:

```bash
python scripts/download_assets.py

# Components can also be downloaded separately:
python scripts/download_assets.py --only model
python scripts/download_assets.py --only reward
python scripts/download_assets.py --only prompts
```

The Sana directory must contain the complete Diffusers pipeline, including `model_index.json`, `scheduler/`, `tokenizer/`, `text_encoder/`, `transformer/` and `vae/`. A standalone transformer checkpoint is insufficient. The loader accepts the selected FP16 transformer/DC-AE files and standard-named text encoder shards, then loads the base components in FP32 for the default recipe.

The prompt download is pinned to an upstream commit and verified by SHA256. Hub weight downloads default to `main`; pass `--sana-revision`, `--pickscore-revision` and `--processor-revision` to pin specific Hub commits for an experiment.

### Reuse existing assets or choose different paths

All asset paths are defined in [`config/paths.py`](config/paths.py). To reuse existing downloads, export these variables **before running training, evaluation or the download helper**:

```bash
export SANA_MODEL_DIR=/absolute/path/to/Sana_600M_1024px_diffusers
export PICKSCORE_MODEL_DIR=/absolute/path/to/pickscore/model
export PICKSCORE_PROCESSOR_DIR=/absolute/path/to/pickscore/processor
export PROMPT_DATA_DIR=/absolute/path/to/pickscore/prompts
```

`PROMPT_DATA_DIR` must contain `train.txt` and `test.txt`, with one nonempty prompt per line. Relative overrides are resolved from the repository root. Alternatively, edit the four defaults in `config/paths.py`; both NFT and ART configs use them. Training and evaluation load weights from local files only. The download helper refuses to overwrite different existing prompt files.

## Train NFT

Run commands from the repository root. First check a single update:

```bash
export OMP_NUM_THREADS=1
torchrun --standalone --nproc_per_node=1 \
  scripts/train_nft_sana.py --config config/sana.py:smoke
```

Then start the two-prompt recipe:

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_nft_sana.py --config config/sana.py:pickscore
```

This collects two prompts with 24 candidates each and performs one optimizer update per rollout. Generation and training microbatches are both 4; 12 image microbatches are accumulated per update. Nine training timesteps are selected per image. Checkpoints are saved every 10 updates and at the end.

NFT also includes `smoke_cfg45` and `pickscore_cfg45` as optional guided-training experiments. The default recipes train with CFG 1. Guidance above 1 applies the same guided field to collection and optimization, with gradients through both current-policy branches; this is an extension of the original CFG-free setup.

## Train ART

Check one rollout and its two updates:

```bash
export OMP_NUM_THREADS=1
torchrun --standalone --nproc_per_node=1 \
  scripts/train_art_sana.py --config config/art.py:smoke
```

Start the two-prompt recipe:

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_art_sana.py --config config/art.py:pickscore
```

Each rollout generates two prompts with 18 candidates each. The 36 images are split into two disjoint windows, with 18 images and 10 cached timesteps per optimizer update. `checkpoint-30` therefore means **30 optimizer updates from 15 rollouts**, not 30 rollouts. ART does not maintain NFT's old-policy/EMA training states.

### Default recipe comparison

| Setting | NFT `pickscore` | ART `pickscore` |
| --- | ---: | ---: |
| GPUs | 1 | 1 |
| Prompt groups per rollout | 2 | 2 |
| Candidates per prompt | 24 | 18 |
| Generated images per rollout | 48 | 36 |
| Optimizer updates per rollout | 1 | 2 |
| Images per optimizer update | 48 | 18 |
| Generation microbatch | 4 images | 4 images |
| Training microbatch | 4 images at one timestep | 4 image-timestep items |
| Training timesteps per image | 9 randomly selected | All 10 rollout slots |
| Training sampler | NFT `dpm2` | Explicit Euler |
| Sampling steps / resolution | 10 / 1024 x 1024 | 10 / 1024 x 1024 |
| Training CFG | 1 | 1 |
| LoRA rank / alpha | 4 / 4 | 4 / 4 |
| Precision | FP32 | FP32 |
| Total optimizer-update budget | 30 | 30 |
| Checkpoint interval | 10 updates | 10 updates |

These are small research runs. Microbatching changes peak memory; prompt count and candidates control the logical collection batch. Larger microbatches are not guaranteed to run faster or fit every 32 GB environment. ART's `pickscore_fast` and `pickscore_p8` presets provide larger microbatch/collection examples; start them as new experiments rather than changing an exact-resume configuration.

## Evaluate base and LoRA checkpoints

Evaluate the base first, then a trained adapter with the same prompts, seed, sampler, CFG, resolution and sampling steps. Evaluation takes the first `--num-images` prompts from the test file, one image per prompt. Use a new output directory for each evaluation.

NFT example:

```bash
python scripts/evaluation_sana.py \
  --sampler nft --guidance-scale 4.5 --steps 40 --seed 42 \
  --num-images 64 --output outputs/eval-nft/base

python scripts/evaluation_sana.py \
  --adapter outputs/sana_pickscore_cfg1_mb4/RUN/checkpoints/checkpoint-30/lora \
  --sampler nft --guidance-scale 4.5 --steps 40 --seed 42 \
  --num-images 64 --output outputs/eval-nft/step-30
```

ART example:

```bash
python scripts/evaluation_art_sana.py \
  --guidance-scale 4.5 --steps 40 --seed 42 \
  --num-images 64 --output outputs/eval-art/base

python scripts/evaluation_art_sana.py \
  --adapter outputs/sana_art_pickscore/RUN/checkpoints/checkpoint-30/lora \
  --guidance-scale 4.5 --steps 40 --seed 42 \
  --num-images 64 --output outputs/eval-art/step-30
```

Replace `RUN` with the timestamp printed by training. NFT can also evaluate `lora_ema/` instead of `lora/`; record which adapter was used. Evaluation CFG is explicit and independent of training CFG. NFT's evaluation additionally supports `--sampler native`; changing it changes the experiment.

Evaluations save PNG images, per-image `scores.jsonl`, `summary.json` and `evaluation_config.json`. Reported `reward_mean` uses raw PickScore units. Internally the reward is divided by 26, following NFT; `reward_mean_normalized` records that value. It is not a probability or a guaranteed 0-to-1 score.

Small fixed evaluations are useful for development, but do not establish general quality improvements. Compare paired per-image changes, inspect generated images, and use additional prompts/seeds for a final report. NFT and ART use different samplers and can have different base scores; compare each adapter with its corresponding base rather than ranking absolute scores alone.

## Checkpoints, logs and continuation

Training creates `outputs/<run-name>/<timestamp>/` containing `config.json`, `train.log`, `metrics.jsonl`, `group_metrics.jsonl`, image previews and `checkpoints/checkpoint-N/`. The console prints compact `METRIC` lines with loss, gradient norm, reward and measured LoRA changes. Previews are limited to four images per rollout; all collected candidates are used for training.

To keep console output in a separate file:

```bash
set -o pipefail
mkdir -p logs
torchrun --standalone --nproc_per_node=1 \
  scripts/train_nft_sana.py --config config/sana.py:pickscore \
  2>&1 | tee logs/nft-pickscore.log
```

For NFT continuation, set these fields inside `config/sana.py:pickscore()` before `return config`:

```python
config.num_epochs = 60
config.max_train_steps = 60
config.resume_from = str(
    PROJECT_ROOT / "outputs/sana_pickscore_cfg1_mb4/RUN/checkpoints/checkpoint-30"
)
```

For ART continuation, assign the recipe to `config` in `config/art.py:pickscore()`, then set:

```python
config.training.max_updates = 60
config.runtime.resume_from = str(
    PROJECT_ROOT / "outputs/sana_art_pickscore/RUN/checkpoints/checkpoint-30"
)
config.runtime.warm_start_lora = ""
return config
```

Rerun the same training command. Budgets are totals: extending 30 to 60 requests 30 additional updates. Resume from the complete `checkpoint-N/` directory, not its `lora/` subdirectory. ART budgets and save intervals must be even, and ART resumes only after a complete two-update rollout boundary. Interrupted work after the last complete checkpoint must be repeated.

Resume checks protect algorithm/config/data/schedule identity, but do not hash every weight file or guarantee identical behavior across dependency or hardware changes. Keep paths, weights and dependency versions fixed. NFT and ART continuation states are not interchangeable. To start a new ART experiment from an adapter only, set `runtime.warm_start_lora` to its compatible `lora/` directory and leave `resume_from` empty; this does not restore NFT optimizer or EMA state.

Monitor GPU use in another terminal:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1
```

## Layout

```text
sana-posttrain/
  config/                     # NFT/ART presets and shared asset paths
  scripts/                    # Download, train and evaluation entry points
  flow_grpo/                  # NFT helpers, Sana adapter and ART kernels
  LICENSES/                   # Additional upstream license texts
  requirements.txt
  constraints.txt
  LICENSE
  NOTICE
  THIRD_PARTY_NOTICES.md
  SOURCE_MANIFEST.json
```

No pretrained models, trained adapters, prompt text, Diffusers source, private development notes or tests are distributed in this release. The release preparation uses static inspection; the standalone installation and changed asset-path setup still need a fresh-environment smoke run. The reference training experiments were run in the maintainer's existing environment, not as a clean-install certification of this package.

## Licensing and acknowledgements

This implementation builds on the following work:

- [NVIDIA Sana](https://github.com/NVlabs/Sana), the Efficient-Large-Model team and collaborators: the base text-to-image model and its Diffusers integration.
- [NVIDIA DiffusionNFT](https://github.com/NVlabs/DiffusionNFT): the NFT training flow, solver integration, reward statistics and EMA helpers.
- [DiffusionART](https://github.com/vanzll/DiffusionART), by contributors from NUS HPC-AI Lab, Kling AI and HKUST: ART loss/covariance kernels, disjoint-window construction and rollout conventions.
- [Flow-GRPO](https://github.com/yifan123/flow_grpo), [MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [PEFT](https://github.com/huggingface/peft), and [PickScore](https://github.com/yuvalkirstain/PickScore) for the underlying research and tooling.

The project adaptations use Apache-2.0; upstream copyright and permission notices remain in place. **The inherited NFT solver is derived from MixGRPO and remains subject to its separate license, including academic-use restrictions. This repository is not uniformly Apache-2.0 and does not grant commercial rights to that solver.** See [LICENSE](LICENSE), [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/MixGRPO.txt](LICENSES/MixGRPO.txt) for the applicable scope.

Model weights, the Gemma text encoder, processor assets and prompt datasets retain their own terms; this repository's license does not relicense them. The [Sana model card](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers) links the applicable Gemma terms. Download and review external assets from their original sources. Source provenance and release modifications are recorded in `SOURCE_MANIFEST.json` and `flow_grpo/art/SOURCE.json`.
