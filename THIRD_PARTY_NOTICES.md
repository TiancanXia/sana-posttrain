# Third-party notices

This repository integrates third-party research code and separately installed
model libraries. The root `LICENSE.md` contains the Apache License, Version 2.0.
It does not replace the terms of the third-party portions identified below.
In particular, the inherited NFT solver has MixGRPO provenance and must not be
presented as carrying unrestricted Apache-2.0 permissions.

## NVIDIA DiffusionNFT

Project: [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT).

The NFT trainer, configuration, sampling solver, EMA, statistics, and reward
integration were copied or adapted from the downloaded DiffusionNFT source
snapshot. The original download did not record a repository commit; the source
inventory records file hashes instead. A current upstream commit must not be
substituted for that unknown snapshot revision.

DiffusionNFT's project license is Apache-2.0, reproduced in the root `LICENSE.md`.
The inherited NVIDIA training-file copyright notice is retained in its source:

> Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The upstream Apache license can be inspected at
[commit d8c606d55be8d90b12d8aa0c3c75e15876aacd41](https://github.com/NVlabs/DiffusionNFT/blob/d8c606d55be8d90b12d8aa0c3c75e15876aacd41/LICENSE).
That reference identifies the inspected license, not the downloaded code snapshot.
DiffusionNFT also acknowledges Flow-GRPO, and its solver explicitly states that it
was modified from MixGRPO. Those underlying terms and notices remain relevant.

## MixGRPO: inherited NFT solver exception

File: `flow_grpo/diffusers_patch/solver.py`.

This file was copied from the downloaded DiffusionNFT snapshot and retains its
`Modified from MixGRPO` attribution. The original source is
[Tencent-Hunyuan/MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO).
The inspected upstream sampling implementation is
[`fastvideo/utils/sampling_utils.py` at 19946facaefa33eca1b565ec87c6fc7cca14e339](https://github.com/Tencent-Hunyuan/MixGRPO/blob/19946facaefa33eca1b565ec87c6fc7cca14e339/fastvideo/utils/sampling_utils.py).
This comparison reference is not asserted to be the exact version used by
DiffusionNFT.

MixGRPO's license identifies Copyright (C) 2025 Tencent and permits use only for
academic purposes, excluding commercial and production purposes. The complete
upstream license, including its third-party component terms, is preserved
without editing in [LICENSES/MixGRPO.txt](LICENSES/MixGRPO.txt). It was retrieved
from [the original license at commit 75c797fc453b0bbd4444aad984bfcee51faa965f](https://github.com/Tencent-Hunyuan/MixGRPO/blob/75c797fc453b0bbd4444aad984bfcee51faa965f/License.txt);
The inspected license file has been unchanged since that commit.

The exact copyright lineage of every solver fragment has not been resolved.
Pending that resolution, this release treats the inherited solver as subject to
the applicable MixGRPO restrictions. Neither the root Apache license nor
DiffusionNFT's Apache license is a statement that Tencent-covered portions have
been relicensed. Components that originated under separate licenses retain those
licenses. This repository includes no FLUX model or MixGRPO checkpoint.

## Flow-GRPO

Project: [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo).

DiffusionNFT's implementation is based on Flow-GRPO. The inherited training,
EMA, statistics, and related utility lineage is acknowledged here.

Copyright (c) 2025 Jie Liu.

The full MIT copyright, permission, and warranty notice is retained in
[LICENSES/Flow-GRPO-MIT.txt](LICENSES/Flow-GRPO-MIT.txt), retrieved from
[the upstream license at 39d3d2c68fb0197290bfe1ffa8cd59f2cd459a78](https://github.com/yifan123/flow_grpo/blob/39d3d2c68fb0197290bfe1ffa8cd59f2cd459a78/LICENSE).
The MIT terms continue to apply to the corresponding portions.

## DiffusionART

Project: [vanzll/DiffusionART](https://github.com/vanzll/DiffusionART).

The ART tensor kernels in `flow_grpo/art/art.py`,
`flow_grpo/art/covariance.py`, and `flow_grpo/art/_validation.py` were copied
from the downloaded DiffusionART source. `flow_grpo/art/windows.py` extracts
the prompt-window function, while `flow_grpo/art/sampling.py` adapts its Euler
sampling and frozen-trajectory conventions for Sana. The ART configuration and
training entry point implement that integration. File-level provenance is
recorded in `flow_grpo/art/SOURCE.json` and `flow_grpo/art/SOURCE.md`.
No exact repository commit was recorded for the downloaded ART snapshot.

The original notice is retained in `flow_grpo/art/NOTICE`:

> DiffusionART
> Copyright 2026 DiffusionART contributors

DiffusionART's original project code is Apache-2.0. Its accompanying
`flow_grpo/art/LICENSE` is retained, along with the applicable attribution
notices. The official [project page](https://vanzll.github.io/diffusionart-project/)
and [paper](https://arxiv.org/html/2610.12572v1) identify collaboration among
HPC-AI Lab at the National University of Singapore, Kling AI, and HKUST.
These research affiliations are credited without changing the upstream
copyright statement or implying endorsement of this integration.

The downloaded upstream project also distributes other algorithms, reward
adapters, wheels, fonts, and video integrations. Those components are not
included in this source release; this notice covers the included portions.

## NVIDIA Sana and separately installed libraries

Project: [NVlabs/Sana](https://github.com/NVlabs/Sana).

Sana supplies the base model architecture and model family used by this
integration. Its [source license](https://github.com/NVlabs/Sana/blob/main/LICENSE)
is Apache-2.0. This repository uses Sana interfaces supplied by separately
installed [Hugging Face Diffusers](https://github.com/huggingface/diffusers);
it does not bundle the native Sana source tree, Diffusers source tree, base-model
weights, tokenizer, text encoder, or autoencoder assets.

The root source-code license does not license those separately obtained assets.
Users must obtain the selected checkpoint and its component assets from their
original publishers and follow the applicable model and component terms.

## PickScore and prompt data

Projects: [PickScore](https://github.com/yuvalkirstain/PickScore),
[PickScore_v1 model](https://huggingface.co/yuvalkirstain/PickScore_v1), and
[Pick-a-Pic](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1).

The included `flow_grpo/pickscore_scorer.py` adapts the inherited DiffusionNFT
wrapper to load separately obtained PickScore and processor assets and apply the
published feature-similarity scoring formula. No PickScore weights, processor
assets, Pick-a-Pic images, preference labels, or copied prompt splits are
redistributed in this repository. Prompt data must be obtained separately.

The upstream PickScore code license is MIT. For any portions of its published
example code incorporated into the scorer, its original notice is reproduced
below exactly as published in
[PickScore/LICENSE](https://github.com/yuvalkirstain/PickScore/blob/main/LICENSE):

```text
MIT License

Copyright (c) 2021

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Model, processor, and dataset terms must be checked independently; the code
license and this attribution do not authorize redistribution of those assets.

## Project names

NVIDIA, Sana, DiffusionNFT, DiffusionART, Kling AI, MixGRPO, and the other project
names above are used to describe the origins and dependencies of this work.
No upstream affiliation, sponsorship, endorsement, or trademark license is claimed.
