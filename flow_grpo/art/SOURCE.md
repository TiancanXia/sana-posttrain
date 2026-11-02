<!-- SPDX-License-Identifier: Apache-2.0 -->

`art.py`, `covariance.py`, and `_validation.py` are byte-for-byte copies of
`DiffusionART-main/src/visual_rl/algorithms/` from the local reference repository.
`windows.py` extracts `prompt_disjoint_windows` from
`DiffusionART-main/src/visual_rl/training/objectives.py` without changing its body.
The reference is Copyright 2026 DiffusionART contributors and Apache-2.0 licensed;
its accompanying `LICENSE` and `NOTICE` are retained here.

`sampling.py` adapts the reference's explicit Euler ODE and pre-update CPU cache
convention (`src/visual_rl/sampling.py`, `training/rollout.py`, `models/sd3.py`) to
Sana. Sana's own scheduler defines the sigma grid, so that grid need not equal
SD3's shifted flow schedule. Time remains floating `sigma * 1000`, with Sana's
`timestep_scale` applied only inside the existing `predict_guided_velocity`.
There is no velocity sign flip, integer time conversion, SD3 VAE shift, or call
to the NFT DPM2 solver. Decoding uses DC-AE's scaling factor in FP32 and transfers
each small image batch immediately to CPU.
