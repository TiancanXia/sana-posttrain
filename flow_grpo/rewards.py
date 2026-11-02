# SPDX-License-Identifier: Apache-2.0
# Adapted from NVlabs/DiffusionNFT/flow_grpo/rewards.py.
# The release keeps only the PickScore route; scoring units are unchanged.
"""In-process PickScore reward with the original multi_score call contract."""

import numpy as np
from PIL import Image
import torch

from flow_grpo.pickscore_scorer import PickScoreScorer


def multi_score(device, score_dict, *, pickscore_model_path=None,
                pickscore_processor_path=None, pickscore_local_files_only=True,
                pickscore_batch_size=1):
    if set(score_dict) != {"pickscore"}:
        raise ValueError("This release supports only reward_fn={'pickscore': weight}")
    scorer = PickScoreScorer(
        device=device, dtype=torch.float32, model_path=pickscore_model_path,
        processor_path=pickscore_processor_path,
        local_files_only=pickscore_local_files_only, batch_size=pickscore_batch_size,
    )
    weight = float(score_dict["pickscore"])

    def score(images, prompts, metadata, only_strict=True):
        if isinstance(images, torch.Tensor):
            pixels = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = [Image.fromarray(image) for image in pixels.transpose(0, 2, 3, 1)]
        elif isinstance(images, np.ndarray):
            images = [Image.fromarray(image) for image in images]
        values = scorer(prompts, images)
        return {"pickscore": values, "avg": [weight * value for value in values]}, {}

    return score
