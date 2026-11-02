# SPDX-License-Identifier: Apache-2.0
"""Shared local asset paths for training, evaluation and optional downloads."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def asset_path(variable, default):
    """Resolve an environment override; relative values are relative to this repo."""
    path = Path(os.environ.get(variable, str(default))).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


SANA_MODEL_DIR = asset_path("SANA_MODEL_DIR", PROJECT_ROOT / "models/Sana_600M_1024px_diffusers")
PICKSCORE_MODEL_DIR = asset_path("PICKSCORE_MODEL_DIR", PROJECT_ROOT / "reward_models/pickscore/model")
PICKSCORE_PROCESSOR_DIR = asset_path("PICKSCORE_PROCESSOR_DIR", PROJECT_ROOT / "reward_models/pickscore/processor")
PROMPT_DATA_DIR = asset_path("PROMPT_DATA_DIR", PROJECT_ROOT / "dataset/pickscore")
