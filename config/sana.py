"""Single-GPU Sana presets adapted from NFT; standalone release asset paths."""

import importlib.util
from config.paths import (
    PROJECT_ROOT, SANA_MODEL_DIR, PICKSCORE_MODEL_DIR,
    PICKSCORE_PROCESSOR_DIR, PROMPT_DATA_DIR,
)
_spec = importlib.util.spec_from_file_location("sana_nft_base_config", PROJECT_ROOT / "config/base.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def _get_config(group_size, max_train_steps, name):
    config = base.get_config()
    config.project_root = str(PROJECT_ROOT)
    config.base_model = "sana"
    config.run_name = name
    config.dataset = str(PROMPT_DATA_DIR)
    config.pretrained.model = str(SANA_MODEL_DIR)
    config.reward_model_path = str(PICKSCORE_MODEL_DIR)
    config.reward_processor_path = str(PICKSCORE_PROCESSOR_DIR)
    config.reward_fn = {"pickscore": 1.0}
    config.prompt_fn = "general_ocr"
    config.resolution = 1024
    config.max_prompt_length = 300
    config.model_offload = True
    config.gradient_checkpointing = True
    # Deliberate resource setting, NOT upstream NFT's rank 32 / alpha 64.
    config.lora_rank = 4
    config.lora_alpha = 4
    config.mixed_precision = "no"
    config.num_epochs = max_train_steps
    config.max_train_steps = max_train_steps
    config.save_freq = 1
    config.eval_freq = 0
    config.eval_max_samples = 16
    config.save_dir = str(PROJECT_ROOT / "outputs" / name)
    config.logdir = str(PROJECT_ROOT / "logs")
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.global_std = True
    # Logical batch keeps the original K-repeat sampler's groups intact.
    config.sample.train_batch_size = group_size
    config.sample.num_image_per_prompt = group_size
    config.sample.num_batches_per_epoch = 1
    config.sample.generation_batch_size = 1
    config.sample.test_batch_size = 1
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = group_size
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.0001
    config.train.ema = True
    config.train.adv_mode = "all"
    config.beta = 1.0
    config.decay_type = 1
    return config


def smoke():
    """One group of four images; one optimizer update."""
    return _get_config(4, 1, "sana_pickscore_smoke")


def pickscore():
    """CFG=1: two 24-candidate groups, microbatch four, save every ten updates."""
    config = _get_config(24, 30, "sana_pickscore_cfg1_mb4")
    config.sample.guidance_scale = 1.0
    # Two prompt groups per round; keep 24 candidates for each prompt.
    config.sample.train_batch_size = 48
    config.sample.generation_batch_size = 4
    config.train.batch_size = 4
    # 4 images x 12 microbatches = 48 images and one optimizer update per round.
    config.train.gradient_accumulation_steps = 12
    # One optimizer update per round; save at steps 10, 20 and 30.
    config.save_freq = 10

    # Optional continuation: set both total budgets and the complete checkpoint path.
    # config.num_epochs = 60
    # config.max_train_steps = 60
    # config.resume_from = str(PROJECT_ROOT / "outputs/sana_pickscore_cfg1_mb4/RUN/checkpoints/checkpoint-30")

    return config


def smoke_cfg45():
    """One guided NFT update; rollout and all loss fields use CFG=4.5."""
    config = _get_config(4, 1, "sana_pickscore_cfg45_smoke")
    config.sample.guidance_scale = 4.5
    return config


def pickscore_cfg45():
    """Ten guided updates, microbatch two, checkpoint every two rounds/updates."""
    config = _get_config(24, 10, "sana_pickscore_cfg45")
    config.sample.guidance_scale = 4.5
    config.sample.generation_batch_size = 2
    config.train.batch_size = 2
    # Keep 24 images per optimizer update: 2 images x 12 accumulation batches.
    config.train.gradient_accumulation_steps = 12
    # One round is one optimizer update in this preset; the final save is forced.
    config.save_freq = 2
    return config


def get_config(name="smoke"):
    configs = {"smoke": smoke, "pickscore": pickscore,
               "smoke_cfg45": smoke_cfg45, "pickscore_cfg45": pickscore_cfg45}
    if name not in configs:
        raise ValueError(f"Unknown Sana config: {name}; choose one of {', '.join(configs)}")
    return configs[name]()
