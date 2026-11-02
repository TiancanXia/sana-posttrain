"""Single-GPU Sana ART presets adapted from the official SD3 PickScore recipe.

Each rollout supplies two disjoint optimizer updates. The ART algorithm options
match DiffusionART-main/configs/paper/image/sd3_pickscore_art.yaml; model,
precision, resolution, prompt count and physical microbatches are local choices.
Modified for the standalone release to use shared configurable asset paths.
"""

from ml_collections import ConfigDict

from config.paths import (
    PROJECT_ROOT, SANA_MODEL_DIR, PICKSCORE_MODEL_DIR,
    PICKSCORE_PROCESSOR_DIR, PROMPT_DATA_DIR,
)


def _get_config(*, name, prompts, candidates, generation_microbatch,
                training_microbatch, max_updates, checkpoint_every):
    config = ConfigDict()
    config.seed = 42
    config.upstream_recipe = "DiffusionART-main/configs/paper/image/sd3_pickscore_art.yaml"

    config.algorithm = ConfigDict()
    config.algorithm.name = "diffusionart"
    config.algorithm.restore_zero_signal = False
    config.algorithm.covariance = ConfigDict(dict(
        reduction="unbiased",
        direction_mode="historical",
        scale_mode="unit_normalized_covariance",
        guidance_gain=1.0,
        zero_signal_mode="exact",
        eps=1e-5,
    ))
    config.algorithm.loss = ConfigDict(dict(
        loss_space="velocity_linear_sigma",
        direction_normalization="rms",
        mask_space="velocity",
        velocity_mask_metric="scheduler_reference_metric",
        corrective_mask_enabled=True,
        q2_reference_sde_noise_level=0.8,
        q2_reference_kl_threshold=1e-5,
        eps=1e-5,
    ))

    config.model = ConfigDict()
    config.model.checkpoint = str(SANA_MODEL_DIR)
    config.model.lora_rank = 4
    config.model.lora_alpha = 4
    config.model.max_prompt_length = 300
    config.model.offload = True
    config.model.gradient_checkpointing = True

    config.reward = ConfigDict()
    config.reward.model_path = str(PICKSCORE_MODEL_DIR)
    config.reward.processor_path = str(PICKSCORE_PROCESSOR_DIR)
    config.reward.microbatch = 1

    config.data = ConfigDict()
    config.data.train_file = str(PROMPT_DATA_DIR / "train.txt")
    config.data.test_file = str(PROMPT_DATA_DIR / "test.txt")

    config.sampling = ConfigDict()
    config.sampling.steps = 10
    config.sampling.prompts_per_rollout = prompts
    config.sampling.samples_per_prompt = candidates
    config.sampling.microbatch = generation_microbatch
    # Decode separately to avoid holding four high-resolution VAE forwards.
    config.sampling.decode_microbatch = 1
    config.sampling.guidance_scale = 1.0
    config.sampling.height = 1024
    config.sampling.width = 1024
    config.sampling.train_timesteps = list(range(10))

    config.training = ConfigDict()
    config.training.max_updates = max_updates
    config.training.learning_rate = 3e-4
    # This counts (image, timestep) items. Partial final microbatches are valid:
    # the trainer weights their summed losses by the whole update's item count.
    config.training.microbatch = training_microbatch
    config.training.precision = "no"
    config.training.allow_tf32 = True
    config.training.grad_clip = 1.0
    config.training.optimizer = "adamw"
    config.training.weight_decay = 1e-4
    config.training.betas = [0.9, 0.999]
    config.training.epsilon = 1e-8

    config.runtime = ConfigDict()
    config.runtime.output_dir = str(PROJECT_ROOT / "outputs" / name)
    # Save only after both disjoint updates, at a complete rollout boundary.
    config.runtime.checkpoint_every = checkpoint_every
    config.runtime.resume_from = ""
    config.runtime.warm_start_lora = ""
    config.runtime.preview_images = 4
    config.runtime.max_no_signal_rollouts = 20

    config.evaluation = ConfigDict()
    config.evaluation.steps = 40
    config.evaluation.guidance_scale = 1.0
    config.evaluation.count = 16
    return config


def smoke():
    """One four-candidate prompt group; one rollout and two optimizer updates."""
    return _get_config(
        name="sana_art_pickscore_smoke", prompts=1, candidates=4,
        generation_microbatch=1, training_microbatch=1,
        max_updates=2, checkpoint_every=2,
    )


def pickscore():
    """Two 18-candidate groups; 15 rollouts and 30 optimizer updates."""
    return _get_config(
        name="sana_art_pickscore", prompts=2, candidates=18,
        generation_microbatch=4, training_microbatch=4,
        max_updates=30, checkpoint_every=10,
    )


def pickscore_fast():
    """Keep two prompt groups; increase physical microbatches for a new run."""
    return _get_config(
        name="sana_art_pickscore_fast", prompts=2, candidates=18,
        generation_microbatch=8, training_microbatch=18,
        max_updates=30, checkpoint_every=10,
    )


def pickscore_p8():
    """Eight prompt groups; 144 images per rollout and 72 per update."""
    config = _get_config(
        name="sana_art_pickscore_p8", prompts=8, candidates=18,
        generation_microbatch=8, training_microbatch=18,
        max_updates=30, checkpoint_every=10,
    )
    # This starts from the base model. To start a NEW experiment from an adapter,
    # set runtime.warm_start_lora to its .../checkpoint-N/lora directory.
    # Keep runtime.resume_from empty when changing sampling or microbatches.
    return config


def get_config(name="smoke"):
    presets = {"smoke": smoke, "pickscore": pickscore,
               "pickscore_fast": pickscore_fast, "pickscore_p8": pickscore_p8}
    if name not in presets:
        raise ValueError(f"Unknown Sana ART config: {name}; choose one of {', '.join(presets)}")
    return presets[name]()
