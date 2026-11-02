"""Sana interfaces around the unchanged DiffusionNFT sampling solver.

Local adaptation of NVlabs/DiffusionNFT's pipeline_with_logprob.py.
Sampling uses NFT's dpm2 solver and the checkpoint's flow sigma schedule.
This module does not replace or monkey-patch the installed Diffusers library.
"""

import inspect
import math
from pathlib import Path

import torch
from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from .solver import run_sampling


def load_pipeline(model_path, device, offload=True):
    """Load existing mixed-variant local weights; train base weights in FP32."""
    model_path = Path(model_path).expanduser().resolve()
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"Missing Sana model_index.json: {model_path}")
    components = {}
    for name, cls in (("transformer", SanaTransformer2DModel), ("vae", AutoencoderDC)):
        variant = (
            {"variant": "fp16"}
            if (model_path / name / "diffusion_pytorch_model.fp16.safetensors").is_file()
            else {}
        )
        components[name] = cls.from_pretrained(
            str(model_path), subfolder=name, torch_dtype=torch.float32,
            local_files_only=True, **variant,
        )
    pipe = SanaPipeline.from_pretrained(
        str(model_path), torch_dtype=torch.float32, local_files_only=True, **components,
    )
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.to(device)
    pipe.text_encoder.to("cpu" if offload else device)
    pipe.vae.to("cpu" if offload else device, dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    scheduler = pipe.scheduler.config
    if scheduler.get("prediction_type") != "flow_prediction" or not scheduler.get("use_flow_sigmas"):
        raise ValueError("This adapter requires Sana flow_prediction with use_flow_sigmas=True")
    if int(scheduler.num_train_timesteps) != 1000:
        raise ValueError("NFT timestep normalization here expects num_train_timesteps=1000")
    return pipe


@torch.no_grad()
def encode_prompts(pipe, prompts, device, max_length=300, offload=True):
    """Original two-result, conditional-only interface used by NFT training."""
    return encode_prompt_conditions(pipe, prompts, device, max_length, offload)[:2]


@torch.no_grad()
def encode_prompt_conditions(pipe, prompts, device, max_length=300, offload=True,
                             guidance_scale=1.0):
    """Encode distinct prompts once and expand masks/embeddings in the same order."""
    if not math.isfinite(guidance_scale) or guidance_scale < 1:
        raise ValueError("guidance_scale must be finite and at least 1")
    unique_prompts = list(dict.fromkeys(prompts))
    if not unique_prompts:
        raise ValueError("Cannot encode an empty prompt batch")
    prompt_index = {prompt: i for i, prompt in enumerate(unique_prompts)}
    inverse = torch.tensor([prompt_index[p] for p in prompts], device=device)
    # Match SanaPipeline.__call__ conditioning, not encode_prompt's None default.
    instruction = inspect.signature(SanaPipeline.__call__).parameters[
        "complex_human_instruction"
    ].default
    pipe.text_encoder.to(device)
    try:
        embeds, mask, negative_embeds, negative_mask = pipe.encode_prompt(
            prompt=unique_prompts, do_classifier_free_guidance=guidance_scale > 1,
            negative_prompt="",
            num_images_per_prompt=1, device=device, clean_caption=False,
            max_sequence_length=max_length, complex_human_instruction=instruction,
        )
        return (
            embeds.index_select(0, inverse), mask.index_select(0, inverse),
            negative_embeds.index_select(0, inverse) if negative_embeds is not None else None,
            negative_mask.index_select(0, inverse) if negative_mask is not None else None,
        )
    finally:
        if offload:
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()


def predict_velocity(model, latents, timesteps, prompt_embeds, prompt_attention_mask):
    """Use the same Sana timestep scale in training and sampling (DDP-safe)."""
    underlying = model.module if hasattr(model, "module") else model
    dtype = next(underlying.parameters()).dtype
    return model(
        hidden_states=latents.to(dtype=dtype),
        timestep=timesteps * underlying.config.timestep_scale,
        encoder_hidden_states=prompt_embeds.to(dtype=dtype),
        encoder_attention_mask=prompt_attention_mask,
        return_dict=False,
    )[0].float()


def predict_guided_velocity(model, latents, timesteps, prompt_embeds, prompt_attention_mask,
                            *, guidance_scale=1.0, negative_prompt_embeds=None,
                            negative_prompt_attention_mask=None, concatenate=False):
    """Cosmos Sana's u + g*(c-u), with gradients through both current branches.

    Inference defaults to sequential forwards. Training uses concatenate=True
    so each policy has one DDP forward for its paired unconditional/conditional
    batch. CFG=1 always preserves the original single conditional call.
    """
    if not math.isfinite(guidance_scale) or guidance_scale < 1:
        raise ValueError("guidance_scale must be finite and at least 1")
    if guidance_scale == 1:
        return predict_velocity(model, latents, timesteps, prompt_embeds, prompt_attention_mask)
    if negative_prompt_embeds is None or negative_prompt_attention_mask is None:
        raise ValueError("CFG>1 requires unconditional embeddings and attention mask")
    batch_size = latents.shape[0]
    if any(value.shape[0] != batch_size for value in (
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask,
    )):
        raise ValueError("Conditional/unconditional batch sizes must match latents")
    if concatenate:
        prediction = predict_velocity(
            model, torch.cat([latents, latents], dim=0),
            torch.cat([timesteps, timesteps], dim=0),
            torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
            torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0),
        )
        unconditional, conditional = prediction.chunk(2, dim=0)
        return unconditional + guidance_scale * (conditional - unconditional)
    unconditional = predict_velocity(
        model, latents, timesteps, negative_prompt_embeds, negative_prompt_attention_mask,
    )
    conditional = predict_velocity(model, latents, timesteps, prompt_embeds, prompt_attention_mask)
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.no_grad()
def sample_images(
    pipe, prompt_embeds, prompt_attention_mask, *, height, width, steps=10,
    solver="dpm2", generator=None, offload=True, guidance_scale=1.0,
    negative_prompt_embeds=None, negative_prompt_attention_mask=None,
    sampling_backend="nft",
):
    """Return BCHW images in [0,1], final clean latents, and raw model times.

    Training keeps deterministic NFT DPM1/2 and explicitly passes its CFG.
    The native scheduler backend is available for evaluation diagnostics.
    """
    if solver not in ("dpm1", "dpm2"):
        raise ValueError("This Sana adapter supports deterministic dpm1/dpm2 only")
    if sampling_backend not in ("nft", "native"):
        raise ValueError("sampling_backend must be nft or native")
    if not math.isfinite(guidance_scale) or guidance_scale < 1:
        raise ValueError("guidance_scale must be finite and at least 1")
    if guidance_scale > 1 and (negative_prompt_embeds is None or negative_prompt_attention_mask is None):
        raise ValueError("CFG>1 requires unconditional embeddings and attention mask")
    scale = pipe.vae_scale_factor
    if height % scale or width % scale:
        raise ValueError(f"Image dimensions must be divisible by DC-AE scale {scale}")
    device = next(pipe.transformer.parameters()).device
    prompt_embeds = prompt_embeds.to(device)
    prompt_attention_mask = prompt_attention_mask.to(device)
    if guidance_scale > 1:
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)
    pipe.scheduler.set_timesteps(steps, device=device)
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
    if len(sigmas) != steps + 1 or len(pipe.scheduler.timesteps) != steps:
        raise ValueError("Unexpected Sana flow sigma/timestep schedule length")
    # NFT's solver casts float32 sigma*1000 to integer. Reuse precisely those
    # times for training; native scheduler integer rounding can differ by 1.
    timesteps = (sigmas[:-1] * 1000).long()
    shape = (len(prompt_embeds), pipe.transformer.config.in_channels, height // scale, width // scale)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

    def velocity(latent, sigma):
        # Preserve original NFT's integer model-time convention.
        times = (sigma * 1000).long().expand(latent.shape[0]).to(device)
        return predict_guided_velocity(
            pipe.transformer, latent, times, prompt_embeds, prompt_attention_mask,
            guidance_scale=guidance_scale, negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

    if sampling_backend == "nft":
        clean, _, _ = run_sampling(
            velocity, latents, sigmas, solver=solver, determistic=True, eta=0.0,
        )
    else:
        # Follow Cosmos Sana's generation loop: model time from the scheduler,
        # the same CFG combination, then the checkpoint scheduler's step().
        # Unlike NFT, do not replace scheduler times by floor(float32 sigma*1000).
        timesteps = pipe.scheduler.timesteps.to(device)
        for timestep in timesteps:
            prediction = predict_guided_velocity(
                pipe.transformer, latents, timestep.expand(latents.shape[0]),
                prompt_embeds, prompt_attention_mask, guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )
            latents = pipe.scheduler.step(prediction, timestep, latents, return_dict=False)[0]
        clean = latents
    # No SD3 shift_factor: DC-AE uses only its own scaling_factor.
    pipe.vae.to(device, dtype=torch.float32)
    try:
        with torch.autocast("cuda", enabled=False):
            decoded = pipe.vae.decode(clean.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(decoded, output_type="pt")
        return images, clean.detach(), timesteps
    finally:
        if offload:
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
