# SPDX-License-Identifier: Apache-2.0
"""Sana Euler rollouts using DiffusionART's frozen-trajectory convention.

Sources: DiffusionART-main/src/visual_rl/{sampling.py,training/rollout.py,
models/sd3.py}; Sana conditioning/decoding reuse the adjacent pipeline_sana.
The model-specific sigma grid comes from Sana's own scheduler. Its velocity has
the same orientation as SD3: x_next = x + (sigma_next - sigma) * velocity.
Model time is floating sigma * 1000; predict_guided_velocity applies Sana's
timestep_scale exactly once. The NFT sampling solver is not used here.
"""

import math

import torch
from diffusers.utils.torch_utils import randn_tensor

from ..diffusers_patch.pipeline_sana import predict_guided_velocity


def _validate_conditioning(embeds, mask, name):
    if (
        not isinstance(embeds, torch.Tensor)
        or embeds.ndim != 3
        or not embeds.is_floating_point()
        or not all(embeds.shape)
        or not torch.isfinite(embeds).all()
    ):
        raise ValueError(f"{name}_embeds must be a finite nonempty [B, L, D] float tensor")
    if (
        not isinstance(mask, torch.Tensor)
        or mask.shape != embeds.shape[:2]
        or not torch.isfinite(mask).all()
        or not ((mask == 0) | (mask == 1)).all()
    ):
        raise ValueError(f"{name}_attention_mask must be a finite binary [B, L] tensor")


def _validate_generator(generator, batch_size, device):
    if generator is None:
        return
    generators = generator if isinstance(generator, list) else [generator]
    if isinstance(generator, list) and len(generators) != batch_size:
        raise ValueError("a generator list must contain one generator per image")
    generator_device = None
    for item in generators:
        if not isinstance(item, torch.Generator):
            raise TypeError("generator must be a torch.Generator or a list of generators")
        source = item.device
        if source.type != "cpu" and source != device:
            raise ValueError("generators must be on CPU or the transformer's device")
        if generator_device is not None and source != generator_device:
            raise ValueError("all generators in a list must use the same device")
        generator_device = source


@torch.no_grad()
def sample_art_trajectories(
    pipe,
    prompt_embeds,
    prompt_attention_mask,
    *,
    height,
    width,
    steps=10,
    generator=None,
    offload=True,
    guidance_scale=1.0,
    negative_prompt_embeds=None,
    negative_prompt_attention_mask=None,
    decode_batch_size=1,
    cache_trajectory=True,
):
    """Return CPU FP32 caches from pre-update states and their old velocities.

    ``states`` and ``old_velocities`` have shape [B, T, C, h, w], endpoints have
    shape [B, C, h, w], and images have shape [B, 3, height, width] in [0, 1].
    ``sigmas``, negative ``dt`` and floating ``timesteps`` contain T entries.
    Evaluation may set ``cache_trajectory=False`` to return None for both caches.
    A CPU generator is supported for CUDA sampling through randn_tensor.
    The VAE stays on CPU during denoising and decodes only small batches in FP32.
    """
    if any(type(value) is not int or value < 1 for value in (
        height, width, steps, decode_batch_size,
    )):
        raise ValueError("height, width, steps and decode_batch_size must be positive integers")
    if type(offload) is not bool or type(cache_trajectory) is not bool:
        raise ValueError("offload and cache_trajectory must be boolean")
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or not math.isfinite(guidance_scale)
        or guidance_scale < 1
    ):
        raise ValueError("guidance_scale must be finite and at least 1")
    _validate_conditioning(prompt_embeds, prompt_attention_mask, "prompt")
    has_negative = negative_prompt_embeds is not None
    if has_negative != (negative_prompt_attention_mask is not None):
        raise ValueError("negative embeddings and attention mask must be supplied together")
    if guidance_scale > 1 and not has_negative:
        raise ValueError("CFG>1 requires negative embeddings and attention mask")
    if has_negative:
        _validate_conditioning(negative_prompt_embeds, negative_prompt_attention_mask, "negative_prompt")
        if (
            negative_prompt_embeds.shape != prompt_embeds.shape
            or negative_prompt_attention_mask.shape != prompt_attention_mask.shape
        ):
            raise ValueError("conditional and unconditional conditioning shapes must match")

    transformer = pipe.transformer
    underlying = transformer.module if hasattr(transformer, "module") else transformer
    device = next(underlying.parameters()).device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("ART Sana sampling requires a CPU or CUDA transformer")
    scale = pipe.vae_scale_factor
    channels = underlying.config.in_channels
    if type(scale) is not int or scale < 1 or type(channels) is not int or channels < 1:
        raise ValueError("VAE scale factor and transformer channels must be positive integers")
    if height % scale or width % scale:
        raise ValueError(f"image dimensions must be divisible by DC-AE scale {scale}")
    vae_scale = float(pipe.vae.config.scaling_factor)
    if not math.isfinite(vae_scale) or vae_scale <= 0:
        raise ValueError("VAE scaling_factor must be finite and positive")
    batch_size = prompt_embeds.shape[0]
    _validate_generator(generator, batch_size, device)

    pipe.scheduler.set_timesteps(steps, device=device)
    schedule = torch.as_tensor(pipe.scheduler.sigmas, dtype=torch.float32, device="cpu").detach().clone()
    scheduler_times = torch.as_tensor(pipe.scheduler.timesteps, device="cpu")
    if (
        schedule.shape != (steps + 1,)
        or scheduler_times.shape != (steps,)
        or not torch.isfinite(schedule).all()
        or not torch.isfinite(scheduler_times).all()
        or (schedule < 0).any()
        or (schedule > 1).any()
        or not (schedule[:-1] > schedule[1:]).all()
    ):
        raise ValueError("Sana sigmas must be finite, strictly decreasing in [0, 1], with T+1 entries")
    sigmas = schedule[:-1].clone()
    dt = schedule[1:] - sigmas
    timesteps = sigmas * 1000.0
    device_dt, device_times = dt.to(device), timesteps.to(device)
    prompt_embeds = prompt_embeds.to(device)
    prompt_attention_mask = prompt_attention_mask.to(device)
    if has_negative:
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

    shape = (batch_size, channels, height // scale, width // scale)
    cache_shape = (batch_size, steps, *shape[1:])
    states = torch.empty(cache_shape, dtype=torch.float32, device="cpu") if cache_trajectory else None
    old_velocities = torch.empty_like(states) if cache_trajectory else None
    # Make this independent of the pipeline loader's offload setting.
    pipe.vae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        for index in range(steps):
            velocity = predict_guided_velocity(
                transformer, latents, device_times[index].expand(batch_size),
                prompt_embeds, prompt_attention_mask, guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )
            if (
                velocity.shape != latents.shape
                or not torch.isfinite(velocity).all()
                or not torch.isfinite(latents).all()
            ):
                raise ValueError("ART states and velocities must be finite with matching latent shapes")
            velocity = velocity.float()
            if cache_trajectory:
                states[:, index].copy_(latents)
                old_velocities[:, index].copy_(velocity)
            # Match DiffusionART flow_transition(kind="ode"); do not negate v.
            latents = latents + device_dt[index] * velocity
        if not torch.isfinite(latents).all():
            raise ValueError("ART endpoints must be finite")
        endpoints = latents.detach().to(device="cpu", dtype=torch.float32)
        del latents, velocity

        pipe.vae.to(device, dtype=torch.float32)
        images = torch.empty((batch_size, 3, height, width), dtype=torch.float32, device="cpu")
        for start in range(0, batch_size, decode_batch_size):
            end = min(start + decode_batch_size, batch_size)
            # DC-AE uses scaling_factor only; SD3's shift_factor does not apply.
            with torch.autocast(device_type=device.type, enabled=False):
                decode_latents = endpoints[start:end].to(device) / vae_scale
                decoded = pipe.vae.decode(decode_latents, return_dict=False)[0]
            if decoded.shape != (end - start, 3, height, width) or not torch.isfinite(decoded).all():
                raise ValueError("VAE must produce finite RGB images at the requested resolution")
            decoded_cpu = decoded.detach().to(device="cpu", dtype=torch.float32)
            del decoded, decode_latents
            image_batch = pipe.image_processor.postprocess(decoded_cpu, output_type="pt")
            if image_batch.shape != (end - start, 3, height, width) or not torch.isfinite(image_batch).all():
                raise ValueError("postprocessed images must be finite RGB tensors")
            images[start:end].copy_(image_batch.clamp(0, 1))
            del decoded_cpu, image_batch
        return {
            "states": states,
            "old_velocities": old_velocities,
            "endpoints": endpoints,
            "sigmas": sigmas,
            "dt": dt,
            "timesteps": timesteps,
            "images": images,
        }
    finally:
        if offload:
            pipe.vae.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
