"""
Diffusion utilities for the SR3-style continuous noise-level UNet.

This module provides helpers that bridge between the ``DDPMScheduler``
(from HuggingFace diffusers) and our custom UNet which is conditioned on
a **continuous** ``sqrt(alpha_cumprod)`` noise level rather than on an
integer timestep.

Key functions
-------------
* ``make_noise_scheduler`` — create a ``DDPMScheduler`` from config dict.
* ``q_sample`` — forward diffusion (add noise at a continuous noise level).
* ``compute_loss`` — single training-step loss.
* ``extract_features`` — get encoder/decoder features for change detection.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def make_noise_scheduler(schedule_opt: dict) -> DDPMScheduler:
    """Create a ``DDPMScheduler`` from a legacy config dict.

    The dict is expected to contain::

        {
            "schedule": "linear" | "cosine",
            "n_timestep": 2000,
            "linear_start": 1e-6,
            "linear_end": 1e-2
        }

    Returns:
        A configured :class:`~diffusers.DDPMScheduler`.
    """
    schedule_name = schedule_opt["schedule"]
    n_timestep = schedule_opt["n_timestep"]
    linear_start = schedule_opt.get("linear_start", 1e-4)
    linear_end = schedule_opt.get("linear_end", 2e-2)

    if schedule_name == "cosine":
        beta_schedule = "squaredcos_cap_v2"
    else:
        beta_schedule = "linear"

    return DDPMScheduler(
        num_train_timesteps=n_timestep,
        beta_start=linear_start,
        beta_end=linear_end,
        beta_schedule=beta_schedule,
        clip_sample=False,
    )


# ---------------------------------------------------------------------------
# Pre-computed alpha tables (needed for continuous noise-level conditioning)
# ---------------------------------------------------------------------------

def precompute_alpha_tables(scheduler: DDPMScheduler):
    """Return ``sqrt_alphas_cumprod_prev`` from a ``DDPMScheduler``.

    The SR3 UNet is conditioned on a **continuous** noise level sampled
    uniformly between ``sqrt(alpha_cumprod[t-1])`` and
    ``sqrt(alpha_cumprod[t])``.  This function pre-computes the table
    ``[1, sqrt(alpha_cumprod_0), sqrt(alpha_cumprod_1), ...]`` needed to
    draw those samples.
    """
    alphas_cumprod = scheduler.alphas_cumprod.numpy()
    sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))
    return sqrt_alphas_cumprod_prev  # length = num_train_timesteps + 1


# ---------------------------------------------------------------------------
# Forward diffusion (q_sample)
# ---------------------------------------------------------------------------

def q_sample(x_start, continuous_sqrt_alpha_cumprod, noise=None):
    """Add noise at a continuous noise level.

    .. math::
        x_t = \\sqrt{\\bar\\alpha_t}\\, x_0
              + \\sqrt{1 - \\bar\\alpha_t}\\, \\epsilon

    Args:
        x_start: clean image ``(B, C, H, W)``
        continuous_sqrt_alpha_cumprod: noise level ``(B, 1, 1, 1)``
        noise: optional pre-sampled noise
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    return (
        continuous_sqrt_alpha_cumprod * x_start
        + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
    )


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def compute_loss(model, x_start, sqrt_alphas_cumprod_prev, num_timesteps, loss_type="l2"):
    """Compute the denoising loss for one training step.

    This replicates the SR3-style continuous-noise-level training:
    1. Sample a random integer timestep ``t``.
    2. Draw a continuous noise level uniformly in
       ``[sqrt_alpha_cumprod[t-1], sqrt_alpha_cumprod[t]]``.
    3. Add noise to ``x_start`` with that level.
    4. Predict the noise and compute the loss.

    Args:
        model: the UNet.
        x_start: clean images ``(B, C, H, W)``.
        sqrt_alphas_cumprod_prev: pre-computed table from
            :func:`precompute_alpha_tables`.
        num_timesteps: total number of diffusion timesteps.
        loss_type: ``"l1"`` or ``"l2"``.

    Returns:
        Scalar loss tensor.
    """
    b, c, h, w = x_start.shape

    t = np.random.randint(1, num_timesteps + 1)
    continuous_sqrt_alpha_cumprod = torch.FloatTensor(
        np.random.uniform(
            sqrt_alphas_cumprod_prev[t - 1],
            sqrt_alphas_cumprod_prev[t],
            size=b,
        )
    ).to(x_start.device)
    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

    noise = torch.randn_like(x_start)
    x_noisy = q_sample(
        x_start=x_start,
        continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
        noise=noise,
    )

    noise_pred = model(x_noisy, continuous_sqrt_alpha_cumprod)

    if loss_type == "l1":
        loss = F.l1_loss(noise_pred, noise, reduction="sum")
    else:
        loss = F.mse_loss(noise_pred, noise, reduction="sum")
    return loss


# ---------------------------------------------------------------------------
# Feature extraction (for change detection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, x, t, sqrt_alphas_cumprod_prev):
    """Extract intermediate encoder/decoder features from the UNet.

    Args:
        model: the UNet (with ``feat_need`` support).
        x: clean image ``(B, C, H, W)``.
        t: integer timestep for noise level sampling.
        sqrt_alphas_cumprod_prev: pre-computed table.

    Returns:
        ``(encoder_feats, decoder_feats)`` — lists of feature tensors.
    """
    b, c, h, w = x.shape

    continuous_sqrt_alpha_cumprod = torch.FloatTensor(
        np.random.uniform(
            sqrt_alphas_cumprod_prev[t - 1],
            sqrt_alphas_cumprod_prev[t],
            size=b,
        )
    ).to(x.device)
    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

    noise = torch.randn_like(x)
    x_noisy = q_sample(
        x_start=x,
        continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
        noise=noise,
    )

    fe, fd = model(x_noisy, continuous_sqrt_alpha_cumprod, feat_need=True)
    return fe, fd
