#!/usr/bin/env python
"""
Fine-tune a change-detection head on top of a pre-trained DDPM feature extractor.

Follows HuggingFace diffusers training conventions:
  * Uses ``accelerate`` for distributed training / mixed precision.
  * Loads the pre-trained UNet via ``UNet.from_pretrained()``.
  * Uses ``DDPMScheduler`` for noise schedule management.
  * Uses ``argparse`` for configuration (diffusers examples style).

Example::

    accelerate launch scripts/train_cd.py \
        --pretrained_model_path experiments/ddpm-pretrain/unet \
        --train_data_dir dataset/LEVIR-CD256 \
        --val_data_dir dataset/LEVIR-CD256 \
        --output_dir experiments/cd-levir \
        --resolution 256 \
        --train_batch_size 8 \
        --num_epochs 120 \
        --learning_rate 1e-4 \
        --ddpm_num_steps 2000 \
        --ddpm_beta_schedule linear \
        --timesteps 50 100 400 \
        --feat_type dec \
        --logger wandb
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` and `libs` are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from torch.optim import lr_scheduler as torch_lr_scheduler
from tqdm.auto import tqdm

from src.models.unet import UNet
from src.models.diffusion import (
    extract_features,
    make_noise_scheduler,
    precompute_alpha_tables,
)
from src.models.cd_modules.cd_head_v2 import cd_head_v2
from src.datasets import create_cd_dataset, create_dataloader
from libs.metric_tools import ConfuseMatrixMeter

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CD head on pre-trained DDPM features.")
    # Data
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--data_len", type=int, default=-1)
    # Pre-trained DDPM
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to pre-trained UNet (save_pretrained dir) or .pth weights.")
    parser.add_argument("--in_channel", type=int, default=3)
    parser.add_argument("--out_channel", type=int, default=3)
    parser.add_argument("--inner_channel", type=int, default=128)
    parser.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 4, 8, 8])
    parser.add_argument("--attn_res", type=int, nargs="+", default=[16])
    parser.add_argument("--res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm_groups", type=int, default=32)
    # Diffusion schedule
    parser.add_argument("--ddpm_num_steps", type=int, default=2000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--ddpm_beta_start", type=float, default=1e-6)
    parser.add_argument("--ddpm_beta_end", type=float, default=1e-2)
    # Change detection
    parser.add_argument("--timesteps", type=int, nargs="+", default=[50, 100, 400],
                        help="Diffusion timesteps for feature extraction.")
    parser.add_argument("--feat_type", type=str, default="dec", choices=["enc", "dec"])
    parser.add_argument("--feat_scales", type=int, nargs="+", default=[2, 5, 8, 11, 14])
    parser.add_argument("--cd_out_channels", type=int, default=2)
    parser.add_argument("--cd_output_size", type=int, default=256)
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce"])
    # Training
    parser.add_argument("--output_dir", type=str, default="experiments/cd-levir")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=120)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_policy", type=str, default="linear",
                        choices=["linear", "step", "cosine"])
    parser.add_argument("--optimizer_type", type=str, default="adam",
                        choices=["adam", "adamw"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=1,
                        help="Validate every N epochs.")
    parser.add_argument("--log_freq", type=int, default=500,
                        help="Log training stats every N steps.")
    # Checkpointing
    parser.add_argument("--resume_cd_weights", type=str, default=None,
                        help="Path to CD head weights (.pth) to resume from.")
    # Logging
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    # Phase
    parser.add_argument("--phase", type=str, default="train",
                        choices=["train", "test"])

    return parser.parse_args()


def build_unet(args):
    """Build the UNet and load pre-trained weights.

    Supports two loading modes:
    1. **diffusers format** — if ``pretrained_model_path`` points to a directory
       with ``config.json``, uses ``UNet.from_pretrained()``.
    2. **raw state_dict** — otherwise, builds from CLI args and loads weights
       from a ``.pth`` file (supports ``*_gen.pth`` suffix convention).
    """
    pretrained = args.pretrained_model_path

    # Try diffusers from_pretrained first
    if os.path.isdir(pretrained) and os.path.exists(os.path.join(pretrained, "config.json")):
        logger.info(f"Loading UNet from_pretrained: {pretrained}")
        unet = UNet.from_pretrained(pretrained)
    else:
        # Fall back to loading raw state_dict
        logger.info(f"Building UNet from args and loading weights: {pretrained}")
        unet = UNet(
            in_channel=args.in_channel,
            out_channel=args.out_channel,
            inner_channel=args.inner_channel,
            norm_groups=args.norm_groups,
            channel_mults=tuple(args.channel_mults),
            attn_res=tuple(args.attn_res),
            res_blocks=args.res_blocks,
            dropout=args.dropout,
            image_size=args.resolution,
        )
        # Support both "_gen.pth" suffix and plain paths
        weight_path = pretrained
        if not weight_path.endswith(".pth"):
            weight_path = f"{pretrained}_gen.pth"
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            unet.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded UNet weights from {weight_path}")

    # Freeze UNet — only train CD head
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    return unet


def build_cd_head(args):
    """Build the change-detection head."""
    cd_head = cd_head_v2(
        feat_scales=args.feat_scales,
        out_channels=args.cd_out_channels,
        inner_channel=args.inner_channel,
        channel_multiplier=args.channel_mults,
        img_size=args.cd_output_size,
        time_steps=args.timesteps,
    )
    return cd_head


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ---- Models ----
    unet = build_unet(args)
    cd_head_model = build_cd_head(args)

    if args.resume_cd_weights and os.path.exists(args.resume_cd_weights):
        cd_head_model.load_state_dict(torch.load(args.resume_cd_weights, map_location="cpu"))
        logger.info(f"Resumed CD head weights from {args.resume_cd_weights}")

    # ---- Scheduler ----
    beta_schedule = "squaredcos_cap_v2" if args.ddpm_beta_schedule == "cosine" else "linear"
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_start=args.ddpm_beta_start,
        beta_end=args.ddpm_beta_end,
        beta_schedule=beta_schedule,
        clip_sample=False,
    )
    sqrt_alphas = precompute_alpha_tables(noise_scheduler)

    # ---- Loss ----
    loss_fn = nn.CrossEntropyLoss()

    # ---- Optimizer ----
    if args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(cd_head_model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(cd_head_model.parameters(), lr=args.learning_rate)

    # ---- LR scheduler ----
    if args.lr_policy == "linear":
        lr_sched = torch_lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0 - epoch / float(args.num_epochs + 1)
        )
    elif args.lr_policy == "step":
        lr_sched = torch_lr_scheduler.StepLR(
            optimizer, step_size=args.num_epochs // 3, gamma=0.1
        )
    else:
        lr_sched = torch_lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )

    # ---- Datasets ----
    common_ds_opt = {
        "resolution": args.resolution,
        "data_len": args.data_len,
        "num_workers": args.dataloader_num_workers,
    }
    if args.phase == "train":
        train_ds_opt = {
            **common_ds_opt,
            "dataroot": args.train_data_dir,
            "batch_size": args.train_batch_size,
            "use_shuffle": True,
            "name": "train",
        }
        train_dataset = create_cd_dataset(train_ds_opt, "train")
        train_dataloader = create_dataloader(train_dataset, train_ds_opt, "train")

        if args.val_data_dir:
            val_ds_opt = {
                **common_ds_opt,
                "dataroot": args.val_data_dir,
                "batch_size": args.val_batch_size,
                "use_shuffle": False,
                "name": "val",
            }
            val_dataset = create_cd_dataset(val_ds_opt, "val")
            val_dataloader = create_dataloader(val_dataset, val_ds_opt, "val")
        else:
            val_dataloader = None
    else:
        test_dir = args.test_data_dir or args.train_data_dir
        test_ds_opt = {
            **common_ds_opt,
            "dataroot": test_dir,
            "batch_size": args.val_batch_size,
            "use_shuffle": False,
            "name": "test",
        }
        test_dataset = create_cd_dataset(test_ds_opt, "test")
        test_dataloader = create_dataloader(test_dataset, test_ds_opt, "test")

    # ---- Accelerate prepare ----
    if args.phase == "train":
        cd_head_model, optimizer, train_dataloader = accelerator.prepare(
            cd_head_model, optimizer, train_dataloader
        )
        if val_dataloader is not None:
            val_dataloader = accelerator.prepare(val_dataloader)
    else:
        cd_head_model = accelerator.prepare(cd_head_model)
        test_dataloader = accelerator.prepare(test_dataloader)

    unet = unet.to(accelerator.device)

    # ---- Trackers ----
    if accelerator.is_main_process:
        accelerator.init_trackers("ddpm-cd")

    # ---- Metrics ----
    running_metric = ConfuseMatrixMeter(n_class=args.cd_out_channels)

    # ====================================================================
    # Training
    # ====================================================================
    if args.phase == "train":
        best_mF1 = 0.0

        for epoch in range(args.num_epochs):
            cd_head_model.train()
            running_metric.clear()

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch}/{args.num_epochs}",
                disable=not accelerator.is_local_main_process,
            )

            for step, batch in enumerate(progress_bar):
                img_A = batch["A"].to(accelerator.device)
                img_B = batch["B"].to(accelerator.device)
                labels = batch["L"].to(accelerator.device).long()

                # Extract features from frozen DDPM
                feats_A, feats_B = [], []
                for t in args.timesteps:
                    fe_A, fd_A = extract_features(unet, img_A, t, sqrt_alphas)
                    fe_B, fd_B = extract_features(unet, img_B, t, sqrt_alphas)
                    if args.feat_type == "dec":
                        feats_A.append(fd_A)
                        feats_B.append(fd_B)
                    else:
                        feats_A.append(fe_A)
                        feats_B.append(fe_B)

                with accelerator.accumulate(cd_head_model):
                    pred = cd_head_model(feats_A, feats_B)
                    loss = loss_fn(pred, labels)

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                # Metrics
                pred_labels = torch.argmax(pred.detach(), dim=1).cpu().numpy()
                gt_labels = labels.detach().cpu().numpy()
                running_metric.update_cm(pr=pred_labels, gt=gt_labels)

                if step % args.log_freq == 0:
                    progress_bar.set_postfix(loss=loss.item())

            # Epoch summary
            scores = running_metric.get_scores()
            lr_sched.step()

            log_msg = (
                f"[Train] Epoch {epoch}: "
                f"mF1={scores['mf1']:.4f}, mIoU={scores['miou']:.4f}, OA={scores['acc']:.4f}"
            )
            logger.info(log_msg)

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "train/mF1": scores["mf1"],
                        "train/mIoU": scores["miou"],
                        "train/OA": scores["acc"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            # ---- Validation ----
            if val_dataloader is not None and epoch % args.val_freq == 0:
                cd_head_model.eval()
                running_metric.clear()

                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
                        img_A = batch["A"].to(accelerator.device)
                        img_B = batch["B"].to(accelerator.device)
                        labels = batch["L"].to(accelerator.device).long()

                        feats_A, feats_B = [], []
                        for t in args.timesteps:
                            fe_A, fd_A = extract_features(unet, img_A, t, sqrt_alphas)
                            fe_B, fd_B = extract_features(unet, img_B, t, sqrt_alphas)
                            if args.feat_type == "dec":
                                feats_A.append(fd_A)
                                feats_B.append(fd_B)
                            else:
                                feats_A.append(fe_A)
                                feats_B.append(fe_B)

                        pred = cd_head_model(feats_A, feats_B)
                        pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
                        gt_labels = labels.cpu().numpy()
                        running_metric.update_cm(pr=pred_labels, gt=gt_labels)

                val_scores = running_metric.get_scores()
                val_msg = (
                    f"[Val] Epoch {epoch}: "
                    f"mF1={val_scores['mf1']:.4f}, mIoU={val_scores['miou']:.4f}, OA={val_scores['acc']:.4f}"
                )
                logger.info(val_msg)

                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "val/mF1": val_scores["mf1"],
                            "val/mIoU": val_scores["miou"],
                            "val/OA": val_scores["acc"],
                        },
                        step=epoch,
                    )

                # Save best + current
                is_best = val_scores["mf1"] > best_mF1
                if is_best:
                    best_mF1 = val_scores["mf1"]
                    logger.info(f"New best mF1: {best_mF1:.4f}")

                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(cd_head_model)
                    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
                    torch.save(
                        unwrapped.state_dict(),
                        os.path.join(ckpt_dir, f"cd_model_E{epoch}.pth"),
                    )
                    if is_best:
                        torch.save(
                            unwrapped.state_dict(),
                            os.path.join(ckpt_dir, "best_cd_model.pth"),
                        )

        accelerator.end_training()
        logger.info("Training finished.")

    # ====================================================================
    # Testing
    # ====================================================================
    else:
        cd_head_model.eval()
        running_metric.clear()

        with torch.no_grad():
            for step, batch in enumerate(
                tqdm(test_dataloader, desc="Testing", disable=not accelerator.is_local_main_process)
            ):
                img_A = batch["A"].to(accelerator.device)
                img_B = batch["B"].to(accelerator.device)
                labels = batch["L"].to(accelerator.device).long()

                feats_A, feats_B = [], []
                for t in args.timesteps:
                    fe_A, fd_A = extract_features(unet, img_A, t, sqrt_alphas)
                    fe_B, fd_B = extract_features(unet, img_B, t, sqrt_alphas)
                    if args.feat_type == "dec":
                        feats_A.append(fd_A)
                        feats_B.append(fd_B)
                    else:
                        feats_A.append(fe_A)
                        feats_B.append(fe_B)

                pred = cd_head_model(feats_A, feats_B)
                pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
                gt_labels = labels.cpu().numpy()
                running_metric.update_cm(pr=pred_labels, gt=gt_labels)

        test_scores = running_metric.get_scores()
        logger.info(
            f"[Test] mF1={test_scores['mf1']:.4f}, "
            f"mIoU={test_scores['miou']:.4f}, OA={test_scores['acc']:.4f}"
        )
        for k, v in test_scores.items():
            logger.info(f"  {k}: {v:.4e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

