#!/usr/bin/env python
"""
Pre-train an unconditional DDPM (SR3-style UNet) on remote-sensing imagery.

Follows HuggingFace diffusers training conventions:
  * Uses ``accelerate`` for distributed training / mixed precision.
  * Uses ``DDPMScheduler`` directly for noise schedule management.
  * Saves models via ``model.save_pretrained()`` (diffusers ``ModelMixin``).
  * Uses ``argparse`` for configuration (diffusers examples style).

Example::

    accelerate launch scripts/train_ddpm.py \
        --train_data_dir dataset/Million-AID-LWD \
        --output_dir experiments/ddpm-pretrain \
        --resolution 256 \
        --train_batch_size 8 \
        --num_train_steps 1000000 \
        --learning_rate 1e-5 \
        --ddpm_num_steps 2000 \
        --ddpm_beta_schedule cosine \
        --save_model_steps 10000 \
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
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from src.models.unet import UNet
from src.models.diffusion import precompute_alpha_tables, q_sample
from src.datasets import create_image_dataset, create_dataloader
from src.pipelines import DDPMCDPipeline

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train DDPM on remote-sensing images.")
    # Data
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Folder containing training images.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--data_len", type=int, default=-1,
                        help="Number of training samples (-1 = all).")
    # Model
    parser.add_argument("--in_channel", type=int, default=3)
    parser.add_argument("--out_channel", type=int, default=3)
    parser.add_argument("--inner_channel", type=int, default=128)
    parser.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 4, 8, 8])
    parser.add_argument("--attn_res", type=int, nargs="+", default=[16])
    parser.add_argument("--res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm_groups", type=int, default=32)
    # Diffusion
    parser.add_argument("--ddpm_num_steps", type=int, default=2000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--ddpm_beta_start", type=float, default=1e-6)
    parser.add_argument("--ddpm_beta_end", type=float, default=1e-2)
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2"])
    # Training
    parser.add_argument("--output_dir", type=str, default="experiments/ddpm-pretrain")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_steps", type=int, default=1000000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    # Checkpointing
    parser.add_argument("--save_model_steps", type=int, default=10000)
    parser.add_argument("--save_image_steps", type=int, default=10000)
    parser.add_argument("--num_eval_images", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # Logging
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    return parser.parse_args()


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

    # ---- Model ----
    model = UNet(
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
    num_timesteps = args.ddpm_num_steps

    # ---- EMA ----
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            model_cls=UNet,
            model_config=model.config,
        )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ---- Dataset ----
    dataset_opt = {
        "dataroot": args.train_data_dir,
        "resolution": args.resolution,
        "data_len": args.data_len,
        "batch_size": args.train_batch_size,
        "use_shuffle": True,
        "num_workers": args.dataloader_num_workers,
        "name": Path(args.train_data_dir).name,
    }
    train_dataset = create_image_dataset(dataset_opt, "train")
    train_dataloader = create_dataloader(train_dataset, dataset_opt, "train")

    # ---- LR scheduler ----
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_steps,
    )

    # ---- Accelerate prepare ----
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.use_ema:
        ema_model.to(accelerator.device)

    # ---- Trackers ----
    if accelerator.is_main_process:
        accelerator.init_trackers("ddpm-pretrain")

    # ---- Training loop ----
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    global_step = 0
    first_epoch = 0

    # Resume
    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total optimization steps = {args.num_train_steps}")

    progress_bar = tqdm(
        range(global_step, args.num_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    epoch = first_epoch
    while global_step < args.num_train_steps:
        epoch += 1
        model.train()
        for batch in train_dataloader:
            if global_step >= args.num_train_steps:
                break

            clean_images = batch["img"]

            with accelerator.accumulate(model):
                # --- SR3-style continuous-noise-level training ---
                b = clean_images.shape[0]
                t = np.random.randint(1, num_timesteps + 1)
                continuous_sqrt_alpha = torch.FloatTensor(
                    np.random.uniform(sqrt_alphas[t - 1], sqrt_alphas[t], size=b)
                ).to(clean_images.device)

                noise = torch.randn_like(clean_images)
                noisy_images = q_sample(
                    clean_images,
                    continuous_sqrt_alpha.view(-1, 1, 1, 1),
                    noise,
                )

                noise_pred = model(noisy_images, continuous_sqrt_alpha.view(b, -1))

                if args.loss_type == "l1":
                    loss = F.l1_loss(noise_pred, noise)
                else:
                    loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # --- Save checkpoint ---
                if accelerator.is_main_process and global_step % args.save_model_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                    unet = accelerator.unwrap_model(model)
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())

                    # Save as a diffusers pipeline
                    pipeline = DDPMCDPipeline(unet=unet, scheduler=noise_scheduler)
                    pipeline.save_pretrained(args.output_dir)

                    if args.use_ema:
                        ema_model.restore(unet.parameters())

                    logger.info(f"Saved checkpoint at step {global_step}")

    accelerator.end_training()
    logger.info("Training finished.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

