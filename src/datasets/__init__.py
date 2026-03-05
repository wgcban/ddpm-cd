"""Dataset and dataloader factories."""

import logging
import torch.utils.data

from src.datasets.image_dataset import ImageDataset
from src.datasets.cd_dataset import CDDataset

logger = logging.getLogger(__name__)


def create_dataloader(dataset, dataset_opt, phase):
    """Create a PyTorch DataLoader."""
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt.get("use_shuffle", True),
            num_workers=dataset_opt.get("num_workers", 4),
            pin_memory=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt.get("batch_size", 1),
            shuffle=False,
            num_workers=dataset_opt.get("num_workers", 1),
            pin_memory=True,
        )


def create_image_dataset(dataset_opt, phase):
    """Create an :class:`ImageDataset` for unconditional DDPM pre-training."""
    dataset = ImageDataset(
        dataroot=dataset_opt["dataroot"],
        resolution=dataset_opt["resolution"],
        split=phase,
        data_len=dataset_opt.get("data_len", -1),
    )
    logger.info(f"Created ImageDataset [{dataset_opt.get('name', '')}] with {len(dataset)} samples.")
    return dataset


def create_cd_dataset(dataset_opt, phase):
    """Create a :class:`CDDataset` for change-detection fine-tuning."""
    dataset = CDDataset(
        dataroot=dataset_opt["dataroot"],
        resolution=dataset_opt["resolution"],
        split=phase,
        data_len=dataset_opt.get("data_len", -1),
    )
    logger.info(f"Created CDDataset [{dataset_opt.get('name', '')} / {phase}] with {len(dataset)} samples.")
    return dataset

