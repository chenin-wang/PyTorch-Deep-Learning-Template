"""Data loading utilities for PyTorch training."""

import torch
from torch.utils.data import DataLoader, random_split, DistributedSampler
from .dataset_builder import CustomTorchDataset
from typing import Dict, List, Optional, Union, Callable
from ..loggers.logging_colors import get_logger

logger = get_logger()


def get_dataloaders(
    dataset: Union[torch.utils.data.Dataset, CustomTorchDataset],
    split: tuple = (0.8, 0.1, 0.1),
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    seed: int = 42,
    **kwargs,
):
    """Create and return PyTorch DataLoaders for train, validation, and test sets.

    This function handles dataset splitting, DistributedSampler setup (if enabled),
    and DataLoader creation for efficient data loading during training.

    Args:
        dataset (torch.utils.data.Dataset): The full dataset to be split.
        split (tuple, optional): Dataset split ratios (train, val, test). Defaults to (0.8, 0.1, 0.1).
        batch_size (int, optional): Batch size for the DataLoaders. Defaults to 32.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        distributed (bool, optional):  Whether to use distributed training. Defaults to False.
        seed (int, optional): Random seed for reproducible splitting. Defaults to 42.
        **kwargs: Additional keyword arguments to pass to the DataLoader.

    Returns:
        tuple: A tuple of DataLoaders (train_loader, val_loader, test_loader).
               If `dataset` is None, returns (None, None, None).
    """

    if dataset is None:
        return None, None, None

    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(
        f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # Create samplers, ensuring correct behavior with DistributedSampler
    samplers = {}
    for split_name, split_dataset in zip(
        ["train", "val", "test"], [train_dataset, val_dataset, test_dataset]
    ):
        if distributed:
            # Use DistributedSampler for distributed training
            # Important: Set shuffle=False for validation and test loaders
            samplers[split_name] = DistributedSampler(
                split_dataset, shuffle=(split_name == "train")
            )
        else:
            samplers[split_name] = None

    dataloaders = {
        split_name: DataLoader(
            split_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=(samplers[split_name] is None),  # Important: shuffle if no sampler
            sampler=samplers[split_name],
            **kwargs,
        )
        for split_name, split_dataset in zip(
            ["train", "val", "test"], [train_dataset, val_dataset, test_dataset]
        )
    }

    logger.info(
        f'Dataloaders created: Train={len(dataloaders["train"])} batches, '
        f'Validation={len(dataloaders["val"])} batches, Test={len(dataloaders["test"])} batches.'
    )

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]
