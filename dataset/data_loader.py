import torch
from torch.utils.data import DataLoader, random_split, DistributedSampler
from .Dataset import CustomTorchDataset
from ..loggers.logging_colors import get_logger

logger = get_logger()


def get_dataloaders(
    data_dir,
    transform=None,
    split=(0.8, 0.1, 0.1),
    batch_size=32,
    num_workers=4,
    distributed=False,
    seed=42,
    *args,
    **kwargs,
):
    """
    Returns train, val and test dataloaders for various training scenarios.

    Args:
        train_dir (str): Path to training data directory
        transform (callable, optional): A function/transform to apply to the data
        split (tuple): Ratios for train, validation and test splits
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        distributed (bool): Whether to use distributed training

    Returns:
        tuple: train_dataloader, val_dataloader, test_dataloader
    """
    full_dataset = CustomTorchDataset(data_dir, transform=transform)

    train_size, val_size = (
        int(split[0] * len(full_dataset)),
        int(split[1] * len(full_dataset)),
    )
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(
        f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}"
    )

    samplers = {
        "train": DistributedSampler(train_dataset) if distributed else None,
        "val": DistributedSampler(val_dataset, shuffle=False) if distributed else None,
        "test": DistributedSampler(test_dataset, shuffle=False)
        if distributed
        else None,
    }

    dataloader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        **kwargs,
    }

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            shuffle=(samplers["train"] is None),
            sampler=samplers["train"],
            **dataloader_args,
        ),
        "val": DataLoader(
            val_dataset, shuffle=False, sampler=samplers["val"], **dataloader_args
        ),
        "test": DataLoader(
            test_dataset, shuffle=False, sampler=samplers["test"], **dataloader_args
        ),
    }

    logger.info(
        f'Dataloaders created: Train={len(dataloaders["train"])} batches, '
        f'Validation={len(dataloaders["val"])} batches, Test={len(dataloaders["test"])} batches.'
    )

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]
