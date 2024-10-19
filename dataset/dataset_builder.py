"""Dataset loading and processing utilities."""

import os
from typing import Dict, List, Optional, Union, Callable
import json
import torch
from torch.utils.data import Dataset
import datasets
from datasets import (
    load_dataset,
    DatasetDict,
    Image,
    Features,
    load_from_disk,
    load_dataset_builder,
    get_dataset_split_names,
    IterableDatasetDict,
    IterableDataset,
)
from .transformations.transforms import (
    train_transforms,
)  # Assuming this is for image transformations
from ..loggers.logging_colors import get_logger

logger = get_logger()


class CustomTorchDataset(Dataset):
    """Custom PyTorch Dataset class for loading image data.

    This class assumes your data is organized in a way that each sample
    can be accessed by an index.
    """

    def __init__(self, data: List[Dict], transforms: Optional[Callable] = None):
        """
        Initializes the CustomTorchDataset.

        Args:
            data (List[Dict]): A list of dictionaries, where each dictionary represents a data sample
                               and has at least 'image' and 'label' keys.
            transforms (Optional[Callable], optional): A function/transform to apply to the data.
                                                        Defaults to None.
        """
        self.data = data
        self.transforms = transforms
        logger.info(f"CustomTorchDataset initialized with {len(self.data)} samples")

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Returns the sample at the given index."""
        item = self.data[idx].copy()  # Make a copy to avoid modifying the original data
        if self.transforms:
            item = self.transforms(item)
        return item


class CustomHuggingFaceDataset(object):
    """Wrapper class for loading and processing Hugging Face Datasets."""

    def __init__(
        self,
        dataset_name: str,
        dataset_script_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
    ):
        """
        Initializes the CustomHuggingFaceDataset.

        Args:
            dataset_name (str):
                * For standard Hugging Face datasets: Name of the dataset (e.g., 'cifar10', 'imagenet').
                * For custom datasets: The name you registered your dataset under.
            dataset_script_path (Optional[str], optional):
                Path to your custom dataset script if using a custom dataset.
                Defaults to None.
            transforms (Optional[Callable], optional):
                A function/transform to apply to the data. Defaults to None.
        """
        self.dataset_name = dataset_name
        self.dataset_script_path = dataset_script_path
        self.transforms = transforms

        # Determine if we're loading a custom dataset
        self.is_custom_dataset = bool(self.dataset_script_path)

        if self.is_custom_dataset:
            # Register the custom dataset if a script path is provided
            datasets.load_dataset(
                self.dataset_script_path, name="default", trust_remote_code=True
            )

        logger.info(
            f"CustomHuggingFaceDataset initialized with dataset: {self.dataset_name}"
            f" (Custom: {self.is_custom_dataset})"
        )

    def info(self) -> None:
        """Print information about the dataset."""
        if self.is_custom_dataset:
            logger.info(f"Custom dataset loaded from: {self.dataset_script_path}")
            # Add logic here to fetch and print info for your custom dataset
            # You might need to access information differently than for standard HF datasets
        else:
            logger.info(f"Dataset: {self.dataset_name}")
            logger.info(f"Description: {self.dataset_builder.info.description}")
            logger.info(f"Features: {self.dataset_builder.info.features}")
            logger.info(f"Splits: {get_dataset_split_names(self.dataset_name)}")

    def load(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> Union[DatasetDict, datasets.Dataset, IterableDatasetDict, IterableDataset]:
        """Load the dataset split.

        This method handles loading both standard Hugging Face datasets and custom
        datasets based on the `self.is_custom_dataset` flag.

        Args:
            split (str, optional): The dataset split to load ('train', 'validation', 'test').
                                    Defaults to 'train'.
            cache_dir (Optional[str], optional): Directory to cache the downloaded dataset.
                                                Defaults to None.
            streaming (bool, optional): Whether to load the dataset in streaming mode.
                                        Defaults to False.

        Returns:
            Union[DatasetDict, datasets.Dataset, IterableDatasetDict, IterableDataset]:
                The loaded dataset split.
        """
        load_kwargs = (
            {"cache_dir": cache_dir, "streaming": streaming}
            if streaming
            else {"cache_dir": cache_dir}
        )

        if self.is_custom_dataset:
            # Load custom dataset
            dataset = load_dataset(
                self.dataset_name, trust_remote_code=True, **load_kwargs
            )
        else:
            # Load standard Hugging Face dataset
            dataset = load_dataset(self.dataset_name, **load_kwargs)

        logger.info(f"Loaded {split} split with {len(dataset[split])} samples")

        if self.transforms:
            # Assuming `self.transforms` is a function that can be mapped over the dataset
            dataset = dataset.map(self.transforms, num_proc=4, batched=True)

        return dataset[split]

    @staticmethod
    def create_from_paths(
        image_paths: List[str],
        label_paths: List[str],
        features: Optional[Features] = None,
        transforms: Optional[Callable] = None,
    ) -> datasets.Dataset:
        """Create a Hugging Face Dataset from lists of image and label paths.

        Args:
            image_paths (List[str]): List of paths to the images.
            label_paths (List[str]): List of paths to the labels.
            features (Optional[Features], optional): Dataset Features. If None, infer from data.
                                                    Defaults to None.
            transforms (Optional[Callable], optional): A function/transform to apply to the data.
                                                        Defaults to None.

        Returns:
            datasets.Dataset: The created Hugging Face Dataset.
        """
        data_dict = {"image": sorted(image_paths), "label": sorted(label_paths)}
        dataset = datasets.Dataset.from_dict(data_dict, features=features)

        if transforms:
            dataset = dataset.map(transforms, num_proc=4, batched=True)

        logger.info(f"Created image dataset with {len(dataset)} samples")
        return dataset
