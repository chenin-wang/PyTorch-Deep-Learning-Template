
import torch
from typing import Dict, List, Optional, Union
import datasets
from datasets import (
    load_dataset, DatasetDict, Image, Features,
    load_dataset_builder, get_dataset_split_names, IterableDatasetDict, IterableDataset
)
from .transformations.transforms import train_transforms
from ..loggers.logging_colors import get_logger
logger = get_logger()

class CustomTorchDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict], transforms=None):
        self.data = data
        self.transforms = transforms
        logger.info(f"CustomTorchDataset initialized with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transforms:
            item = self.transforms(item)
        return item

class CustomHuggingFaceDataset:
    def __init__(self, path: str = "rotten_tomatoes", transforms=None):
        self.path = path
        self.dataset_builder = load_dataset_builder(self.path)
        self.transforms = transforms
        logger.info(f"CustomHuggingFaceDataset initialized with path: {self.path}")

    def info(self) -> None:
        """Get general information about the dataset."""
        logger.info(f"Dataset: {self.path}")
        logger.info(f"Description: {self.dataset_builder.info.description}")
        logger.info(f"Features: {self.dataset_builder.info.features}")
        logger.info(f"Splits: {get_dataset_split_names(self.path)}")

    def load(self, split: str = "train", batched: bool = False) -> Union[DatasetDict, datasets.Dataset, IterableDatasetDict, IterableDataset]:
        """Load a specific split of the dataset."""
        dataset = load_dataset(self.path, trust_remote_code=True)
        logger.info(f"Loaded {split} split with {len(dataset[split])} samples")
        if self.transforms:
            dataset[split] = dataset[split].map(self.transforms, num_proc=4, batched=batched)
        return dataset[split]

    @staticmethod
    def create_dataset(image_paths: List[str], label_paths: List[str], transforms=None, batched: bool = False) -> datasets.Dataset:
        """Create a dataset from image and label paths."""
        dataset = datasets.Dataset.from_dict({
            "image": sorted(image_paths),
            "label": sorted(label_paths)
        })
        dataset = dataset.cast_column("image", Image()).cast_column("label", Image())
        logger.info(f"Created image dataset with {len(dataset)} samples")
        if transforms:
            dataset = dataset.map(transforms, num_proc=4, batched=batched)
        return dataset

if __name__ == "__main__":
    # Example usage for CustomTorchDataset
    data = [{"image": "path/to/image1.jpg", "label": 0}, 
            {"image": "path/to/image2.jpg", "label": 1}]
    torch_dataset = CustomTorchDataset(data, transforms=train_transforms)
    logger.info(f"CustomTorchDataset sample: {torch_dataset[0]}")

    # Example usage for CustomHuggingFaceDataset
    hf_dataset = CustomHuggingFaceDataset()
    hf_dataset.info()
    train_data = hf_dataset.load("train")
    logger.info(f"CustomHuggingFaceDataset sample: {train_data[0]}")

    # Example for image dataset creation
    image_dataset = CustomHuggingFaceDataset.create_dataset(
        image_paths=["path/to/train_image_1.jpg", "path/to/train_image_2.jpg"],
        label_paths=["path/to/train_label_1.png", "path/to/train_label_2.png"],
        transforms=train_transforms
    )
    logger.info(f"Image dataset sample: {image_dataset[0]}")

    logger.info("Dataset creation and transformation completed")
