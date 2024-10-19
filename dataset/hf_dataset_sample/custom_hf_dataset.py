"""Sample dataset implementation for Hugging Face Datasets.

This script demonstrates how to create a custom dataset for use with the
Hugging Face Datasets library. It follows the guidelines provided in the
Hugging Face Datasets documentation:

https://huggingface.co/docs/datasets/v3.0.0/en/dataset_script

The dataset in this example is for demonstration purposes only and loads
data from local JSONL files. You can adapt this template for your own datasets.
"""

import json
import os
from typing import Dict, Tuple, Iterator

import datasets

_CITATION = """\
@InProceedings{dataset:dataset,
title = {A great new dataset},
author={author, Inc.
},
year={2024}
}
"""

_DESCRIPTION = """\
This is a sample dataset for demonstration purposes. 
"""

_HOMEPAGE = ""
_LICENSE = "MIT"

# Provide the base directory where your data is located.
_DATA_DIR = "E:/code_learning/PyTorch-Deep-Learning-Template/dataset/hf_dataset_sample"


class CustomImageDataset(datasets.GeneratorBasedBuilder):
    """A custom image dataset with input, output, and labels."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="Default configuration for the custom image dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        """Define the dataset schema and metadata."""
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "input_image": datasets.Image(decode=True),
                    "description": datasets.Value("string"),
                    "output_image": datasets.Image(decode=True),
                    "label": datasets.ClassLabel(names=["0", "1"]),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """Define the dataset splits."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(_DATA_DIR, "val.jsonl"),
                    "split": "val",
                },
            ),
        ]

    def _generate_examples(
        self, filepath: str, split: str
    ) -> Iterator[Tuple[int, Dict]]:
        """Generate dataset examples from JSONL files."""
        with open(filepath, encoding="utf-8") as f:
            for key, line in enumerate(f):
                data = json.loads(line)
                yield (
                    key,
                    {
                        "input_image": data["input_image"],
                        "description": data["description"],
                        "output_image": data["output_image"],
                        "label": data["label"],
                    },
                )


if __name__ == "__main__":
    """
    cd PyTorch-Deep-Learning-Template
    python .\dataset\hf_dataset_sample\custom_hf_dataset.py
    """

    from datasets import load_dataset
    import matplotlib.pyplot as plt
    import os

    # 加载自定义数据集
    dataset = load_dataset(
        "dataset/hf_dataset_sample/custom_hf_dataset.py",
        split="train",
        trust_remote_code=True,
        num_proc=8,
    )
    print(f"current working directory:{os.getcwd()}")
    print("Dataset Info:")
    print(dataset.info)

    print("\nFirst Example:")
    print(dataset[0])

    # 可视化数据集中的前几个图像
    def visualize_dataset(dataset, num_samples=2):
        for i in range(num_samples):
            # 读取数据集中的第 i 个样本
            sample = dataset[i]

            # 获取输入和输出图像的路径
            input_img = sample["input_image"]
            output_img = sample["output_image"]

            # 绘制图像
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(input_img)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(output_img)
            axes[1].set_title("Output Image")
            axes[1].axis("off")

            plt.show()

    # Visualize the dataset
    visualize_dataset(dataset, num_samples=2)
