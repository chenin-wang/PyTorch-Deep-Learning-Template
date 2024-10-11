"""sample dataset.
https://huggingface.co/docs/datasets/v3.0.0/en/dataset_script
"""
import json
import os
import datasets

_CITATION = """\
@InProceedings{dataset:dataset,
title = {A great new dataset},
author={author, Inc.
},
year={2024}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This is a sample dataset.
"""

_HOMEPAGE = ""
_LICENSE = "MIT"
_URLS = {
    "first_domain": "E:\code_learning\PyTorch-Deep-Learning-Template\dataset\hf_dataset_sample",
}


class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="first_domain",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        )
    ]

    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if (
            self.config.name == "first_domain"
        ):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "input_image": datasets.Image(decode=True),
                    "description": datasets.Value("string"),
                    "output_image": datasets.Image(decode=True),
                    "label": datasets.ClassLabel(names=["0", "1"]),
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)
        data_dir = _URLS[self.config.name]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "val.jsonl"),
                    "split": "val",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "first_domain":
                    # Yields examples as (key, example) tuples
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

    # 可视化前4个样本
    visualize_dataset(dataset, num_samples=2)

    # 打印数据集基本信息
    print(dataset.info)
