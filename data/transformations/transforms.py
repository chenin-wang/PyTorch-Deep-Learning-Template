from torchvision.transforms import ColorJitter
from transformers import AutoImageProcessor
import numpy as np
import torchvision.transforms as T


jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)


def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs