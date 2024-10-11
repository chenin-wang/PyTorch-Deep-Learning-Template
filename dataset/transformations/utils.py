import torch

from torch import Tensor
from typing import Optional, Sequence, Union
from ...utils.decorators import allow_np
import numpy as np
from PIL import Image
from numpy.typing import NDArray
from ...utils.common import eps

StatsRGB = tuple[float, float, float]
_mean = (0.485, 0.456, 0.406)  # ImageNet mean
_std = (0.229, 0.224, 0.225)  # ImageNet std
_coeffs = (0.299, 0.587, 0.114)  # Grayscale coefficients


@allow_np(permute=True)
def standardize(x: Tensor, /, mean: StatsRGB = _mean, std: StatsRGB = _std) -> Tensor:
    """Apply standardization. Default uses ImageNet statistics."""
    shape = [1] * (x.ndim - 3) + [3, 1, 1]
    mean = x.new_tensor(mean).view(shape)
    std = x.new_tensor(std).view(shape)
    x = (x - mean) / std
    return x


@allow_np(permute=True)
def unstandardize(x: Tensor, /, mean: StatsRGB = _mean, std: StatsRGB = _std) -> Tensor:
    """Remove standardization. Default uses ImageNet statistics."""
    shape = [1] * (x.ndim - 3) + [3, 1, 1]
    mean = x.new_tensor(mean).view(shape)
    std = x.new_tensor(std).view(shape)
    x = x * std + mean
    return x


@allow_np(permute=True)
def to_gray(x: Tensor, /, coeffs: StatsRGB = _coeffs, keepdim: bool = False) -> Tensor:
    """Convert image to grayscale."""
    shape = [1] * (x.ndim - 3) + [3, 1, 1]
    coeffs = x.new_tensor(coeffs).view(shape)
    x = (x * coeffs).sum(dim=1, keepdim=keepdim)
    return x


def mean_normalize(x: Tensor, /, dim: Union[int, Sequence[int]] = (2, 3)) -> Tensor:
    """Apply mean normalization across the specified dimensions.

    :param x: (Tensor) (*) Input tensor to normalize of any shape.
    :param dim: (int | Sequence[int]) Dimension(s) to compute the mean across.
    :return: (Tensor) (*) Mean normalized input with the same shape.
    """
    return x / x.mean(dim=dim, keepdim=True).clamp(min=eps(x))


# IMAGE CONVERSION
# ------------------------------------------------------------------------------
def pil2np(img: Image, /) -> NDArray:
    """Convert PIL image [0, 255] into numpy [0, 1]."""
    return np.array(img, dtype=np.float32) / 255.0  # Default is float64!


def np2pil(arr: NDArray, /) -> Image:
    """Convert numpy image [0, 1] into PIL [0, 255]."""
    return Image.fromarray((arr * 255).astype(np.uint8))


# ------------------------------------------------------------------------------
