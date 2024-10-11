import torch
from typing import Optional


def eps(x: Optional[torch.Tensor] = None, /) -> float:
    """Return the `eps` value for the given `input` dtype. (default=float32 ~= 1.19e-7)"""
    dtype = torch.float32 if x is None else x.dtype
    return torch.finfo(dtype).eps
