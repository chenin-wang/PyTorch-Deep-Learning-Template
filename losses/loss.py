"""Custom loss functions."""

import torch
import torch.nn as nn

from ..loggers.logging_colors import get_logger

logger = get_logger()


# pylint: disable=unused-argument
def my_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculates the mean squared error between the output and target tensors.

    This is a simple example loss function.

    Args:
        output (torch.Tensor): The model's output tensor.
        target (torch.Tensor): The ground truth target tensor.

    Returns:
        torch.Tensor: The calculated mean squared error loss.
    """
    loss = torch.mean((output - target) ** 2)
    return loss
