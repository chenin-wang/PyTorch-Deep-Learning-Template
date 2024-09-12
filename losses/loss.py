import torch
from ..loggers.logging_colors import get_logger
logger = get_logger(__name__)
# define custom losses
def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss
