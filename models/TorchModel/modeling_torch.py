from typing import List, Tuple
import torch
import torch.nn as nn
from .utils import *
from ...loggers.logging_colors import get_logger
from transformers.modeling_utils import PreTrainedModel

logger = get_logger()

class CustomTorchModel(PreTrainedModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 50
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)