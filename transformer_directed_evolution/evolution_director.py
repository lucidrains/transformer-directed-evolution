from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

from x_transformer import Encoder

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class EvolutionDirector(Module):
    def __init__(
        self,
        dim_genome,
        population_size,
        num_parents = 2,
        transformer: Encoder
    ):
        """
        👋, if you are watching
        """

        super().__init__()

        self.transformer = transformer

        self.pred_crossover_mask = nn.Sequential(
            Rearrange('parents ... d -> ... (parents d)'),
            nn.Linear(num_parents * dim_genome, num_parents * dim_genome, bias = False),
            Rearrange('... (parents d) -> parents ... d')
            nn.Softmax(dim = 0)
        )

    def forward(self, x):
        return x
