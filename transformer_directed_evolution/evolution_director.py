from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange

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
        transformer: Encoder
    ):
        super().__init__()

        self.transformer = transformer

    def forward(self, x):
        return x
