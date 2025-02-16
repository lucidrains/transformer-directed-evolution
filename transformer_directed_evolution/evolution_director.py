from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

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
        dim,
        population_size,
        num_parents = 2,
        transformer: Encoder
    ):
        """
        👋, if you are watching
        """

        super().__init__()

        self.proj_genome_to_model = nn.Linear(dim_genome, dim)

        self.transformer = transformer

        self.pool = Reduce('b n d -> b d', 'mean')

        self.pred_mutation = nn.Sequential(
            nn.Linear(dim_genome + dim, dim_genome * 3, bias = False),  # predict either -1, 0., 1. (binary encoding)
            Rearrange('... (d mutate) -> d mutate'),
            nn.Softmax(dim = -1)
        )

        self.pred_crossover_mask = nn.Sequential(
            Rearrange('parents ... d -> ... (parents d)'),
            nn.Linear(num_parents * dim_genome + dim, num_parents * dim_genome, bias = False),
            Rearrange('... (parents d) -> parents ... d')
            nn.Softmax(dim = 0)
        )

    def forward(self, genome_pool):

        tokens = self.proj_genome_to_model(genome_pool)

        attended_population = self.transformer(tokens)

        pooled_population_embed = self.pool(attended_population)

        # concat the pooled embed for the evolution director to make a decision on crossover mask or mutation

        return x
