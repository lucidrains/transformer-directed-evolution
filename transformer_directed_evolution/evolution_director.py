from __future__ import annotations

import torch
from torch import tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from x_transformers import Encoder

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# the environment, which in this case, is a petri dish running genetic algorithm
# start with the most basic toy task before going for TSP

class ToyGeneticAlgorithmEnv(Module):
    def __init__(
        self,
        goal = 'Attention is all you need',
        population_size = 100,
        mutation_rate = 0.05,
        frac_fittest_survive = 0.25,
        frac_tournament = 0.25
    ):  
        super().__init__()

        gene_length = len(goal)
        gene_midpoint = gene_length // 2
        target_gene = self.encode(goal)

        keep_fittest_len = int(population_size * frac_fittest_survive)
        num_tournament_contenders = int(keep_fittest_len * frac_tournament)
        num_children = population_size - keep_fittest_len
        num_mutate = mutation_rate * gene_length

        assert num_tournament_contenders >= 2

        self.gene_length = gene_length
        self.gene_midpoint = gene_midpoint
        self.keep_fittest_len = keep_fittest_len
        self.num_tournament_contenders = num_tournament_contenders
        self.num_children = num_children
        self.num_mutate = num_mutate

        self.register_buffer('target_gene', target_gene)
        self.register_buffer('generation', tensor(0))
        self.register_buffer('gene_pool', torch.randint(0, 255, (population_size, gene_length)))
        self.register_buffer('done', tensor(False))

    def encode(self, s):
        return torch.tensor([ord(c) for c in s])

    def decode(self, t):
        return ''.join([chr(i) for i in t.tolist()])

    def to_environment_generator(self):
        actions = yield self.gene_pool

        done = self.done.item()

        while not done:
            actions = default(actions, dict(display = True))

            done, fitnesses = self.forward(**actions)

            actions = yield self.gene_pool, fitnesses, done

    def forward(
        self,
        display = False,
        crossover_mask = None,
        mutation_rate = None,
        mutation_strength = 0.5
    ):
        device = self.target_gene.device

        pool = self.gene_pool

        # sort population by fitness

        fitnesses = 1. / torch.square(pool - self.target_gene).sum(dim = -1)

        indices = fitnesses.sort(descending = True).indices
        pool, fitnesses = pool[indices], fitnesses[indices]

        # keep the fittest

        pool, fitnesses = pool[:self.keep_fittest_len], fitnesses[:self.keep_fittest_len]

        # display every generation

        if display:
            for gene, fitness in zip(pool, fitnesses):
                print(f"{self.decode(gene)} ({fitness.item():.3f})")

        # solved if any fitness is inf

        if (fitnesses == float('inf')).any():
            self.done.copy_(tensor(True))
            return True, fitnesses

        # deterministic tournament selection - let top 2 winners become parents

        contender_ids = torch.randn((self.num_children, self.keep_fittest_len)).argsort(dim = -1)[..., :self.num_tournament_contenders]
        participants, tournaments = pool[contender_ids], fitnesses[contender_ids]
        top2_winners = tournaments.topk(2, dim = -1, largest = True, sorted = False).indices

        # parents = get_at('p [t] g, p w -> p w g', participants, top2_winners)

        top2_winners = repeat(top2_winners, 'p w -> p w g', g = participants.shape[-1])
        parents = participants.gather(1, top2_winners)

        # cross over recombination of parents

        parent1, parent2 = parents.unbind(dim = 1)

        if not exists(crossover_mask):
            crossover_mask = torch.randint(0, 2, parent1.shape, device = device).bool()

        children = torch.where(crossover_mask, parent1, parent2)

        pool = torch.cat((pool, children))

        # mutate genes in population

        num_mutate = self.num_mutate

        if exists(mutation_rate):
            num_mutate = mutation_rate * self.gene_length

        # mutate

        mutate_mask = torch.randn(pool.shape, device = device).argsort(dim = -1) < num_mutate

        noise = (torch.rand(pool.shape, device = device) < mutation_strength) * 2 - 1
        pool = torch.where(mutate_mask, pool + noise, pool)
        pool.clamp_(0, 255)

        self.gene_pool.copy_(pool)
        self.generation.add_(1)

        return False, fitnesses

# main class

class EvolutionDirector(Module):
    def __init__(
        self,
        dim_genome,
        dim = 64,
        transformer: Encoder,
        num_parents = 2,
    ):
        """
        👋, if you are watching
        """

        super().__init__()

        self.proj_genome_to_model = nn.Linear(dim_genome, dim)

        self.transformer = transformer

        self.pool = Reduce('b n d -> b d', 'mean')

        self.pred_interfere_mutation = nn.Linear(
            nn.Linear(dim_genome + dim, 1, bias = False),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()
        )

        self.pred_mutation = nn.Sequential(
            nn.Linear(dim_genome + dim, dim_genome * 3, bias = False),  # predict either -1, 0., 1. (binary encoding)
            Rearrange('... (d mutate) -> d mutate'),
            nn.Softmax(dim = -1)
        )

        self.pred_interfere_crossover = nn.Linear(
            Rearrange('parents ... d -> ... (parents d)'),
            nn.Linear(num_parents * dim_genome + dim, 1, bias = False),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()
        )

        self.pred_crossover_mask = nn.Sequential(
            Rearrange('parents ... d -> ... (parents d)'),
            nn.Linear(num_parents * dim_genome + dim, num_parents * dim_genome, bias = False),
            Rearrange('... (parents d) -> parents ... d'),
            nn.Softmax(dim = 0)
        )

    def forward(self, genome_pool):

        tokens = self.proj_genome_to_model(genome_pool)

        attended_population = self.transformer(tokens)

        pooled_population_embed = self.pool(attended_population)

        # concat the pooled embed for the evolution director to make a decision on crossover mask or mutation

        return x

# quick test

if __name__ == '__main__':

    toy = ToyGeneticAlgorithmEnv()

    god = EvolutionDirector(toy.gene_length)

    gen = toy.to_environment_generator()

    state = next(gen)

    done = False

    while not done:
        actions = god(state)

        state, _, done = gen.send(actions)
