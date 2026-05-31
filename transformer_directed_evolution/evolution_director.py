from __future__ import annotations

import wandb

import torch
from torch import nn, tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.distributions import Distribution, Categorical, Bernoulli

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

from x_transformers import Encoder

from evolutionary_policy_optimization import LatentGenePool

from tqdm import tqdm

from assoc_scan import AssocScan

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

# plackett luce distribution

class PlackettLuce(Distribution):
    arg_constraints = dict(
        logits = torch.distributions.constraints.real
    )

    def __init__(self, logits, validate_args = None):
        self.logits = logits
        super().__init__(
            batch_shape = logits.shape[:-1],
            event_shape = torch.Size(),
            validate_args = validate_args
        )

    def sample(self, sample_shape = torch.Size()):
        logits = self.logits

        for s in reversed(sample_shape):
            logits = repeat(logits, '... -> s ...', s = s)

        return (logits + gumbel_noise(logits)).argsort(dim = -1, descending = True)

    def log_prob(self, value):
        device = value.device
        k, n = value.shape[-1], self.logits.shape[-1]

        logits = self.logits
        if logits.ndim < value.ndim:
            logits = logits.expand(*value.shape[:-1], -1)

        log_num = logits.gather(-1, value)

        one_hot = F.one_hot(value, num_classes = n).float()

        tril = torch.tril(torch.ones(k, k, device = device), diagonal = -1)
        mask = einsum(tril, one_hot, 'i j, ... j n -> ... i n')

        logits = rearrange(self.logits, '... n -> ... 1 n')
        masked_logits = logits.masked_fill(mask.bool(), float('-inf'))

        log_denom = torch.logsumexp(masked_logits, dim = -1)
        return reduce(log_num - log_denom, '... k -> ...', 'sum')

    def entropy(self):
        return -self.log_prob(self.sample())

# gen advantage estimate

def calc_generalized_advantage_estimate(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    use_accelerated = default(use_accelerated, rewards.is_cuda)
    device = rewards.device

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[:-1], values[1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    return scan(gates, delta)

# the environment, which in this case, is a petri dish running genetic algorithm
# start with the most basic toy task before going for TSP

class ToyGeneticAlgorithmEnv(Module):
    def __init__(
        self,
        goal = 'Attention is all you need',
        population_size = 100,
        mutation_rate = 0.05,
        frac_fittest_survive = 0.25,
        frac_tournament = 0.25,
        display = False,
        max_steps = 500
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
        self.population_size = population_size

        self.display = display
        self.max_steps = max_steps

        self.register_buffer('target_gene', target_gene)

        self.reset()

    @property
    def device(self):
        return self.target_gene.device

    @property
    def diversity(self):
        pool = self.gene_pool
        pool_size = pool.shape[-2]
        num_pairs = (pool_size * pool_size) / 2 - pool_size
        distances = torch.cdist(pool.float(), pool.float())

        return distances.tril(-1).sum() / num_pairs

    def reset(self):
        self.register_buffer('initted', tensor(False, device = self.device))
        self.register_buffer('generation', tensor(0, device = self.device))
        self.register_buffer('parent_ids', torch.zeros((self.num_children, 2), device = self.device, dtype = torch.long))
        self.register_buffer('gene_pool', torch.randint(0, 255, (self.population_size, self.gene_length), device = self.device))
        self.register_buffer('done', tensor(False, device = self.device))

    def encode(self, s):
        return tensor([ord(c) for c in s])

    def decode(self, t):
        return ''.join([chr(i) for i in t.tolist()])

    def to_environment_generator(self):
        actions = yield self.gene_pool, self.parent_ids, None

        done = self.done.item()

        while not done:
            actions = default(actions, dict(display = self.display))

            done, fitnesses = self.forward(**actions)

            actions = yield self.gene_pool, self.parent_ids, fitnesses, self.diversity, done

    @torch.no_grad()
    def run(
        self,
        num_trials = 1,
        *,
        director: EvolutionDirector | None = None,
        display = None,
        pass_fitness_to_director = False
    ):
        if exists(director):
            director.eval()

        display = default(display, self.display)

        max_fitnesses = []
        generation_completed_at = []

        for _ in tqdm(range(num_trials), desc = 'trial'):
            self.reset()

            gen = self.to_environment_generator()

            state, parent_ids, fitnesses = next(gen)

            done = False

            step = 0

            while not done and step < self.max_steps:

                actions = dict(display = display)

                if exists(director):
                    kwargs = dict(fitnesses = fitnesses) if pass_fitness_to_director else {}
                    actions.update(director(state, parent_ids, **kwargs))

                state, parent_ids, fitnesses, diversity, done = gen.send(actions)

                step += 1

            generation_completed_at.append(self.generation.item())

            max_fitnesses.append(fitnesses.amax())

        completed_at = tensor(generation_completed_at, device = self.device)
        max_fitnesses = stack(max_fitnesses)

        return completed_at, max_fitnesses

    def forward(
        self,
        display = None,
        crossover_mask = None,
        mutation_rate = None,
        mutation_strength = 0.5
    ):
        display = default(display, self.display)
        device = self.target_gene.device

        # get the gene pool

        pool = self.gene_pool

        # if initted, carry out the crossover and mutation, taking into account any actions passed in

        if self.initted:
            parents = pool[self.parent_ids]

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

            self.register_buffer('gene_pool', pool)
            self.generation.add_(1)

        # sort population by fitness
        diff = pool - self.target_gene
        fitnesses = 1. / einsum(diff, diff, '... d, ... d -> ...').clamp(min = 1e-10)

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

        contender_ids = torch.randn((self.num_children, self.keep_fittest_len), device = self.device).argsort(dim = -1)[..., :self.num_tournament_contenders]
        tournaments = fitnesses[contender_ids]
        top2_tournament_indices = tournaments.topk(2, dim = -1, largest = True, sorted = False).indices

        top2_contender_ids = contender_ids.gather(-1, top2_tournament_indices)

        self.register_buffer('gene_pool', pool)
        self.parent_ids.copy_(top2_contender_ids)

        if not self.initted:
            self.initted.copy_(tensor(True, device = self.device))

        return False, fitnesses

# main class

class EvolutionDirector(Module):
    def __init__(
        self,
        dim_genome,
        transformer: Encoder | dict,
        mutation_rate_bins = 25,
        max_mutation_rate = 0.2,
        critic_fitness_weight = 1.,
        critic_diversity_weight = 0.5,
    ):
        """
        👋, if you are watching
        """

        super().__init__()

        if isinstance(transformer, dict):
            transformer = Encoder(**transformer)

        dim = transformer.dim

        self.proj_genome_to_model = nn.Linear(dim_genome, dim)

        self.to_fitness_embed = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim),
        )

        # shared stem

        self.transformer = transformer

        self.pool = Reduce('b n d -> b d', 'mean')

        # actor head

        self.mutation_rate_bins = mutation_rate_bins
        self.max_mutation_rate = max_mutation_rate

        self.pred_selection_operator = nn.Sequential(
            nn.Linear(dim, 1, bias = False),
            Rearrange('... 1 -> ...'),
        )

        self.pred_interfere_mutation = nn.Sequential(
            nn.Linear(dim_genome + dim, 1, bias = False),
            Rearrange('... 1 -> ...'),
        )

        self.pred_mutation = nn.Linear(dim, mutation_rate_bins, bias = False)

        self.pred_interfere_crossover = nn.Sequential(
            Rearrange('parents ... d -> ... (parents d)'),
            nn.Linear(2 * dim_genome + dim, 1, bias = False),
            Rearrange('... 1 -> ...'),
        )

        self.pred_crossover_mask = nn.Linear(2 * dim_genome + dim, dim_genome, bias = False)

        # critic head

        self.critic_fitness_weight = critic_fitness_weight
        self.critic_diversity_weight = critic_diversity_weight

        self.pred_value = nn.Sequential(
            Rearrange('parents ... d -> ... (parents d)'),
            nn.Linear(2 * dim_genome + dim, 2, bias = False),
            Rearrange('... 1 -> ...')
        )

    def critic_loss(
        self,
        advantages,
        values,
        rewards
    ):
        rewards = advantages + values
        return F.mse_loss(values, rewards)

    def actor_loss(
        self,
        dist: Distribution,
        actions,
        old_log_probs,
        advantages,
        eps_clip = 0.2,
        entropy_weight = .01,
        norm_eps = 1e-5
    ):
        advantages = F.layer_norm(advantages, advantages.shape[-1:], eps = norm_eps)

        fitness_advantage, diversity_advantage = advantages.unbind(dim = 0)
        weighted_advantages = fitness_advantage * self.critic_fitness_weight + diversity_advantage * self.critic_diversity_weight

        log_probs = dist.log_prob(actions)

        if log_probs.ndim > weighted_advantages.ndim:
            log_probs = reduce(log_probs, '... e -> ...', 'sum')

        ratio = (log_probs - old_log_probs).exp()

        # classic clipped surrogate loss from ppo
        clipped_ratio = ratio.clamp(min = 1. - eps_clip, max = 1. + eps_clip)
        surrogate_loss = -torch.min(clipped_ratio * weighted_advantages, ratio * weighted_advantages)

        # add entropy loss for exploration
        entropy = dist.entropy()
        if entropy.ndim > weighted_advantages.ndim:
            entropy = reduce(entropy, '... e -> ...', 'sum')

        return (surrogate_loss - entropy_weight * entropy).mean()

    def forward(
        self,
        genome_pool,
        parent_ids,
        fitnesses = None,
        pred_selection_operator = False,
        natural_selection_size = None,
        return_distributions = False
    ):
        parents = genome_pool[parent_ids]
        genome_pool = rearrange(genome_pool, '... -> 1 ...')

        tokens = self.proj_genome_to_model(genome_pool.float())

        if exists(fitnesses):
            fitness_tokens = self.to_fitness_embed(fitnesses)
            tokens = tokens + fitness_tokens

        attended_population = self.transformer(tokens)

        pool_stats_embed = self.pool(attended_population)

        # mutation rate
        mutation_rate_logits = self.pred_mutation(pool_stats_embed)
        mutation_rate_dist = Categorical(logits = mutation_rate_logits)
        mutation_rate = (mutation_rate_dist.sample().float() / self.mutation_rate_bins) * self.max_mutation_rate

        # crossover mask
        crossover_mask_input = cat((
            rearrange(parents, 'b parents d -> b (parents d)'),
            repeat(pool_stats_embed, '1 d -> b d', b = parents.shape[0])
        ), dim = -1)

        crossover_mask_logits = self.pred_crossover_mask(crossover_mask_input)
        crossover_mask_dist = Bernoulli(logits = crossover_mask_logits)

        crossover_mask = crossover_mask_dist.sample().bool()

        actions = dict(
            crossover_mask = crossover_mask,
            mutation_rate = mutation_rate
        )

        if return_distributions:
            actions.update(
                crossover_mask_dist = crossover_mask_dist,
                mutation_rate_dist = mutation_rate_dist
            )

        if pred_selection_operator:
            assert exists(natural_selection_size)

            selection_logits = self.pred_selection_operator(attended_population)
            plackett_luce = PlackettLuce(logits = selection_logits)

            sel_indices = plackett_luce.sample()[..., :natural_selection_size]
            selection_mask = torch.zeros_like(selection_logits).scatter(-1, sel_indices, True)

            actions.update(
                selection_mask = selection_mask,
                selection_indices = sel_indices,
            )

            if return_distributions:
                actions.update(plackett_luce_dist = plackett_luce)

        return actions

# quick test

if __name__ == '__main__':

    trials = 5
    petri_dish = ToyGeneticAlgorithmEnv()

    human = EvolutionDirector(
        petri_dish.gene_length,
        dict(
            dim = 64,
            depth = 2,
            attn_dim_head = 64,
            heads = 4,
        )
    )

    results_without_intervention, _ = petri_dish.run(trials)
    results_with_intervention, _ = petri_dish.run(trials, director = human, pass_fitness_to_director = True)

    assert results_without_intervention.shape == results_with_intervention.shape
