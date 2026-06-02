# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "einx",
#     "einops",
#     "fire",
#     "accelerate",
#     "wandb",
#     "transformer-directed-evolution"
# ]
# ///

from __future__ import annotations

import fire
from accelerate import Accelerator

from einx import get_at, less, where
from einops import rearrange, repeat, reduce

import torch
from torch import tensor, cat, stack
from torch.nn import Module
from tqdm import tqdm

from transformer_directed_evolution.evolution_director import EvolutionDirector, default, exists

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def batch_randperm(shape, device = None):
    return torch.randn(shape, device = device).argsort(dim = -1)

# eml functions

def generate_paper_target(d):
    if d < 2:
        return "x"

    s = "x"
    for _ in range(d - 1):
        s = f"x - ln({s})"
    return s

def eml(x, y, ln_min = 1e-20):
    return torch.exp(x) - torch.log(y.clamp(min = ln_min))

def evaluate_target_string(target_str, x_vals):
    context = {
        'x': x_vals,
        'log': torch.log,
        'ln': torch.log,
        'exp': torch.exp,
        'eml': eml
    }
    return eval(target_str, {"__builtins__": {}}, context)

def decode_tree(tree_array):
    def _decode(idx):
        if idx >= len(tree_array):
            return "ERROR"
        token = tree_array[idx]
        if token == 0:
            return "1"
        elif token == 1:
            return "x"
        elif token == 2:
            left = 2 * idx + 1
            right = 2 * idx + 2
            return f"eml({_decode(left)}, {_decode(right)})"
        return "ERROR"
    return _decode(0)

# vectorized evaluator

def evaluate_population(islands, tree_depth, x_vals):
    device = islands.device
    
    node_values = where('i p g, , n -> i p g n', islands == 0, 1., x_vals)

    for d in reversed(range(tree_depth)):
        start_idx = 2 ** d - 1
        end_idx = 2 ** (d + 1) - 1

        parent_indices = torch.arange(start_idx, end_idx, device = device)
        left_indices = 2 * parent_indices + 1
        right_indices = 2 * parent_indices + 2

        left_vals = node_values[:, :, left_indices, :]
        right_vals = node_values[:, :, right_indices, :]

        eml_vals = eml(left_vals, right_vals)

        node_values[:, :, parent_indices, :] = where(
            'i p k, i p k n, i p k n -> i p k n',
            islands[:, :, parent_indices] == 2,
            eml_vals,
            node_values[:, :, parent_indices, :]
        )

    return node_values[:, :, 0, :]

# environment container

class EMLSymbolicRegressionEnv(Module):
    def __init__(
        self,
        target: str | None = None,
        depth: int | None = None,
        tree_depth: int | None = None,
        num_points = 100,
        pop_size = 1000,
        num_islands = 10,
        power_law_beta = 1.1,
        strong_mutation_rate = 0.25,
        frac_fittest_survive = 0.25,
        frac_tournament = 0.10,
        elite_frac = 0.05,
        migrate_every = 250,
        frac_migrate = 0.1,
        max_elite_age = 5,
        fuss_parent_selection = False,
        fuss_eps = 1e-5,
        max_steps = 500,
        display = False,
        device = 'cpu'
    ):
        super().__init__()
        
        assert not (exists(target) and exists(depth)), 'cannot pass in both depth and target'

        if not exists(target) and not exists(depth):
            depth = 3

        if exists(depth):
            target = generate_paper_target(depth)
            tree_depth = default(tree_depth, depth + 2)
        else:
            tree_depth = default(tree_depth, 5)

        self.target = target
        self.depth = depth
        self.tree_depth = tree_depth

        self.num_points = num_points
        self.pop_size = pop_size
        self.num_islands = num_islands

        self.power_law_beta = power_law_beta
        self.strong_mutation_rate = strong_mutation_rate
        self.frac_fittest_survive = frac_fittest_survive
        self.frac_tournament = frac_tournament
        self.elite_frac = elite_frac
        self.migrate_every = migrate_every
        self.frac_migrate = frac_migrate
        self.max_elite_age = max_elite_age
        self.fuss_parent_selection = fuss_parent_selection
        self.fuss_eps = fuss_eps

        self.max_steps = max_steps
        self.display = display
        self._device = device

        self.num_nodes = 2 ** (self.tree_depth + 1) - 1
        self.num_internal = 2 ** self.tree_depth - 1

        self.x_vals = torch.linspace(3.0, 5.0, self.num_points, device = device)
        self.y_target = evaluate_target_string(self.target, self.x_vals)

        self.keep_fittest_len = int(pop_size * frac_fittest_survive)
        self.num_elite = int(pop_size * elite_frac)
        self.num_tournament_contenders = max(2, int(self.keep_fittest_len * frac_tournament))
        self.num_children = pop_size - self.keep_fittest_len
        self.num_migrants = int(pop_size * frac_migrate)

        self.strong_num_mutate = int(strong_mutation_rate * self.num_nodes)
        
        power_law_cdf = torch.linspace(1, self.num_nodes, self.num_nodes, device = device).pow(-power_law_beta).cumsum(dim = -1)
        self.register_buffer('power_law_cdf', power_law_cdf / power_law_cdf[-1])
        
        self.reset()

    @property
    def device(self):
        return self._device

    @property
    def diversity(self):
        pool = rearrange(self.islands, 'i p g -> (i p) g')
        pool_size = pool.shape[0]

        if pool_size > 1000:
            indices = torch.randperm(pool_size, device = self.device)[:1000]
            pool = pool[indices]
            pool_size = 1000

        distances = torch.cdist(pool.float(), pool.float())
        num_pairs = (pool_size * pool_size) / 2 - pool_size

        return distances.tril(-1).sum() / num_pairs

    def reset(self):
        self.register_buffer('initted', tensor(False, device = self.device))
        self.register_buffer('generation', tensor(0, device = self.device))
        self.register_buffer('done', tensor(False, device = self.device))

        islands = torch.randint(0, 3, (self.num_islands, self.pop_size, self.num_nodes), device = self.device)
        islands[..., self.num_internal:] = islands[..., self.num_internal:].clamp(0, 1)
        self.register_buffer('islands', islands)

        ages = torch.zeros((self.num_islands, self.pop_size), dtype = torch.long, device = self.device)
        self.register_buffer('ages', ages)

        self.register_buffer('flat_parent_ids', torch.zeros((self.num_islands * self.num_children, 2), dtype = torch.long, device = self.device))

    def to_environment_generator(self):
        flat_pool = rearrange(self.islands, 'i p g -> (i p) g')
        actions = yield flat_pool, self.flat_parent_ids, None

        done = self.done.item()

        while not done:
            actions = default(actions, dict(display = self.display))
            done, flat_fitnesses = self.forward(**actions)

            flat_pool = rearrange(self.islands, 'i p g -> (i p) g')
            actions = yield flat_pool, self.flat_parent_ids, flat_fitnesses, self.diversity, done

    @torch.no_grad()
    def run(
        self,
        num_trials = 1,
        *,
        director = None,
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

            if display:
                best_idx = fitnesses.argmax()
                best_tree = state[best_idx].tolist()
                best_mse = (1. / fitnesses[best_idx].item()) - 1e-8
                print(f"\n--- trial complete ---")
                print(f"global best mse: {best_mse:.6f}")
                print(f"discovered expression: {decode_tree(best_tree)}\n")

        completed_at = tensor(generation_completed_at, device = self.device)
        max_fitnesses = stack(max_fitnesses)

        return completed_at, max_fitnesses

    def forward(
        self,
        display = None,
        crossover_mask = None,
        mutation_rate = None,
        mutation_strength = None
    ):
        display = default(display, self.display)
        device = self.device

        if self.initted:
            flat_pool = rearrange(self.islands, 'i p g -> (i p) g')
            parents = flat_pool[self.flat_parent_ids]
            
            queen, drone = parents[:, 0, :], parents[:, 1, :]

            strong_mutate_mask = batch_randperm(drone.shape, device = device) < self.strong_num_mutate
            noise = torch.randint(0, 3, drone.shape, device = device)
            
            mutated_drone = where('b g, b g, b g -> b g', strong_mutate_mask, noise, drone)
            mutated_drone[..., self.num_internal:] = mutated_drone[..., self.num_internal:].clamp(0, 1)

            if exists(crossover_mask):
                children = where('b g, b g, b g -> b g', crossover_mask, mutated_drone, queen)
            else:
                mutated_drone_islands = rearrange(mutated_drone, '(i c) g -> i c g', i = self.num_islands)
                queen_islands = rearrange(queen, '(i c) g -> i c g', i = self.num_islands)

                uniform_mask = torch.randint(0, 2, mutated_drone_islands.shape, device = device).bool()
                cut_points = torch.randint(0, self.num_nodes, (self.num_islands, self.num_children, 1), device = device)
                indices = torch.arange(self.num_nodes, device = device)
                traditional_mask = indices < cut_points

                is_uniform = torch.arange(self.num_islands, device = device) % 2 == 0
                c_mask = where('i, i c g, i c g -> i c g', is_uniform, uniform_mask, traditional_mask)

                children_islands = where('i c g, i c g, i c g -> i c g', c_mask, mutated_drone_islands, queen_islands)
                children = rearrange(children_islands, 'i c g -> (i c) g')

            children_islands = rearrange(children, '(i c) g -> i c g', i = self.num_islands)
            
            self.islands.copy_(cat((self.islands[:, :self.keep_fittest_len], children_islands), dim = 1))
            
            children_ages = torch.zeros((self.num_islands, self.num_children), dtype = torch.long, device = device)
            self.ages.copy_(cat((self.ages[:, :self.keep_fittest_len], children_ages), dim = 1))

            if exists(mutation_rate):
                num_mutate = (mutation_rate * self.num_nodes).long()
                mutate_mask = batch_randperm(self.islands.shape, device = device) < num_mutate.unsqueeze(-1).unsqueeze(-1)
            else:
                rand_probs = torch.rand((self.num_islands, self.pop_size), device = device)
                num_mutate = torch.searchsorted(self.power_law_cdf, rand_probs)
                mutate_mask = less('i p g, i p -> i p g', batch_randperm(self.islands.shape, device = device), num_mutate)

            is_elite = torch.arange(self.pop_size, device = device) < self.num_elite
            is_elite = rearrange(is_elite, 'p -> 1 p 1')
            
            is_too_old = self.ages >= self.max_elite_age
            is_too_old = rearrange(is_too_old, 'i p -> i p 1')
            
            protect_mask = is_elite & ~is_too_old
            mutate_mask = mutate_mask & ~protect_mask

            noise = torch.randint(0, 3, self.islands.shape, device = device)
            self.islands.copy_(where('i p g, i p g, i p g -> i p g', mutate_mask, noise, self.islands))
            self.islands[..., self.num_internal:] = self.islands[..., self.num_internal:].clamp(0, 1)

            has_mutated = mutate_mask.any(dim = -1)
            self.ages.copy_(where('i p, , i p -> i p', has_mutated, 0, self.ages))

            if divisible_by(self.generation.item(), self.migrate_every) and self.generation.item() > 0:
                island_rand_order = batch_randperm((self.num_islands, self.pop_size), device = device)
                
                self.islands.copy_(get_at('i [p1] g, i p2 -> i p2 g', self.islands, island_rand_order))
                self.ages.copy_(get_at('i [a1], i a2 -> i a2', self.ages, island_rand_order))

                migrants, remaining = self.islands[:, :self.num_migrants], self.islands[:, self.num_migrants:]
                migrants = torch.roll(migrants, 1, dims = 0)
                self.islands.copy_(cat((migrants, remaining), dim = 1))

                migrant_ages, remaining_ages = self.ages[:, :self.num_migrants], self.ages[:, self.num_migrants:]
                migrant_ages = torch.roll(migrant_ages, 1, dims = 0)
                self.ages.copy_(cat((migrant_ages, remaining_ages), dim = 1))
                
            self.generation.add_(1)

        y_pred = evaluate_population(self.islands, self.tree_depth, self.x_vals)
        mse = (y_pred - self.y_target).pow(2).mean(dim = -1)
        
        inf_tensor = tensor(float('inf'), device = device)
        mse = where('i p, , i p -> i p', torch.isnan(mse) | torch.isinf(mse), inf_tensor, mse)
        island_fitnesses = 1. / (mse + 1e-8)

        indices = island_fitnesses.sort(descending = True, dim = -1).indices
        
        self.islands.copy_(get_at('i [p1] g, i p2 -> i p2 g', self.islands, indices))
        island_fitnesses = get_at('i [f1], i f2 -> i f2', island_fitnesses, indices)
        self.ages.copy_(get_at('i [a1], i a2 -> i a2', self.ages, indices))

        self.ages.add_(1)

        best_island_idx = island_fitnesses[:, 0].argmax()
        best_fitness = island_fitnesses[best_island_idx, 0].item()
        best_mse = (1. / best_fitness) - 1e-8

        if display and (self.generation.item() == 0 or divisible_by(self.generation.item(), 10)):
            print(f"generation {self.generation.item()} | best mse: {best_mse:.6f}")

        if best_mse < 1e-7:
            self.done.copy_(tensor(True, device = device))
            if display:
                print(f"exact match found at gen {self.generation.item()}!")
            return True, rearrange(island_fitnesses, 'i p -> (i p)')

        if self.fuss_parent_selection:
            sorted_fitness = island_fitnesses.flip(dims = (-1,))
            padded = cat((sorted_fitness[:, :1], sorted_fitness, sorted_fitness[:, -1:]), dim = -1)
            voronoi_cell_sizes = (padded[:, 2:] - padded[:, :-2]) / 2
            
            selected = torch.multinomial(voronoi_cell_sizes + self.fuss_eps, self.num_children, replacement = True)
            drone_indices = (self.pop_size - 1) - selected
        else:
            contender_ids = batch_randperm((self.num_islands, self.num_children, self.pop_size), device = device)[..., :self.num_tournament_contenders]
            tournament_results = get_at('i [f], i c tf -> i c tf', island_fitnesses, contender_ids)
            top1_winner = tournament_results.topk(1, dim = -1, largest = True, sorted = False).indices
            drone_indices = contender_ids.gather(-1, top1_winner).squeeze(-1) # get_at is fine too but gather is clean here

        queen_indices = torch.zeros((self.num_islands, self.num_children), dtype = torch.long, device = device)

        island_offsets = rearrange(torch.arange(self.num_islands, device = device) * self.pop_size, 'i -> i 1')
        
        flat_queen_indices = queen_indices + island_offsets
        flat_drone_indices = drone_indices + island_offsets
        
        flat_parent_ids = rearrange(stack((flat_queen_indices, flat_drone_indices), dim = -1), 'i c p -> (i c) p')
        self.flat_parent_ids.copy_(flat_parent_ids)

        if not self.initted:
            self.initted.copy_(tensor(True, device = device))

        return False, rearrange(island_fitnesses, 'i p -> (i p)')

# main execution

def run(
    target: str | None = None,
    depth: int | None = None,
    tree_depth: int | None = None,
    num_points = 100,
    pop_size = 1000,
    num_islands = 10,
    generations = 500,
    power_law_beta = 1.1,
    strong_mutation_rate = 0.25,
    frac_fittest_survive = 0.25,
    frac_tournament = 0.10,
    elite_frac = 0.05,
    migrate_every = 250,
    frac_migrate = 0.1,
    max_elite_age = 5,
    fuss_parent_selection = False,
    fuss_eps = 1e-5,
    cpu = False,
    seed = 42,
    trials = 1,
    use_director = True
):
    torch.manual_seed(seed)
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device
    print(f"using device: {device}")

    env = EMLSymbolicRegressionEnv(
        target = target,
        depth = depth,
        tree_depth = tree_depth,
        num_points = num_points,
        pop_size = pop_size,
        num_islands = num_islands,
        power_law_beta = power_law_beta,
        strong_mutation_rate = strong_mutation_rate,
        frac_fittest_survive = frac_fittest_survive,
        frac_tournament = frac_tournament,
        elite_frac = elite_frac,
        migrate_every = migrate_every,
        frac_migrate = frac_migrate,
        max_elite_age = max_elite_age,
        fuss_parent_selection = fuss_parent_selection,
        fuss_eps = fuss_eps,
        max_steps = generations,
        display = True,
        device = device
    ).to(device)

    print(f"\ntarget: {env.target}")

    if exists(depth):
        print(f"target depth: {depth}")

    print(f"tree depth: {env.tree_depth}")
    print(f"population: {env.num_islands} islands x {env.pop_size} individuals")
    print("\n")

    director = None

    if use_director:
        director = EvolutionDirector(
            env.num_nodes,
            dict(
                dim = 64,
                depth = 2,
                attn_dim_head = 64,
                heads = 4,
            )
        ).to(device)
        print("Using EvolutionDirector intervention.\n")

    completed_at, max_fitnesses = env.run(
        num_trials = trials,
        director = director,
        pass_fitness_to_director = True
    )

    print("\n--- training complete ---")
    print(f"Completed at generations: {completed_at.tolist()}")

if __name__ == "__main__":
    fire.Fire(run)
