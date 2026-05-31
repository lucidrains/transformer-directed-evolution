import torch
import pytest
from transformer_directed_evolution.evolution_director import EvolutionDirector, ToyGeneticAlgorithmEnv, PlackettLuce

def test_evolution_director_e2e():
    trials = 1
    petri_dish = ToyGeneticAlgorithmEnv(population_size=100, max_steps=5)

    human = EvolutionDirector(
        petri_dish.gene_length,
        dict(
            dim = 16,
            depth = 1,
            attn_dim_head = 16,
            heads = 2,
        )
    )

    # Test running without director
    results_without_intervention, _ = petri_dish.run(trials)

    # Test running with director (default actions)
    results_with_intervention, _ = petri_dish.run(trials, director = human, pass_fitness_to_director = True)

    assert results_without_intervention.shape == results_with_intervention.shape

def test_evolution_director_selection_operator():
    # Test specifically the selection operator pathway
    petri_dish = ToyGeneticAlgorithmEnv(population_size=100)

    human = EvolutionDirector(
        petri_dish.gene_length,
        dict(
            dim = 16,
            depth = 1,
            attn_dim_head = 16,
            heads = 2,
        )
    )

    petri_dish.reset()
    gen = petri_dish.to_environment_generator()
    state, parent_ids, fitnesses = next(gen)

    # Ask for selection operator prediction
    actions = human(
        state,
        parent_ids,
        fitnesses=fitnesses,
        pred_selection_operator=True,
        natural_selection_size=10,
        return_distributions=True
    )

    assert 'selection_mask' in actions
    assert 'selection_indices' in actions
    assert 'plackett_luce_dist' in actions

    mask = actions['selection_mask']
    indices = actions['selection_indices']
    dist = actions['plackett_luce_dist']

    assert mask.shape[-1] == 100
    assert mask.sum() == 10
    assert indices.shape[-1] == 10

    # Test log_prob of PlackettLuce distribution
    log_probs = dist.log_prob(indices)
    assert log_probs.shape == indices.shape[:-1]

def test_plackett_luce_distribution():
    # Test PlackettLuce standalone
    logits = torch.randn(2, 5) # batch of 2, 5 items
    dist = PlackettLuce(logits=logits)

    # Sample a permutation
    sample = dist.sample()
    assert sample.shape == (2, 5)

    # Sample top 3
    sample_top3 = sample[..., :3]
    log_prob = dist.log_prob(sample_top3)
    assert log_prob.shape == (2,)
