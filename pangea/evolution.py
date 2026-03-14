"""
Evolution -- selection and mutation logic.
============================================================
Stateless functions that implement the genetic algorithm.

At the end of each generation:
    1. Evaluate fitness for all creatures
    2. Select the top performers
    3. Clone and mutate them to fill the next generation

No crossover -- mutation-only neuroevolution (standard for small NNs).
"""

from __future__ import annotations

import random

import numpy as np

from pangea.config import (
    EVOLUTION_POINTS,
    FITNESS_ENERGY_WEIGHT,
    FITNESS_FOOD_WEIGHT,
    FITNESS_TIME_WEIGHT,
    MUTATION_RATE,
    MUTATION_STRENGTH,
    POPULATION_SIZE,
    TOP_PERFORMERS_COUNT,
    TRAIT_MUTATION_RANGE,
)
from pangea.creature import Creature
from pangea.dna import DNA


# -- Fitness --------------------------------------------------------------


def evaluate_fitness(creature: Creature) -> float:
    """
    Calculate a creature's fitness score.

    Fitness = (food_eaten * 10) + (survival_time * 0.1) + (remaining_energy * 0.05)

    Food eaten is weighted most heavily because it's the primary survival skill.
    """
    return (
        creature.food_eaten * FITNESS_FOOD_WEIGHT
        + creature.age * FITNESS_TIME_WEIGHT
        + creature.energy * FITNESS_ENERGY_WEIGHT
    )


# -- Selection ------------------------------------------------------------


def select_top(
    creatures: list[Creature],
    n: int = TOP_PERFORMERS_COUNT,
) -> list[DNA]:
    """
    Select the top N creatures by fitness and return their DNA.

    Args:
        creatures: All creatures from the generation (alive or dead).
        n:         Number of top performers to keep.

    Returns:
        List of DNA from the top N creatures, sorted best-first.
    """
    ranked = sorted(creatures, key=evaluate_fitness, reverse=True)
    return [c.dna for c in ranked[:n]]


# -- Mutation -------------------------------------------------------------


def mutate_weights(
    weights: list[np.ndarray],
    rate: float = MUTATION_RATE,
    strength: float = MUTATION_STRENGTH,
) -> list[np.ndarray]:
    """
    Mutate neural network weights with Gaussian noise.

    For each weight value, with probability `rate`, add noise ~ N(0, strength).

    Args:
        weights:  List of weight arrays [W1, b1, W2, b2].
        rate:     Probability of mutating each individual weight.
        strength: Standard deviation of the Gaussian noise.

    Returns:
        New list of mutated weight arrays (originals are NOT modified).
    """
    mutated = []
    for w in weights:
        w_copy = w.copy()
        # Create a mask: True where mutation should happen
        mask = np.random.random(w_copy.shape) < rate
        # Add Gaussian noise where mask is True
        noise = np.random.randn(*w_copy.shape) * strength
        w_copy += mask * noise
        mutated.append(w_copy)
    return mutated


def mutate_traits(dna: DNA) -> tuple[int, int, int, int]:
    """
    Mutate physical trait allocations while preserving the budget.

    Each trait gets a random delta in [-TRAIT_MUTATION_RANGE, +TRAIT_MUTATION_RANGE].
    After applying deltas, traits are clamped to minimum 1 and rescaled to sum to
    EVOLUTION_POINTS.

    Args:
        dna: The parent DNA to mutate traits from.

    Returns:
        Tuple of (speed, size, vision, efficiency) after mutation.
    """
    traits = [dna.speed, dna.size, dna.vision, dna.efficiency]
    r = TRAIT_MUTATION_RANGE

    # Apply random deltas
    traits = [t + random.randint(-r, r) for t in traits]

    # Clamp to minimum of 1
    traits = [max(1, t) for t in traits]

    # Rescale to sum to EVOLUTION_POINTS
    total = sum(traits)
    if total != EVOLUTION_POINTS:
        # Proportional rescaling
        scale = EVOLUTION_POINTS / total
        traits = [max(1, round(t * scale)) for t in traits]

        # Fix any rounding error by adjusting the largest trait
        diff = EVOLUTION_POINTS - sum(traits)
        if diff != 0:
            largest_idx = traits.index(max(traits))
            traits[largest_idx] += diff

    return traits[0], traits[1], traits[2], traits[3]


# -- Next Generation -----------------------------------------------------


def create_next_generation(
    top_dna: list[DNA],
    population_size: int = POPULATION_SIZE,
    mutation_rate: float = MUTATION_RATE,
    mutation_strength: float = MUTATION_STRENGTH,
) -> list[DNA]:
    """
    Create the next generation by cloning and mutating top performers.

    Each top performer is cloned `population_size // len(top_dna)` times.
    Any remainder slots are filled by cloning from the best performers.
    Each clone has its weights and traits independently mutated.

    Args:
        top_dna:         List of DNA from the top performers.
        population_size: Target population size for the new generation.

    Returns:
        List of new mutated DNA objects.
    """
    if not top_dna:
        # Extinction -- start fresh with random DNA
        return [DNA.random() for _ in range(population_size)]

    new_generation: list[DNA] = []
    clones_per_parent = population_size // len(top_dna)
    remainder = population_size % len(top_dna)

    for i, parent in enumerate(top_dna):
        # How many clones for this parent
        count = clones_per_parent + (1 if i < remainder else 0)

        for _ in range(count):
            # Mutate weights (using provided rate and strength)
            new_weights = mutate_weights(parent.weights, mutation_rate, mutation_strength)
            # Mutate traits
            speed, size, vision, efficiency = mutate_traits(parent)
            # Create new DNA
            new_generation.append(
                DNA(
                    weights=new_weights,
                    speed=speed,
                    size=size,
                    vision=vision,
                    efficiency=efficiency,
                )
            )

    return new_generation


def create_next_generation_convergence(
    creatures: list[Creature],
    top_n: int = TOP_PERFORMERS_COUNT // 2,
) -> list[DNA]:
    """
    Create next generation for convergence mode (two competing lineages).

    Selection happens within each lineage independently.
    Top performers from each lineage are cloned to fill their half of the population.

    Args:
        creatures: All creatures from the generation.
        top_n:     Top performers to keep per lineage.

    Returns:
        List of new DNA objects, tagged lineages preserved via creature creation.
    """
    from pangea.config import CREATURES_PER_LINEAGE

    lineage_a = [c for c in creatures if c.lineage == "A"]
    lineage_b = [c for c in creatures if c.lineage == "B"]

    result: list[DNA] = []

    for group, lineage_tag in [(lineage_a, "A"), (lineage_b, "B")]:
        if not group:
            # Lineage went extinct -- no creatures to evolve
            continue

        top = select_top(group, min(top_n, len(group)))
        children = create_next_generation(top, CREATURES_PER_LINEAGE)
        # Tag each child DNA with lineage info (stored as metadata for creature creation)
        for child in children:
            child._lineage = lineage_tag  # type: ignore[attr-defined]
        result.extend(children)

    return result
