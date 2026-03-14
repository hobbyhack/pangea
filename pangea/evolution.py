"""
Evolution -- selection and mutation logic.
============================================================
Stateless functions that implement the genetic algorithm.

At the end of each generation:
    1. Evaluate fitness for all creatures
    2. Select the top performers
    3. Clone and mutate them to fill the next generation

Supports optional crossover (sexual reproduction) to blend
two parents' NN weights and traits before mutation.
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


# -- Fitness ------------------------------------------------------------------


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


# -- Selection ----------------------------------------------------------------


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


# -- Mutation -----------------------------------------------------------------


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


# -- Crossover ----------------------------------------------------------------


def crossover_weights(
    parent_a_weights: list[np.ndarray],
    parent_b_weights: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Blend two parents' NN weights using a random 50/50 mask.

    For each weight array pair, create a random boolean mask and use
    np.where to pick from either parent.

    Args:
        parent_a_weights: Weight arrays [W1, b1, W2, b2] from parent A.
        parent_b_weights: Weight arrays [W1, b1, W2, b2] from parent B.

    Returns:
        New list of blended weight arrays.
    """
    blended = []
    for wa, wb in zip(parent_a_weights, parent_b_weights):
        mask = np.random.random(wa.shape) < 0.5
        blended.append(np.where(mask, wa, wb))
    return blended


def crossover_traits(
    parent_a: DNA,
    parent_b: DNA,
) -> tuple[int, int, int, int]:
    """
    Randomly pick each trait from one of two parents, then rescale.

    For each trait (speed, size, vision, efficiency), randomly pick
    from parent_a or parent_b. Rescale to sum to EVOLUTION_POINTS
    and ensure all traits >= 1.

    Args:
        parent_a: First parent DNA.
        parent_b: Second parent DNA.

    Returns:
        Tuple of (speed, size, vision, efficiency) after crossover.
    """
    a_traits = [parent_a.speed, parent_a.size, parent_a.vision, parent_a.efficiency]
    b_traits = [parent_b.speed, parent_b.size, parent_b.vision, parent_b.efficiency]

    # Randomly pick each trait from one parent
    traits = [
        a if random.random() < 0.5 else b
        for a, b in zip(a_traits, b_traits)
    ]

    # Clamp to minimum of 1
    traits = [max(1, t) for t in traits]

    # Rescale to sum to EVOLUTION_POINTS
    total = sum(traits)
    if total != EVOLUTION_POINTS:
        scale = EVOLUTION_POINTS / total
        traits = [max(1, round(t * scale)) for t in traits]

        # Fix any rounding error by adjusting the largest trait
        diff = EVOLUTION_POINTS - sum(traits)
        if diff != 0:
            largest_idx = traits.index(max(traits))
            traits[largest_idx] += diff

    return traits[0], traits[1], traits[2], traits[3]


# -- Next Generation ---------------------------------------------------------


def create_next_generation(
    top_dna: list[DNA],
    population_size: int = POPULATION_SIZE,
    mutation_rate: float = MUTATION_RATE,
    mutation_strength: float = MUTATION_STRENGTH,
    crossover: bool = False,
) -> list[DNA]:
    """
    Create the next generation by cloning and mutating top performers.

    Each top performer is cloned `population_size // len(top_dna)` times.
    Any remainder slots are filled by cloning from the best performers.
    Each clone has its weights and traits independently mutated.

    When crossover=True and at least 2 parents are available, pairs of
    parents are randomly selected, their weights and traits are blended
    via crossover, then mutation is applied to the offspring.

    Args:
        top_dna:           List of DNA from the top performers.
        population_size:   Target population size for the new generation.
        mutation_rate:      Probability of mutating each weight.
        mutation_strength:  Standard deviation of the Gaussian noise.
        crossover:         Whether to use sexual reproduction (crossover).

    Returns:
        List of new mutated DNA objects.
    """
    if not top_dna:
        # Extinction -- start fresh with random DNA
        return [DNA.random() for _ in range(population_size)]

    # Fall back to clone-and-mutate when crossover requested but only 1 parent
    use_crossover = crossover and len(top_dna) >= 2

    new_generation: list[DNA] = []

    if use_crossover:
        for _ in range(population_size):
            # Pick two distinct parents
            parent_a, parent_b = random.sample(top_dna, 2)

            # Crossover weights and traits
            child_weights = crossover_weights(parent_a.weights, parent_b.weights)
            speed, size, vision, efficiency = crossover_traits(parent_a, parent_b)

            # Apply mutation on top of crossover
            child_weights = mutate_weights(child_weights, mutation_rate, mutation_strength)

            # Create child DNA
            child_dna = DNA(
                weights=child_weights,
                speed=speed,
                size=size,
                vision=vision,
                efficiency=efficiency,
            )
            # Mutate traits
            speed, size, vision, efficiency = mutate_traits(child_dna)
            child_dna.speed = speed
            child_dna.size = size
            child_dna.vision = vision
            child_dna.efficiency = efficiency

            new_generation.append(child_dna)
    else:
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
