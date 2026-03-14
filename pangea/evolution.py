"""
Evolution — selection and mutation logic.
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
from typing import TYPE_CHECKING

import numpy as np

from pangea.config import (
    DIET_CARNIVORE,
    DIET_HERBIVORE,
    DIET_SCAVENGER,
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

DIET_MUTATION_RATE = 0.05  # 5% chance diet changes per offspring
from pangea.creature import Creature
from pangea.dna import DNA

if TYPE_CHECKING:
    from pangea.settings import SimSettings


# ── Fitness ──────────────────────────────────────────────────


def evaluate_fitness(
    creature: Creature,
    settings: SimSettings | None = None,
) -> float:
    """
    Calculate a creature's fitness score.

    Fitness = (food_eaten * food_weight) + (survival_time * time_weight)
              + (remaining_energy * energy_weight)
              + (territory_cells * territory_weight)

    Food eaten is weighted most heavily because it's the primary survival skill.
    Weights come from settings if provided, otherwise from config defaults.
    """
    food_w = settings.fitness_food_weight if settings else FITNESS_FOOD_WEIGHT
    time_w = settings.fitness_time_weight if settings else FITNESS_TIME_WEIGHT
    energy_w = settings.fitness_energy_weight if settings else FITNESS_ENERGY_WEIGHT
    territory_w = settings.territory_fitness_weight if settings else 0.0
    return (
        creature.food_eaten * food_w
        + creature.age * time_w
        + creature.energy * energy_w
        + len(creature.territory_cells) * territory_w
    )


# ── Selection ────────────────────────────────────────────────


def select_top(
    creatures: list[Creature],
    n: int = TOP_PERFORMERS_COUNT,
    settings: SimSettings | None = None,
) -> list[DNA]:
    """
    Select the top N creatures by fitness and return their DNA.

    Args:
        creatures: All creatures from the generation (alive or dead).
        n:         Number of top performers to keep.
        settings:  Optional SimSettings for fitness weights.

    Returns:
        List of DNA from the top N creatures, sorted best-first.
    """
    ranked = sorted(creatures, key=lambda c: evaluate_fitness(c, settings), reverse=True)
    return [c.dna for c in ranked[:n]]


# ── Mutation ─────────────────────────────────────────────────


def mutate_weights(
    weights: list[np.ndarray],
    rate: float = MUTATION_RATE,
    strength: float = MUTATION_STRENGTH,
    weight_clamp: float = 0.0,
) -> list[np.ndarray]:
    """
    Mutate neural network weights with Gaussian noise.

    For each weight value, with probability `rate`, add noise ~ N(0, strength).
    If weight_clamp > 0, all weights are clamped to [-clamp, +clamp].

    Args:
        weights:      List of weight arrays [W1, b1, W2, b2].
        rate:         Probability of mutating each individual weight.
        strength:     Standard deviation of the Gaussian noise.
        weight_clamp: Max absolute weight value (0 = no clamping).

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
        if weight_clamp > 0:
            np.clip(w_copy, -weight_clamp, weight_clamp, out=w_copy)
        mutated.append(w_copy)
    return mutated


def mutate_traits(
    dna: DNA,
    mutation_range: int = TRAIT_MUTATION_RANGE,
) -> tuple[int, int, int, int, int]:
    """
    Mutate physical trait allocations while preserving the budget.

    Each trait gets a random delta in [-mutation_range, +mutation_range].
    After applying deltas, traits are clamped to minimum 1 and rescaled to sum to
    EVOLUTION_POINTS.

    Args:
        dna:            The parent DNA to mutate traits from.
        mutation_range: Max +/- change per trait.

    Returns:
        Tuple of (speed, size, vision, efficiency, lifespan) after mutation.
    """
    traits = [dna.speed, dna.size, dna.vision, dna.efficiency, dna.lifespan]
    r = mutation_range

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

    return traits[0], traits[1], traits[2], traits[3], traits[4]


# ── Crossover ────────────────────────────────────────────────


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
) -> tuple[int, int, int, int, int]:
    """
    Randomly pick each trait from one of two parents, then rescale.

    For each trait (speed, size, vision, efficiency, lifespan), randomly pick
    from parent_a or parent_b. Rescale to sum to EVOLUTION_POINTS
    and ensure all traits >= 1.

    Args:
        parent_a: First parent DNA.
        parent_b: Second parent DNA.

    Returns:
        Tuple of (speed, size, vision, efficiency, lifespan) after crossover.
    """
    a_traits = [parent_a.speed, parent_a.size, parent_a.vision, parent_a.efficiency, parent_a.lifespan]
    b_traits = [parent_b.speed, parent_b.size, parent_b.vision, parent_b.efficiency, parent_b.lifespan]

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

    return traits[0], traits[1], traits[2], traits[3], traits[4]


# ── Next Generation ─────────────────────────────────────────


def create_next_generation(
    top_dna: list[DNA],
    population_size: int = POPULATION_SIZE,
    mutation_rate: float = MUTATION_RATE,
    mutation_strength: float = MUTATION_STRENGTH,
    crossover_rate: float = 0.0,
    min_parents: int = 0,
    weight_clamp: float = 0.0,
    trait_mutation_range: int = TRAIT_MUTATION_RANGE,
) -> list[DNA]:
    """
    Create the next generation by cloning and mutating top performers.

    Per offspring, with probability crossover_rate (if >= 2 parents),
    two parents are blended via crossover; otherwise a single parent
    is cloned.  Mutation is always applied on top.

    If min_parents > 0 and fewer parents survived, random DNA is
    injected to reach the minimum parent pool size.

    Args:
        top_dna:              List of DNA from the top performers.
        population_size:      Target population size for the new generation.
        mutation_rate:        Probability of mutating each weight.
        mutation_strength:    Standard deviation of the Gaussian noise.
        crossover_rate:       Per-offspring probability of crossover (0-1).
        min_parents:          Minimum parent pool size (pad with random DNA).
        weight_clamp:         Max absolute weight value (0 = no clamp).
        trait_mutation_range: Max +/- change per trait per generation.

    Returns:
        List of new mutated DNA objects.
    """
    if not top_dna:
        # Extinction — start fresh with random DNA
        return [DNA.random() for _ in range(population_size)]

    # Ensure minimum parent diversity
    if min_parents > 0 and len(top_dna) < min_parents:
        top_dna = list(top_dna)  # don't mutate caller's list
        while len(top_dna) < min_parents:
            top_dna.append(DNA.random())

    can_crossover = crossover_rate > 0 and len(top_dna) >= 2

    new_generation: list[DNA] = []

    for i in range(population_size):
        if can_crossover and random.random() < crossover_rate:
            # Crossover: blend two parents
            parent_a, parent_b = random.sample(top_dna, 2)
            child_weights = crossover_weights(
                parent_a.weights, parent_b.weights,
            )
            speed, size, vision, efficiency, lifespan = crossover_traits(
                parent_a, parent_b,
            )
            diet = random.choice([parent_a.diet, parent_b.diet])
        else:
            # Clone a single parent (distribute evenly)
            parent = top_dna[i % len(top_dna)]
            child_weights = [w.copy() for w in parent.weights]
            speed = parent.speed
            size = parent.size
            vision = parent.vision
            efficiency = parent.efficiency
            lifespan = parent.lifespan
            diet = parent.diet

        # Mutate weights
        child_weights = mutate_weights(
            child_weights, mutation_rate, mutation_strength, weight_clamp,
        )

        # Mutate diet (small chance to switch)
        if random.random() < DIET_MUTATION_RATE:
            diet = random.choice([DIET_HERBIVORE, DIET_CARNIVORE, DIET_SCAVENGER])

        # Create child DNA and mutate traits
        child_dna = DNA(
            weights=child_weights,
            speed=speed, size=size, vision=vision,
            efficiency=efficiency, lifespan=lifespan,
            diet=diet,
        )
        s, sz, v, e, lf = mutate_traits(child_dna, trait_mutation_range)
        child_dna.speed = s
        child_dna.size = sz
        child_dna.vision = v
        child_dna.efficiency = e
        child_dna.lifespan = lf

        new_generation.append(child_dna)

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
            # Lineage went extinct — no creatures to evolve
            continue

        top = select_top(group, min(top_n, len(group)))
        children = create_next_generation(top, CREATURES_PER_LINEAGE)
        # Tag each child DNA with lineage info (stored as metadata for creature creation)
        for child in children:
            child._lineage = lineage_tag  # type: ignore[attr-defined]
        result.extend(children)

    return result
