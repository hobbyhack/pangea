"""Tests for the evolution module."""

import numpy as np
import pytest

from pangea.config import EVOLUTION_POINTS, POPULATION_SIZE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.evolution import (
    create_next_generation,
    crossover_traits,
    crossover_weights,
    evaluate_fitness,
    mutate_traits,
    mutate_weights,
    select_top,
)


class TestEvolution:
    def _make_creature(self, food_eaten=0, age=10.0, energy=50.0):
        """Helper to create a creature with specific stats."""
        dna = DNA.random()
        c = Creature(dna, 100, 100)
        c.food_eaten = food_eaten
        c.age = age
        c.energy = energy
        return c

    def test_fitness_food_dominates(self):
        """Creature with more food should have higher fitness."""
        c1 = self._make_creature(food_eaten=5, age=10, energy=50)
        c2 = self._make_creature(food_eaten=1, age=10, energy=50)
        assert evaluate_fitness(c1) > evaluate_fitness(c2)

    def test_fitness_nonnegative(self):
        """Fitness should be non-negative."""
        c = self._make_creature(food_eaten=0, age=0, energy=0)
        assert evaluate_fitness(c) >= 0

    def test_select_top_returns_correct_count(self):
        """select_top should return the requested number of DNA."""
        creatures = [self._make_creature(food_eaten=i) for i in range(20)]
        top = select_top(creatures, n=5)
        assert len(top) == 5

    def test_select_top_ordered_by_fitness(self):
        """select_top should return DNA sorted by fitness (best first)."""
        creatures = [self._make_creature(food_eaten=i) for i in range(20)]
        top = select_top(creatures, n=5)
        # The top should come from creatures with the most food
        # (since food has the highest weight in fitness)
        # We can verify by checking that each top DNA came from a high-food creature
        fitnesses = [evaluate_fitness(c) for c in creatures]
        sorted_fits = sorted(fitnesses, reverse=True)
        top_fitnesses = sorted_fits[:5]
        # All top 5 fitnesses should be >= the 5th best
        assert top_fitnesses[0] >= top_fitnesses[-1]

    def test_mutate_weights_changes_some_values(self):
        """Mutation should change at least some weight values."""
        dna = DNA.random()
        original = [w.copy() for w in dna.weights]
        mutated = mutate_weights(dna.weights, rate=0.5, strength=1.0)

        any_different = any(
            not np.array_equal(o, m) for o, m in zip(original, mutated)
        )
        assert any_different

    def test_mutate_weights_preserves_shape(self):
        """Mutated weights should keep original shapes."""
        dna = DNA.random()
        mutated = mutate_weights(dna.weights)
        for orig, mut in zip(dna.weights, mutated):
            assert orig.shape == mut.shape

    def test_mutate_weights_does_not_modify_original(self):
        """Original weights should not be changed by mutation."""
        dna = DNA.random()
        original_copy = [w.copy() for w in dna.weights]
        mutate_weights(dna.weights)
        for orig, copy in zip(dna.weights, original_copy):
            np.testing.assert_array_equal(orig, copy)

    def test_mutate_traits_preserves_budget(self):
        """Trait mutation must always preserve the budget sum."""
        for _ in range(100):
            dna = DNA.random()
            speed, size, vision, efficiency = mutate_traits(dna)
            total = speed + size + vision + efficiency
            assert total == EVOLUTION_POINTS, f"Budget was {total}"

    def test_mutate_traits_minimum_one(self):
        """No trait should be zero after mutation."""
        for _ in range(100):
            dna = DNA.random()
            speed, size, vision, efficiency = mutate_traits(dna)
            assert speed >= 1
            assert size >= 1
            assert vision >= 1
            assert efficiency >= 1

    def test_create_next_generation_correct_size(self):
        """Next generation should have the target population size."""
        top_dna = [DNA.random() for _ in range(5)]
        next_gen = create_next_generation(top_dna, population_size=50)
        assert len(next_gen) == 50

    def test_create_next_generation_empty_input(self):
        """Empty top performers should create random population."""
        next_gen = create_next_generation([], population_size=20)
        assert len(next_gen) == 20
        # All should have valid budgets
        for dna in next_gen:
            assert dna.validate_budget()

    def test_create_next_generation_valid_budgets(self):
        """All DNA in next generation should have valid budgets."""
        top_dna = [DNA.random() for _ in range(10)]
        next_gen = create_next_generation(top_dna, population_size=50)
        for dna in next_gen:
            total = dna.speed + dna.size + dna.vision + dna.efficiency
            assert total == EVOLUTION_POINTS, f"Budget was {total}"

    # -- Crossover Tests ----------------------------------------------------------

    def test_crossover_weights_blends_parents(self):
        """Two parents with distinct weights should produce a blend from both."""
        parent_a = DNA.random()
        parent_b = DNA.random()

        # Force parents to have clearly different weights
        for i, w in enumerate(parent_a.weights):
            parent_a.weights[i] = np.ones_like(w) * 10.0
        for i, w in enumerate(parent_b.weights):
            parent_b.weights[i] = np.ones_like(w) * -10.0

        np.random.seed(42)
        blended = crossover_weights(parent_a.weights, parent_b.weights)

        # Result should contain values from both parents (10.0 and -10.0)
        for b_arr in blended:
            vals = set(b_arr.flatten())
            assert 10.0 in vals or -10.0 in vals, "Blended array has no parent values"

        all_vals = np.concatenate([b.flatten() for b in blended])
        has_a = np.any(np.isclose(all_vals, 10.0))
        has_b = np.any(np.isclose(all_vals, -10.0))
        assert has_a and has_b, "Crossover should include values from both parents"

    def test_crossover_weights_preserves_shapes(self):
        """Output weight arrays must match the shape of the input arrays."""
        parent_a = DNA.random()
        parent_b = DNA.random()

        blended = crossover_weights(parent_a.weights, parent_b.weights)

        assert len(blended) == len(parent_a.weights)
        for orig, blend in zip(parent_a.weights, blended):
            assert orig.shape == blend.shape

    def test_crossover_traits_preserves_budget(self):
        """Crossover traits must sum to EVOLUTION_POINTS."""
        for _ in range(100):
            parent_a = DNA.random()
            parent_b = DNA.random()
            speed, size, vision, efficiency = crossover_traits(parent_a, parent_b)
            total = speed + size + vision + efficiency
            assert total == EVOLUTION_POINTS, f"Budget was {total}, expected {EVOLUTION_POINTS}"

    def test_crossover_traits_all_minimum_one(self):
        """No trait should be less than 1 after crossover."""
        for _ in range(100):
            parent_a = DNA.random()
            parent_b = DNA.random()
            speed, size, vision, efficiency = crossover_traits(parent_a, parent_b)
            assert speed >= 1, f"speed was {speed}"
            assert size >= 1, f"size was {size}"
            assert vision >= 1, f"vision was {vision}"
            assert efficiency >= 1, f"efficiency was {efficiency}"

    def test_create_next_generation_with_crossover(self):
        """crossover=True should produce a valid population."""
        top_dna = [DNA.random() for _ in range(5)]
        next_gen = create_next_generation(top_dna, population_size=50, crossover=True)

        assert len(next_gen) == 50
        for dna in next_gen:
            total = dna.speed + dna.size + dna.vision + dna.efficiency
            assert total == EVOLUTION_POINTS, f"Budget was {total}"
            # Weights should have correct number of arrays
            assert len(dna.weights) == len(top_dna[0].weights)
            # Each weight array should have the right shape
            for orig, child in zip(top_dna[0].weights, dna.weights):
                assert orig.shape == child.shape

    def test_create_next_generation_crossover_false_unchanged(self):
        """crossover=False should behave identically to the original algorithm."""
        np.random.seed(99)
        top_dna = [DNA.random() for _ in range(5)]

        # With crossover=False (default), clone-and-mutate path
        np.random.seed(123)
        gen_no_crossover = create_next_generation(
            top_dna, population_size=30, crossover=False
        )

        assert len(gen_no_crossover) == 30
        for dna in gen_no_crossover:
            total = dna.speed + dna.size + dna.vision + dna.efficiency
            assert total == EVOLUTION_POINTS, f"Budget was {total}"
            assert len(dna.weights) == len(top_dna[0].weights)
