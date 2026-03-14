"""Tests for the Predator class and predator-world integration."""

import numpy as np
import pytest

from pangea.config import PREDATOR_DAMAGE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.world import Predator, World


def _make_creature(x=100.0, y=100.0, speed=25, size=25, vision=25, efficiency=25):
    """Helper to create a creature with specified traits."""
    weights = [
        np.random.randn(4, 8) * 0.5,
        np.zeros(8),
        np.random.randn(8, 2) * 0.5,
        np.zeros(2),
    ]
    dna = DNA(weights=weights, speed=speed, size=size, vision=vision, efficiency=efficiency)
    return Creature(dna, x, y)


class TestPredators:
    def test_predators_created_at_init(self):
        """World should create the correct number of predators from settings."""
        settings = SimSettings(predator_count=3)
        creatures = [_make_creature()]
        world = World(creatures, settings=settings)
        assert len(world.predators) == 3

    def test_predator_chases_creature(self):
        """Predator near a creature should move toward it."""
        creature = _make_creature(x=200.0, y=200.0)
        creature.alive = True

        predator = Predator(x=100.0, y=200.0, speed=2.0, vision=150.0)
        old_x = predator.x

        predator.update([creature], dt=1 / 60, width=800, height=600)

        # Predator should have moved to the right (toward creature at x=200)
        assert predator.x > old_x

    def test_predator_damages_on_contact(self):
        """Overlapping predator should drain creature energy."""
        creature = _make_creature(x=100.0, y=100.0)
        initial_energy = creature.energy

        settings = SimSettings(predator_count=0)
        world = World([creature], settings=settings)

        # Manually add a predator right on top of the creature
        predator = Predator(x=100.0, y=100.0)
        world.predators.append(predator)

        dt = 1 / 60
        world._check_predator_collisions(dt)

        expected_drain = PREDATOR_DAMAGE * dt * 60
        assert creature.energy == pytest.approx(initial_energy - expected_drain)

    def test_predator_count_zero(self):
        """No predators should be created when setting is 0."""
        settings = SimSettings(predator_count=0)
        creatures = [_make_creature()]
        world = World(creatures, settings=settings)
        assert len(world.predators) == 0

    def test_predator_wanders_without_prey(self):
        """Predator should still move when no creatures are nearby."""
        predator = Predator(x=400.0, y=300.0, speed=2.0, vision=150.0)
        old_x = predator.x
        old_y = predator.y

        # Update with no creatures
        predator.update([], dt=1 / 60, width=800, height=600)

        # Predator should have moved (speed > 0)
        moved = (predator.x != old_x) or (predator.y != old_y)
        assert moved
