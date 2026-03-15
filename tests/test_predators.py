"""Tests for the Predator class and predator-world integration."""

import numpy as np
import pytest

from pangea.config import NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE, PREDATOR_DAMAGE, SIZE_ARMOR_SCALE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.world import Predator, World


def _make_creature(x=100.0, y=100.0, speed=20, size=20, vision=20, efficiency=20, lifespan=20):
    """Helper to create a creature with specified traits."""
    weights = [
        np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
        np.zeros(NN_HIDDEN_SIZE),
        np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
        np.zeros(NN_OUTPUT_SIZE),
    ]
    dna = DNA(weights=weights, speed=speed, size=size, vision=vision,
              efficiency=efficiency, lifespan=lifespan)
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

        armor = creature.dna.effective_radius * SIZE_ARMOR_SCALE
        expected_drain = PREDATOR_DAMAGE * max(0.1, 1.0 - armor) * dt * 60
        assert creature.energy == pytest.approx(initial_energy - expected_drain)

    def test_predator_count_zero(self):
        """No predators should be created when setting is 0."""
        settings = SimSettings(predator_count=0)
        creatures = [_make_creature()]
        world = World(creatures, settings=settings)
        assert len(world.predators) == 0

    def test_creature_senses_predator(self):
        """Creature should detect a nearby predator via sensors."""
        creature = _make_creature(x=100.0, y=100.0)
        predator = Predator(x=150.0, y=100.0)
        inputs = creature.sense([], 800, 600, predators=[predator])
        # Predator distance sensor (index 7) should be < 1.0
        assert inputs[7] < 1.0
        assert inputs.shape == (12,)

    def test_no_predator_sensor_defaults(self):
        """Without predators, predator sensors should be at defaults."""
        creature = _make_creature(x=100.0, y=100.0)
        inputs = creature.sense([], 800, 600, predators=[])
        assert inputs[7] == 1.0  # no predator → max distance
        assert inputs[8] == 0.0  # no predator → no angle

    def test_size_armor_reduces_damage(self):
        """Larger creatures should take less predator damage."""
        # Small creature (size=5) vs large creature (size=60)
        small = _make_creature(x=100.0, y=100.0, size=5, speed=35, vision=20,
                               efficiency=20, lifespan=20)
        large = _make_creature(x=100.0, y=100.0, size=60, speed=5, vision=15,
                               efficiency=10, lifespan=10)

        settings = SimSettings(predator_count=0)
        world_s = World([small], settings=settings)
        world_l = World([large], settings=settings)

        pred_s = Predator(x=100.0, y=100.0)
        pred_l = Predator(x=100.0, y=100.0)
        world_s.predators.append(pred_s)
        world_l.predators.append(pred_l)

        dt = 1 / 60
        energy_before = small.energy
        world_s._check_predator_collisions(dt)
        small_drain = energy_before - small.energy

        energy_before = large.energy
        world_l._check_predator_collisions(dt)
        large_drain = energy_before - large.energy

        # Large creature should take less damage
        assert large_drain < small_drain

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
