"""Tests for the Creature class."""

import math

import numpy as np
import pytest

from pangea.config import BASE_ENERGY, NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.world import Food


class TestCreature:
    def _make_creature(self, x=100.0, y=100.0, speed=25, size=25, vision=25, efficiency=25):
        """Helper to create a creature with specified traits."""
        weights = [
            np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
            np.zeros(NN_HIDDEN_SIZE),
            np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
            np.zeros(NN_OUTPUT_SIZE),
        ]
        dna = DNA(weights=weights, speed=speed, size=size, vision=vision, efficiency=efficiency)
        return Creature(dna, x, y)

    def test_initial_state(self):
        """New creature should start alive with full energy."""
        c = self._make_creature()
        assert c.alive
        assert c.energy == BASE_ENERGY
        assert c.food_eaten == 0
        assert c.age == 0.0

    def test_sensor_output_shape(self):
        """Sense should return array of shape (7,)."""
        c = self._make_creature()
        food = [Food(x=150, y=100)]
        inputs = c.sense(food, 800, 600)
        assert inputs.shape == (7,)

    def test_sensor_food_distance_normalized(self):
        """Food distance should be normalized between 0 and 1."""
        c = self._make_creature(x=100, y=100, vision=50)
        food = [Food(x=150, y=100)]
        inputs = c.sense(food, 800, 600)
        # Food is 50 px away, vision is 50 + 50*4 = 250
        assert 0.0 <= inputs[0] <= 1.0

    def test_sensor_no_food_visible(self):
        """When no food is in range, food distance should be 1.0 and angle 0.0."""
        c = self._make_creature(x=100, y=100, vision=1)
        # Vision = 50 + 1*4 = 54 px, food is far away
        food = [Food(x=500, y=500)]
        inputs = c.sense(food, 800, 600)
        assert inputs[0] == 1.0
        assert inputs[1] == 0.0

    def test_sensor_wall_distance_bounded(self):
        """Wall distance should be between 0 and 1."""
        c = self._make_creature(x=5, y=5)
        inputs = c.sense([], 800, 600)
        assert 0.0 <= inputs[2] <= 1.0

    def test_sensor_energy_normalized(self):
        """Energy sensor should be in [0, 1]."""
        c = self._make_creature()
        inputs = c.sense([], 800, 600)
        assert inputs[3] == pytest.approx(1.0)

        c.energy = BASE_ENERGY / 2
        inputs = c.sense([], 800, 600)
        assert inputs[3] == pytest.approx(0.5)

    def test_energy_drains_on_update(self):
        """Energy should decrease after update."""
        c = self._make_creature()
        c.speed = 1.0
        initial_energy = c.energy
        c.update(1 / 60)
        assert c.energy < initial_energy

    def test_creature_dies_at_zero_energy(self):
        """Creature should die when energy reaches zero."""
        c = self._make_creature()
        c.energy = 0.01
        c.speed = 10.0
        c.update(1.0)
        assert not c.alive
        assert c.energy == 0

    def test_eat_increases_energy_and_count(self):
        """Eating should increase energy and food_eaten counter."""
        c = self._make_creature()
        c.energy = 50.0
        c.eat(30.0)
        assert c.energy == 80.0
        assert c.food_eaten == 1

    def test_dead_creature_does_not_update(self):
        """Dead creatures should not move or lose energy."""
        c = self._make_creature()
        c.alive = False
        c.energy = 50.0
        pos_x, pos_y = c.x, c.y
        c.update(1 / 60)
        assert c.x == pos_x
        assert c.y == pos_y
        assert c.energy == 50.0

    def test_lineage_attribute(self):
        """Creature should store lineage correctly."""
        dna = DNA.random()
        c = Creature(dna, 100, 100, lineage="A")
        assert c.lineage == "A"

    # ── New sensor tests ─────────────────────────────────────

    def test_sensor_nearest_creature_distance(self):
        """Nearest creature distance should reflect actual distance."""
        c1 = self._make_creature(x=100, y=100, vision=50)
        c2 = self._make_creature(x=130, y=100, vision=25)
        # Distance is 30 px, vision is 50 + 50*4 = 250
        inputs = c1.sense([], 800, 600, creatures=[c1, c2])
        expected_dist = 30.0 / c1.dna.effective_vision
        assert inputs[4] == pytest.approx(expected_dist, abs=0.01)

    def test_sensor_nearest_creature_angle(self):
        """Nearest creature angle should reflect relative angle."""
        c1 = self._make_creature(x=100, y=100, vision=50)
        c1.heading = 0.0  # facing right
        c2 = self._make_creature(x=100, y=130, vision=25)
        # c2 is directly below (positive y), angle = pi/2 relative to heading 0
        inputs = c1.sense([], 800, 600, creatures=[c1, c2])
        # Angle to c2 is atan2(30, 0) = pi/2, minus heading 0 = pi/2
        # Normalized by pi = 0.5
        assert inputs[5] == pytest.approx(0.5, abs=0.01)

    def test_sensor_own_speed(self):
        """Own speed sensor should reflect current speed normalized by max_speed."""
        c = self._make_creature(x=100, y=100, speed=25)
        max_spd = c.dna.max_speed
        c.speed = max_spd * 0.5
        inputs = c.sense([], 800, 600)
        assert inputs[6] == pytest.approx(0.5, abs=0.01)

    def test_sensor_no_creatures_defaults(self):
        """When creatures=None, new sensors should return defaults (1.0, 0.0, 0.0)."""
        c = self._make_creature(x=100, y=100)
        c.speed = 0.0
        inputs = c.sense([], 800, 600, creatures=None)
        assert inputs[4] == pytest.approx(1.0)   # creature dist default
        assert inputs[5] == pytest.approx(0.0)   # creature angle default
        assert inputs[6] == pytest.approx(0.0)   # own speed (0 speed)
