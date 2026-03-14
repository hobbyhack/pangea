"""Tests for the Creature class."""

import math

import numpy as np
import pytest

from pangea.config import (
    BASE_ENERGY,
    CARNIVORE_FOOD_PENALTY,
    DIET_CARNIVORE,
    DIET_HERBIVORE,
    DIET_SCAVENGER,
    HERBIVORE_FOOD_BONUS,
    NN_HIDDEN_SIZE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
)
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.world import Food


class TestCreature:
    def _make_creature(self, x=100.0, y=100.0, speed=20, size=20, vision=20,
                       efficiency=20, lifespan=20):
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

    def test_initial_state(self):
        """New creature should start alive with full energy."""
        c = self._make_creature()
        assert c.alive
        assert c.energy == BASE_ENERGY
        assert c.food_eaten == 0
        assert c.age == 0.0

    def test_sensor_output_shape(self):
        """Sense should return array of shape (10,)."""
        c = self._make_creature()
        food = [Food(x=150, y=100)]
        inputs = c.sense(food, 800, 600)
        assert inputs.shape == (10,)

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

    def test_sensor_creature_distance(self):
        """Creature distance sensor should detect nearby creatures."""
        c1 = self._make_creature(x=100, y=100)
        c2 = self._make_creature(x=150, y=100)
        inputs = c1.sense([], 800, 600, creatures=[c1, c2])
        # Creature distance should be < 1.0 (neighbor detected)
        assert inputs[4] < 1.0

    def test_sensor_own_speed(self):
        """Own speed sensor should reflect current speed."""
        c = self._make_creature()
        c.speed = 0.0
        inputs = c.sense([], 800, 600)
        assert inputs[6] == pytest.approx(0.0)

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

    def test_creature_dies_at_lifespan(self):
        """Creature should die when age exceeds effective lifespan."""
        c = self._make_creature(lifespan=20)
        # effective_lifespan = 10.0 + 20 * 0.5 = 20.0
        c.energy = 9999  # prevent energy death
        c.age = 19.9
        c.update(0.2)  # age becomes 20.1, exceeds 20.0
        assert not c.alive

    def test_eat_increases_energy_and_count(self):
        """Eating should increase energy and food_eaten counter."""
        from pangea.config import HERBIVORE_FOOD_BONUS
        c = self._make_creature()
        c.energy = 50.0
        c.eat(30.0)
        # Default diet is herbivore, gets bonus
        assert c.energy == 50.0 + 30.0 * HERBIVORE_FOOD_BONUS
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

    def test_herbivore_food_bonus(self):
        """Herbivores should get bonus energy from food."""
        c = self._make_creature()
        c.dna.diet = DIET_HERBIVORE
        c.energy = 50.0
        c.eat(20.0)
        assert c.energy == pytest.approx(50.0 + 20.0 * HERBIVORE_FOOD_BONUS)

    def test_carnivore_food_penalty(self):
        """Carnivores should get reduced energy from plant food."""
        c = self._make_creature()
        c.dna.diet = DIET_CARNIVORE
        c.energy = 50.0
        c.eat(20.0)
        assert c.energy == pytest.approx(50.0 + 20.0 * CARNIVORE_FOOD_PENALTY)

    def test_scavenger_normal_food(self):
        """Scavengers should eat food at normal rate."""
        c = self._make_creature()
        c.dna.diet = DIET_SCAVENGER
        c.energy = 50.0
        c.eat(20.0)
        assert c.energy == pytest.approx(70.0)

    def test_gain_energy(self):
        """gain_energy should add energy without food count."""
        c = self._make_creature()
        c.energy = 50.0
        c.gain_energy(15.0)
        assert c.energy == pytest.approx(65.0)
        assert c.food_eaten == 0

    def test_speed_multiplier_affects_movement(self):
        """speed_multiplier should scale movement distance."""
        c1 = self._make_creature(x=100, y=100)
        c2 = self._make_creature(x=100, y=100)
        c1.heading = 0.0
        c2.heading = 0.0
        c1.speed = 2.0
        c2.speed = 2.0

        c1.update(1 / 60, speed_multiplier=1.0)
        c2.update(1 / 60, speed_multiplier=0.5)

        dist1 = abs(c1.x - 100.0)
        dist2 = abs(c2.x - 100.0)
        assert dist2 < dist1
