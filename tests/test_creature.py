"""Tests for the Creature class."""

import math

import numpy as np
import pytest

from pangea.config import BASE_ENERGY
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.world import Food


class TestCreature:
    def _make_creature(self, x=100.0, y=100.0, speed=25, size=25, vision=25, efficiency=25):
        """Helper to create a creature with specified traits."""
        weights = [
            np.random.randn(4, 8) * 0.5,
            np.zeros(8),
            np.random.randn(8, 2) * 0.5,
            np.zeros(2),
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
        """Sense should return array of shape (4,)."""
        c = self._make_creature()
        food = [Food(x=150, y=100)]
        inputs = c.sense(food, 800, 600)
        assert inputs.shape == (4,)
