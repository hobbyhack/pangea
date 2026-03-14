"""Tests for the day/night cycle feature."""

import math

import numpy as np
import pytest

from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.world import Food, World


def _make_dna(speed=25, size=25, vision=25, efficiency=25):
    """Create a DNA with specified traits."""
    weights = [
        np.random.randn(4, 8) * 0.5,
        np.zeros(8),
        np.random.randn(8, 2) * 0.5,
        np.zeros(2),
    ]
    return DNA(weights=weights, speed=speed, size=size, vision=vision, efficiency=efficiency)


def _make_creature(x=100.0, y=100.0, **kwargs):
    """Create a creature at the given position."""
    return Creature(_make_dna(**kwargs), x, y)


class TestDaylightFactor:
    def test_daylight_factor_range(self):
        """Daylight factor should stay within [0, 1] across a full cycle."""
        settings = SimSettings(day_night_cycle_length=60.0)
        creatures = [_make_creature()]
        world = World(creatures, settings=settings)

        # Sample many points across two full cycles
        for _ in range(1000):
            factor = world.daylight_factor
            assert 0.0 <= factor <= 1.0, f"daylight_factor {factor} out of range"
            world.day_night_time += 0.12  # advance time

    def test_daylight_factor_oscillates(self):
        """Daylight factor should change over time (not remain constant)."""
        settings = SimSettings(day_night_cycle_length=60.0)
        creatures = [_make_creature()]
        world = World(creatures, settings=settings)

        values = set()
        for i in range(100):
            world.day_night_time = i * 0.6  # sample across a full cycle
            values.add(round(world.daylight_factor, 4))

        # Should have many distinct values, not just one
        assert len(values) > 5


class TestVisionMultiplier:
    def test_night_reduces_vision(self):
        """sense() with vision_multiplier < 1 should not detect distant food."""
        # Create creature with known vision: 50 + 25*4 = 150 px
        creature = _make_creature(x=100.0, y=100.0, vision=25)
        effective_vision = creature.dna.effective_vision
        assert effective_vision == pytest.approx(150.0)

        # Place food at 120 px away (within full vision, outside 30% vision)
        food = [Food(x=220.0, y=100.0)]

        # With full vision (multiplier=1.0), food should be detected
        inputs_full = creature.sense(food, 800, 600, vision_multiplier=1.0)
        assert inputs_full[0] < 1.0, "Food should be detected with full vision"

        # With reduced vision (multiplier=0.3), effective = 45 px, food at 120 is out of range
        inputs_night = creature.sense(food, 800, 600, vision_multiplier=0.3)
        assert inputs_night[0] == 1.0, "Food should NOT be detected with reduced vision"
        assert inputs_night[1] == 0.0, "Food angle should be 0 when not detected"

    def test_vision_multiplier_default(self):
        """sense() without vision_multiplier should behave identically to multiplier=1.0."""
        creature = _make_creature(x=100.0, y=100.0)
        food = [Food(x=200.0, y=100.0)]

        inputs_default = creature.sense(food, 800, 600)
        inputs_explicit = creature.sense(food, 800, 600, vision_multiplier=1.0)

        np.testing.assert_array_equal(inputs_default, inputs_explicit)


class TestWorldDayNight:
    def test_day_night_time_increments(self):
        """world.day_night_time should increase with each update() call."""
        creatures = [_make_creature()]
        world = World(creatures)

        assert world.day_night_time == 0.0
        world.update(0.1)
        assert world.day_night_time == pytest.approx(0.1)
        world.update(0.2)
        assert world.day_night_time == pytest.approx(0.3)
