"""Tests for the terrain/biomes system."""

import math

import numpy as np
import pytest

from pangea.config import BIOME_SPEED_MULTIPLIERS
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.world import Biome, World


def _make_dna(speed=25, size=25, vision=25, efficiency=25):
    """Helper to create a DNA with default weights."""
    weights = [
        np.random.randn(4, 8) * 0.5,
        np.zeros(8),
        np.random.randn(8, 2) * 0.5,
        np.zeros(2),
    ]
    return DNA(weights=weights, speed=speed, size=size, vision=vision, efficiency=efficiency)


def _make_creature(x=400.0, y=300.0, speed=25, size=25, vision=25, efficiency=25):
    """Helper to create a creature at a given position."""
    dna = _make_dna(speed=speed, size=size, vision=vision, efficiency=efficiency)
    return Creature(dna, x, y)


class TestBiomes:
    def test_biomes_generated_at_init(self):
        """World with biome_count=N should have exactly N biomes."""
        for n in (0, 1, 3, 5):
            settings = SimSettings(biome_count=n, initial_food_count=0)
            world = World([], width=800, height=600, settings=settings)
            assert len(world.biomes) == n

    def test_water_biome_slows_creature(self):
        """A creature inside a water biome should move less distance per update."""
        c_water = _make_creature(x=400.0, y=300.0)
        c_normal = _make_creature(x=400.0, y=300.0)

        # Give both the same heading and speed
        c_water.heading = 0.0
        c_normal.heading = 0.0
        c_water.speed = 2.0
        c_normal.speed = 2.0

        dt = 1 / 60
        water_mult = BIOME_SPEED_MULTIPLIERS["water"]  # 0.5
        c_water.update(dt, speed_multiplier=water_mult)
        c_normal.update(dt, speed_multiplier=1.0)

        dist_water = math.sqrt((c_water.x - 400.0) ** 2 + (c_water.y - 300.0) ** 2)
        dist_normal = math.sqrt((c_normal.x - 400.0) ** 2 + (c_normal.y - 300.0) ** 2)

        assert dist_water < dist_normal
        assert dist_water == pytest.approx(dist_normal * water_mult, rel=1e-6)

    def test_road_biome_speeds_creature(self):
        """A creature on a road biome should move more distance per update."""
        c_road = _make_creature(x=400.0, y=300.0)
        c_normal = _make_creature(x=400.0, y=300.0)

        c_road.heading = 0.0
        c_normal.heading = 0.0
        c_road.speed = 2.0
        c_normal.speed = 2.0

        dt = 1 / 60
        road_mult = BIOME_SPEED_MULTIPLIERS["road"]  # 1.5
        c_road.update(dt, speed_multiplier=road_mult)
        c_normal.update(dt, speed_multiplier=1.0)

        dist_road = math.sqrt((c_road.x - 400.0) ** 2 + (c_road.y - 300.0) ** 2)
        dist_normal = math.sqrt((c_normal.x - 400.0) ** 2 + (c_normal.y - 300.0) ** 2)

        assert dist_road > dist_normal
        assert dist_road == pytest.approx(dist_normal * road_mult, rel=1e-6)

    def test_no_biome_normal_speed(self):
        """Outside all biomes, get_speed_multiplier should return 1.0."""
        settings = SimSettings(biome_count=0, initial_food_count=0)
        world = World([], width=800, height=600, settings=settings)

        # No biomes at all — every position should be normal speed
        assert world.get_speed_multiplier(100, 100) == 1.0
        assert world.get_speed_multiplier(400, 300) == 1.0
        assert world.get_speed_multiplier(0, 0) == 1.0

    def test_speed_multiplier_default(self):
        """creature.update() without speed_multiplier should work as before (1.0)."""
        c_default = _make_creature(x=400.0, y=300.0)
        c_explicit = _make_creature(x=400.0, y=300.0)

        c_default.heading = 0.0
        c_explicit.heading = 0.0
        c_default.speed = 2.0
        c_explicit.speed = 2.0

        dt = 1 / 60
        c_default.update(dt)  # no speed_multiplier arg
        c_explicit.update(dt, speed_multiplier=1.0)

        assert c_default.x == pytest.approx(c_explicit.x, abs=1e-10)
        assert c_default.y == pytest.approx(c_explicit.y, abs=1e-10)
        assert c_default.energy == pytest.approx(c_explicit.energy, abs=1e-10)
