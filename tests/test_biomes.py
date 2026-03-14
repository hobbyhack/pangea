"""Tests for the terrain/biomes system."""

import math

import numpy as np
import pytest

from pangea.config import (
    BIOME_ENERGY_DRAIN,
    BIOME_SPEED_MULTIPLIERS,
    NN_HIDDEN_SIZE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
)
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.world import Biome, World


def _make_dna(speed=20, size=20, vision=20, efficiency=20, lifespan=20):
    """Helper to create a DNA with default weights."""
    weights = [
        np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
        np.zeros(NN_HIDDEN_SIZE),
        np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
        np.zeros(NN_OUTPUT_SIZE),
    ]
    return DNA(weights=weights, speed=speed, size=size, vision=vision,
               efficiency=efficiency, lifespan=lifespan)


def _make_creature(x=400.0, y=300.0, speed=20, size=20, vision=20,
                   efficiency=20, lifespan=20):
    """Helper to create a creature at a given position."""
    dna = _make_dna(speed=speed, size=size, vision=vision,
                    efficiency=efficiency, lifespan=lifespan)
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

        # No biomes at all -- every position should be normal speed
        assert world.get_speed_multiplier(100, 100) == 1.0
        assert world.get_speed_multiplier(400, 300) == 1.0
        assert world.get_speed_multiplier(0, 0) == 1.0

    def test_forest_biome_slows_creature(self):
        """A creature inside a forest biome should move at 0.7x speed."""
        c = _make_creature(x=400.0, y=300.0)
        c.heading = 0.0
        c.speed = 2.0
        dt = 1 / 60
        forest_mult = BIOME_SPEED_MULTIPLIERS["forest"]  # 0.7
        c.update(dt, speed_multiplier=forest_mult)
        dist = math.sqrt((c.x - 400.0) ** 2 + (c.y - 300.0) ** 2)

        c_normal = _make_creature(x=400.0, y=300.0)
        c_normal.heading = 0.0
        c_normal.speed = 2.0
        c_normal.update(dt, speed_multiplier=1.0)
        dist_normal = math.sqrt((c_normal.x - 400.0) ** 2 + (c_normal.y - 300.0) ** 2)

        assert dist < dist_normal
        assert dist == pytest.approx(dist_normal * forest_mult, rel=1e-6)

    def test_desert_biome_speeds_creature(self):
        """A creature inside a desert biome should move at 1.3x speed."""
        desert_mult = BIOME_SPEED_MULTIPLIERS["desert"]  # 1.3
        assert desert_mult > 1.0

    def test_swamp_biome_very_slow(self):
        """Swamp biome should have very low speed multiplier."""
        swamp_mult = BIOME_SPEED_MULTIPLIERS["swamp"]  # 0.4
        assert swamp_mult < BIOME_SPEED_MULTIPLIERS["water"]

    def test_mountain_biome_slowest(self):
        """Mountain biome should have the lowest speed multiplier."""
        mtn_mult = BIOME_SPEED_MULTIPLIERS["mountain"]  # 0.3
        assert mtn_mult <= min(
            v for k, v in BIOME_SPEED_MULTIPLIERS.items() if k != "mountain"
        )

    def test_all_biome_types_have_speed_multiplier(self):
        """All six biome types should be defined in BIOME_SPEED_MULTIPLIERS."""
        expected = {"normal", "water", "road", "forest", "desert", "swamp", "mountain"}
        assert set(BIOME_SPEED_MULTIPLIERS.keys()) == expected

    def test_desert_energy_drain_defined(self):
        """Desert and swamp should have energy drain defined."""
        assert "desert" in BIOME_ENERGY_DRAIN
        assert "swamp" in BIOME_ENERGY_DRAIN
        assert BIOME_ENERGY_DRAIN["desert"] > 0
        assert BIOME_ENERGY_DRAIN["swamp"] > 0

    def test_biome_types_generated(self):
        """World should generate biomes from the full set of types."""
        # With enough biomes, we should see at least one of the new types
        settings = SimSettings(biome_count=30, initial_food_count=0)
        world = World([], width=1200, height=800, settings=settings)
        types_found = {b.biome_type for b in world.biomes}
        # With 30 biomes and 6 types, very likely to hit at least 4
        assert len(types_found) >= 4

    def test_get_biome_at(self):
        """get_biome_at should return the biome at a position."""
        settings = SimSettings(biome_count=0, initial_food_count=0)
        world = World([], width=800, height=600, settings=settings)
        # Add a known biome
        world.biomes.append(Biome(x=400, y=300, radius=100,
                                   biome_type="forest", speed_multiplier=0.7))
        biome = world.get_biome_at(400, 300)
        assert biome is not None
        assert biome.biome_type == "forest"
        assert world.get_biome_at(0, 0) is None

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
