"""Tests for food cluster spawning and seasonal food mechanics."""

import math

import pytest

from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.world import World


def _make_world(
    food_cluster_size: int = 4,
    season_length: float = 60.0,
    season_min_rate: float = 0.2,
    food_spawn_rate: float = 2.0,
    initial_food_count: int = 0,
) -> World:
    """Helper to create a World with custom food/season settings."""
    settings = SimSettings(
        food_cluster_size=food_cluster_size,
        season_length=season_length,
        season_min_rate=season_min_rate,
        food_spawn_rate=food_spawn_rate,
        initial_food_count=initial_food_count,
    )
    # Create a single creature so the world is valid
    dna = DNA.random()
    creature = Creature(dna, 400.0, 400.0)
    return World([creature], settings=settings)


class TestFoodClusters:
    def test_cluster_spawns_multiple_food(self):
        """When spawn triggers, cluster_size items appear near the same location."""
        world = _make_world(food_cluster_size=4, food_spawn_rate=100.0, initial_food_count=0)
        # Force a large dt so the accumulator exceeds 1.0
        world.spawn_food(1.0)
        # With rate=100 and dt=1.0, many clusters should have spawned
        # Each cluster spawns 4 items, so total should be a multiple of 4
        assert len(world.food) >= 4
        assert len(world.food) % 4 == 0

    def test_cluster_size_one_is_single(self):
        """cluster_size=1 matches original single-spawn behavior."""
        world = _make_world(food_cluster_size=1, food_spawn_rate=100.0, initial_food_count=0)
        world.spawn_food(1.0)
        assert len(world.food) >= 1

    def test_cluster_items_near_same_location(self):
        """Cluster food items should be within reasonable distance of each other."""
        world = _make_world(food_cluster_size=4, food_spawn_rate=1.0, initial_food_count=0)
        # Set accumulator high enough to trigger exactly one cluster
        world._food_spawn_accum = 1.0
        world.spawn_food(0.0)  # dt=0 so no additional accumulation
        assert len(world.food) == 4

        # All items should be within ~100px of each other (Gaussian with sigma=30)
        xs = [f.x for f in world.food]
        ys = [f.y for f in world.food]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        # Very unlikely to exceed 200px with sigma=30
        assert x_range < 200
        assert y_range < 200


class TestSeasonalFood:
    def test_seasonal_multiplier_oscillates(self):
        """Spawn rate should vary over a full seasonal cycle."""
        world = _make_world(season_length=60.0, season_min_rate=0.2)

        # Sample multiplier at different times
        multipliers = []
        for t in range(0, 61, 5):
            world.season_time = float(t)
            multipliers.append(world.seasonal_multiplier())

        # Should have variation (not all the same)
        assert max(multipliers) > min(multipliers)
        # Maximum should be close to 1.0
        assert max(multipliers) == pytest.approx(1.0, abs=0.01)

    def test_season_min_rate_respected(self):
        """At minimum phase, rate should be scaled down to season_min_rate."""
        world = _make_world(season_length=60.0, season_min_rate=0.2)

        # The minimum of sin occurs at 3/4 of the cycle (sin = -1)
        world.season_time = 60.0 * 3 / 4
        mult = world.seasonal_multiplier()
        assert mult == pytest.approx(0.2, abs=0.01)

    def test_season_max_rate(self):
        """At maximum phase, rate should be 1.0."""
        world = _make_world(season_length=60.0, season_min_rate=0.2)

        # The maximum of sin occurs at 1/4 of the cycle (sin = 1)
        world.season_time = 60.0 * 1 / 4
        mult = world.seasonal_multiplier()
        assert mult == pytest.approx(1.0, abs=0.01)

    def test_season_time_tracked(self):
        """world.season_time should increment with updates."""
        world = _make_world()
        assert world.season_time == 0.0

        world.update(0.5)
        assert world.season_time == pytest.approx(0.5)

        world.update(0.3)
        assert world.season_time == pytest.approx(0.8)

    def test_season_affects_food_count(self):
        """Seasonal variation should affect how much food spawns."""
        world_peak = _make_world(
            season_length=60.0, season_min_rate=0.2,
            food_spawn_rate=10.0, initial_food_count=0,
        )
        world_peak.season_time = 15.0  # peak: sin=1, multiplier=1.0
        world_peak.spawn_food(1.0)
        peak_count = len(world_peak.food)

        world_trough = _make_world(
            season_length=60.0, season_min_rate=0.2,
            food_spawn_rate=10.0, initial_food_count=0,
        )
        world_trough.season_time = 45.0  # trough: sin=-1, multiplier=0.2
        world_trough.spawn_food(1.0)
        trough_count = len(world_trough.food)

        assert peak_count > trough_count
