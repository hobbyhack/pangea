"""Tests for food decay (aging and expiration)."""

import pytest

from pangea.settings import SimSettings
from pangea.world import Food, World


class TestFoodDecay:
    def _make_world(self, food_decay_time: float = 15.0) -> World:
        """Create a minimal world with no creatures and custom decay time."""
        settings = SimSettings(
            initial_food_count=0,
            food_spawn_rate=0.0,
            food_decay_time=food_decay_time,
        )
        return World(creatures=[], width=400, height=300, settings=settings)

    def test_food_has_age_and_lifetime(self):
        """Food starts with age=0 and has a lifetime set from settings."""
        world = self._make_world(food_decay_time=10.0)
        world.add_food_at(100, 100)

        food = world.food[0]
        assert food.age == 0.0
        assert food.lifetime == 10.0

    def test_food_ages_during_update(self):
        """After world.update(), food age increases by dt."""
        world = self._make_world(food_decay_time=20.0)
        world.add_food_at(100, 100)

        world.update(0.5)
        assert world.food[0].age == pytest.approx(0.5)

        world.update(0.5)
        assert world.food[0].age == pytest.approx(1.0)

    def test_expired_food_removed(self):
        """Food past its lifetime is removed from the world."""
        world = self._make_world(food_decay_time=2.0)
        world.add_food_at(100, 100)
        assert len(world.food) == 1

        # Age food to just before expiration
        world.update(1.9)
        assert len(world.food) == 1

        # Push past the lifetime
        world.update(0.2)
        assert len(world.food) == 0

    def test_food_decay_setting_applies(self):
        """Changing settings.food_decay_time affects new food lifetime."""
        world = self._make_world(food_decay_time=30.0)
        world.add_food_at(100, 100)
        assert world.food[0].lifetime == 30.0

        # Change the setting and add new food
        world.settings.food_decay_time = 5.0
        world.add_food_at(200, 200)
        assert world.food[1].lifetime == 5.0

        # Original food keeps its original lifetime
        assert world.food[0].lifetime == 30.0
