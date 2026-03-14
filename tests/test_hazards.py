"""Tests for hazard zones in the World."""

import numpy as np
import pytest

from pangea.config import BASE_ENERGY, HAZARD_DAMAGE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.world import Hazard, World


def _make_creature(x: float = 100.0, y: float = 100.0) -> Creature:
    """Helper to create a creature at a given position."""
    weights = [
        np.zeros((4, 8)),
        np.zeros(8),
        np.zeros((8, 2)),
        np.zeros(2),
    ]
    dna = DNA(weights=weights, speed=25, size=25, vision=25, efficiency=25)
    return Creature(dna, x, y)


class TestHazards:
    def test_hazards_generated_at_init(self):
        """World with hazard_count=3 should generate exactly 3 hazards."""
        settings = SimSettings(hazard_count=3, initial_food_count=0)
        creature = _make_creature()
        world = World([creature], settings=settings)
        assert len(world.hazards) == 3

    def test_hazard_damages_creature_at_center(self):
        """A creature sitting at a hazard's center should lose energy."""
        settings = SimSettings(hazard_count=0, initial_food_count=0)
        creature = _make_creature(x=400.0, y=400.0)
        world = World([creature], settings=settings)

        # Manually place a hazard at the creature's location
        world.hazards.append(Hazard(x=400.0, y=400.0, radius=50.0, damage_rate=2.0))

        energy_before = creature.energy
        dt = 1.0 / 60.0
        world._apply_hazard_effects(creature, dt)

        # At center, dist=0, intensity = (1 - 0/50) * 2.0 = 2.0
        # Energy loss = 2.0 * dt * 60 = 2.0
        assert creature.energy < energy_before
        assert creature.energy == pytest.approx(energy_before - 2.0)

    def test_hazard_no_damage_outside_radius(self):
        """A creature outside the hazard radius should take no damage."""
        settings = SimSettings(hazard_count=0, initial_food_count=0)
        creature = _make_creature(x=400.0, y=400.0)
        world = World([creature], settings=settings)

        # Place hazard far away from creature
        world.hazards.append(Hazard(x=100.0, y=100.0, radius=50.0, damage_rate=2.0))

        energy_before = creature.energy
        dt = 1.0 / 60.0
        world._apply_hazard_effects(creature, dt)

        assert creature.energy == energy_before

    def test_hazard_damage_proportional_to_proximity(self):
        """Closer creatures should take more damage than distant ones."""
        settings = SimSettings(hazard_count=0, initial_food_count=0)

        # Creature at center of hazard
        creature_center = _make_creature(x=400.0, y=400.0)
        # Creature at edge of hazard (radius=50, place at distance 25)
        creature_edge = _make_creature(x=425.0, y=400.0)

        world = World([creature_center, creature_edge], settings=settings)
        world.hazards.append(Hazard(x=400.0, y=400.0, radius=50.0, damage_rate=2.0))

        dt = 1.0 / 60.0
        world._apply_hazard_effects(creature_center, dt)
        world._apply_hazard_effects(creature_edge, dt)

        center_loss = BASE_ENERGY - creature_center.energy
        edge_loss = BASE_ENERGY - creature_edge.energy

        assert center_loss > edge_loss
        assert center_loss > 0
        assert edge_loss > 0

    def test_hazard_count_zero(self):
        """No hazards should be generated when hazard_count is 0."""
        settings = SimSettings(hazard_count=0, initial_food_count=0)
        creature = _make_creature()
        world = World([creature], settings=settings)
        assert len(world.hazards) == 0
