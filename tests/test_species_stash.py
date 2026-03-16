"""Tests for species stash/unstash when toggling enabled."""

import numpy as np

from pangea.config import NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.species import Species, SpeciesSettings, SpeciesRegistry
from pangea.world import World


def _make_species(species_id: str, enabled: bool = True) -> Species:
    return Species(
        id=species_id,
        name=species_id.title(),
        color=(100, 100, 100),
        settings=SpeciesSettings(),
        enabled=enabled,
    )


def _make_creature(x: float, y: float, species: Species) -> Creature:
    weights = [
        np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
        np.zeros(NN_HIDDEN_SIZE),
        np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
        np.zeros(NN_OUTPUT_SIZE),
    ]
    dna = DNA(weights=weights, speed=20, size=20, vision=20,
              efficiency=20, lifespan=20, species_id=species.id)
    return Creature(dna, x, y, species=species)


def _make_world(species_list, creatures):
    registry = SpeciesRegistry()
    for sp in species_list:
        registry.register(sp)
    settings = SimSettings(
        species_registry=registry,
        initial_food_count=0,
        hazard_count=0,
        biome_count=0,
    )
    return World(creatures, settings=settings)


class TestSpeciesStash:
    def test_disable_stashes_creatures(self):
        """Disabling a species removes its creatures from the world."""
        sp_a = _make_species("alpha")
        sp_b = _make_species("beta")
        creatures = [_make_creature(100, 100, sp_a), _make_creature(200, 200, sp_b)]
        world = _make_world([sp_a, sp_b], creatures)

        assert world.alive_count() == 2

        sp_a.enabled = False
        world._sync_species_enabled()

        assert world.alive_count() == 1
        assert world.alive_count_by_species("alpha") == 0
        assert world.alive_count_by_species("beta") == 1
        assert len(world._stashed_creatures["alpha"]) == 1

    def test_reenable_restores_creatures(self):
        """Re-enabling a species restores stashed creatures."""
        sp_a = _make_species("alpha")
        creatures = [_make_creature(100, 100, sp_a), _make_creature(200, 200, sp_a)]
        world = _make_world([sp_a], creatures)

        sp_a.enabled = False
        world._sync_species_enabled()
        assert world.alive_count() == 0

        sp_a.enabled = True
        world._sync_species_enabled()
        assert world.alive_count() == 2
        assert "alpha" not in world._stashed_creatures

    def test_disable_does_not_stash_dead(self):
        """Dead creatures stay in the main list, not stashed."""
        sp_a = _make_species("alpha")
        c1 = _make_creature(100, 100, sp_a)
        c2 = _make_creature(200, 200, sp_a)
        c2.alive = False
        world = _make_world([sp_a], [c1, c2])

        sp_a.enabled = False
        world._sync_species_enabled()

        # 1 alive creature stashed, 1 dead stays in world.creatures
        assert len(world._stashed_creatures["alpha"]) == 1
        assert any(not c.alive and c.dna.species_id == "alpha" for c in world.creatures)

    def test_stash_persists_across_updates(self):
        """Stashed creatures stay stashed across multiple update calls."""
        sp_a = _make_species("alpha")
        sp_b = _make_species("beta")
        creatures = [_make_creature(100, 100, sp_a), _make_creature(200, 200, sp_b)]
        world = _make_world([sp_a, sp_b], creatures)
        world.freeplay = True

        sp_a.enabled = False
        world.update(0.016)
        world.update(0.016)

        assert world.alive_count_by_species("alpha") == 0
        assert len(world._stashed_creatures["alpha"]) == 1
