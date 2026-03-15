"""Tests for threat detection — creatures sensing hostile attackers."""

import numpy as np
import pytest

from pangea.config import NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.species import Species, SpeciesSettings, SpeciesRegistry
from pangea.world import World


def _make_species(species_id: str, can_attack_other: bool = False,
                  can_attack_own: bool = False) -> Species:
    """Create a species with specified attack flags."""
    return Species(
        id=species_id,
        name=species_id.title(),
        color=(100, 100, 100),
        settings=SpeciesSettings(),
        can_eat_plants=True,
        plant_food_multiplier=1.0,
        can_attack_other_species=can_attack_other,
        can_attack_own_species=can_attack_own,
    )


def _make_creature(x=100.0, y=100.0, species: Species | None = None,
                   species_id: str = "herbivore"):
    """Helper to create a creature with specified traits."""
    weights = [
        np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
        np.zeros(NN_HIDDEN_SIZE),
        np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
        np.zeros(NN_OUTPUT_SIZE),
    ]
    dna = DNA(weights=weights, speed=20, size=20, vision=20,
              efficiency=20, lifespan=20, species_id=species_id)
    return Creature(dna, x, y, species=species)


class TestThreatSensing:
    def test_no_threats_defaults(self):
        """Without hostile creatures, threat sensors should be at defaults."""
        herb_sp = _make_species("herbivore")
        creature = _make_creature(species=herb_sp, species_id="herbivore")
        inputs = creature.sense([], 800, 600, creatures=[])
        assert inputs[7] == 1.0  # no threat → max distance
        assert inputs[8] == 0.0  # no threat → no angle

    def test_non_attacker_not_detected_as_threat(self):
        """A nearby herbivore should NOT register as a threat."""
        herb_sp = _make_species("herbivore")
        c1 = _make_creature(x=100, y=100, species=herb_sp, species_id="herbivore")
        c2 = _make_creature(x=150, y=100, species=herb_sp, species_id="herbivore")
        inputs = c1.sense([], 800, 600, creatures=[c1, c2])
        assert inputs[7] == 1.0  # herbivore is not a threat

    def test_attacker_detected_as_threat(self):
        """A nearby carnivore should register as a threat to a herbivore."""
        herb_sp = _make_species("herbivore")
        carn_sp = _make_species("carnivore", can_attack_other=True)
        victim = _make_creature(x=100, y=100, species=herb_sp, species_id="herbivore")
        attacker = _make_creature(x=150, y=100, species=carn_sp, species_id="carnivore")
        inputs = victim.sense([], 800, 600, creatures=[victim, attacker])
        assert inputs[7] < 1.0  # carnivore IS a threat
        assert inputs.shape == (12,)

    def test_same_species_attacker_detected(self):
        """A species that can_attack_own should be detected as threat by same species."""
        sp = _make_species("cannibal", can_attack_own=True)
        c1 = _make_creature(x=100, y=100, species=sp, species_id="cannibal")
        c2 = _make_creature(x=150, y=100, species=sp, species_id="cannibal")
        inputs = c1.sense([], 800, 600, creatures=[c1, c2])
        assert inputs[7] < 1.0  # same-species attacker IS a threat

    def test_same_species_not_threat_when_only_attacks_other(self):
        """A carnivore that only attacks OTHER species shouldn't threaten its own."""
        carn_sp = _make_species("carnivore", can_attack_other=True)
        c1 = _make_creature(x=100, y=100, species=carn_sp, species_id="carnivore")
        c2 = _make_creature(x=150, y=100, species=carn_sp, species_id="carnivore")
        inputs = c1.sense([], 800, 600, creatures=[c1, c2])
        assert inputs[7] == 1.0  # same-species carnivore is NOT a threat

    def test_dead_attacker_not_detected(self):
        """A dead attacker should not register as a threat."""
        herb_sp = _make_species("herbivore")
        carn_sp = _make_species("carnivore", can_attack_other=True)
        victim = _make_creature(x=100, y=100, species=herb_sp, species_id="herbivore")
        attacker = _make_creature(x=150, y=100, species=carn_sp, species_id="carnivore")
        attacker.alive = False
        inputs = victim.sense([], 800, 600, creatures=[victim, attacker])
        assert inputs[7] == 1.0  # dead creature is not a threat

    def test_under_attack_from_creature_combat(self):
        """Creature combat should set under_attack flag (tested via world)."""
        carn_sp = _make_species("carnivore", can_attack_other=True)
        herb_sp = _make_species("herbivore")

        registry = SpeciesRegistry()
        registry.register(carn_sp)
        registry.register(herb_sp)
        settings = SimSettings(species_registry=registry)

        attacker = _make_creature(x=100, y=100, species=carn_sp, species_id="carnivore")
        victim = _make_creature(x=100, y=100, species=herb_sp, species_id="herbivore")

        world = World([attacker, victim], settings=settings)
        dt = 1 / 60
        world._check_creature_attacks(dt)

        assert victim.under_attack == 1.0

    def test_no_predators_in_world(self):
        """World should not have a predators attribute."""
        settings = SimSettings()
        creatures = [_make_creature()]
        world = World(creatures, settings=settings)
        assert not hasattr(world, "predators")
