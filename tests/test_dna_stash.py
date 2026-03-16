"""Tests for DNA stash on Species — serialization, init from stash, helpers."""

import numpy as np

from pangea.config import NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.species import Species, SpeciesSettings, SpeciesRegistry


def _make_dna(species_id: str = "alpha") -> DNA:
    weights = [
        np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
        np.zeros(NN_HIDDEN_SIZE),
        np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
        np.zeros(NN_OUTPUT_SIZE),
    ]
    return DNA(weights=weights, speed=25, size=15, vision=20,
               efficiency=25, lifespan=15, species_id=species_id)


def _make_species(species_id: str = "alpha", **kwargs) -> Species:
    return Species(
        id=species_id,
        name=species_id.title(),
        color=(100, 100, 100),
        settings=SpeciesSettings(**kwargs),
    )


class TestDnaStashSerialization:
    def test_species_without_stash_roundtrips(self):
        """Species without dna_stash serializes and deserializes correctly."""
        sp = _make_species("test")
        d = sp.to_dict()
        assert "dna_stash" not in d
        sp2 = Species.from_dict(d)
        assert sp2.dna_stash is None
        assert not sp2.has_dna_stash

    def test_species_with_stash_roundtrips(self):
        """Species with dna_stash includes it in to_dict and restores it."""
        sp = _make_species("test")
        dna = _make_dna("test")
        sp.dna_stash = [dna.to_dict()]

        d = sp.to_dict()
        assert "dna_stash" in d
        assert len(d["dna_stash"]) == 1

        sp2 = Species.from_dict(d)
        assert sp2.has_dna_stash
        assert len(sp2.dna_stash) == 1
        # Verify DNA can be reconstructed from stash
        restored = DNA.from_dict(sp2.dna_stash[0])
        assert restored.species_id == "test"
        assert restored.speed == 25
        np.testing.assert_array_almost_equal(restored.weights[0], dna.weights[0])

    def test_stash_none_vs_empty(self):
        """None means no stash, empty list means stash cleared."""
        sp = _make_species("x")
        assert not sp.has_dna_stash

        sp.dna_stash = []
        assert not sp.has_dna_stash  # empty list is falsy

        sp.dna_stash = [_make_dna("x").to_dict()]
        assert sp.has_dna_stash

    def test_copy_preserves_stash(self):
        """Species.copy() preserves the dna_stash."""
        sp = _make_species("test")
        sp.dna_stash = [_make_dna("test").to_dict()]
        sp2 = sp.copy()
        assert sp2.has_dna_stash
        assert len(sp2.dna_stash) == 1

    def test_registry_roundtrip_with_stash(self):
        """SpeciesRegistry serialization preserves dna_stash on species."""
        reg = SpeciesRegistry()
        sp = _make_species("alpha")
        sp.dna_stash = [_make_dna("alpha").to_dict(), _make_dna("alpha").to_dict()]
        reg.register(sp)
        reg.register(_make_species("beta"))  # no stash

        data = reg.to_list()
        reg2 = SpeciesRegistry.from_list(data)

        alpha = reg2.get("alpha")
        assert alpha.has_dna_stash
        assert len(alpha.dna_stash) == 2

        beta = reg2.get("beta")
        assert not beta.has_dna_stash


class TestAutoStashSetting:
    def test_default_off(self):
        """auto_stash_dna defaults to False."""
        ss = SpeciesSettings()
        assert ss.auto_stash_dna is False

    def test_roundtrip(self):
        """auto_stash_dna survives serialization."""
        ss = SpeciesSettings(auto_stash_dna=True)
        d = ss.to_dict()
        assert d["auto_stash_dna"] is True
        ss2 = SpeciesSettings.from_dict(d)
        assert ss2.auto_stash_dna is True


class TestStashHelpers:
    def test_stash_species_dna(self):
        """stash_species_dna selects top performers and stores on species."""
        from pangea.save_load import stash_species_dna

        sp = _make_species("alpha", top_performers_count=2)
        creatures = []
        for i in range(5):
            dna = _make_dna("alpha")
            c = Creature(dna, 100, 100, species=sp)
            c.food_eaten = i  # fitness correlates with food_eaten
            creatures.append(c)

        count = stash_species_dna(sp, creatures)
        assert count == 2
        assert sp.has_dna_stash
        assert len(sp.dna_stash) == 2

    def test_stash_filters_by_species(self):
        """stash_species_dna only considers creatures of the target species."""
        from pangea.save_load import stash_species_dna

        sp_a = _make_species("alpha")
        sp_b = _make_species("beta")
        creatures = [
            Creature(_make_dna("alpha"), 100, 100, species=sp_a),
            Creature(_make_dna("beta"), 200, 200, species=sp_b),
        ]
        creatures[0].food_eaten = 5
        creatures[1].food_eaten = 10

        count = stash_species_dna(sp_a, creatures)
        assert count == 1
        assert sp_a.has_dna_stash

        # Beta not stashed
        assert not sp_b.has_dna_stash

    def test_stash_empty_creatures(self):
        """stash_species_dna returns 0 when no matching creatures."""
        from pangea.save_load import stash_species_dna

        sp = _make_species("alpha")
        count = stash_species_dna(sp, [])
        assert count == 0
        assert not sp.has_dna_stash

    def test_clear_stash(self):
        """clear_species_dna_stash removes the stash."""
        from pangea.save_load import clear_species_dna_stash

        sp = _make_species("alpha")
        sp.dna_stash = [_make_dna("alpha").to_dict()]
        assert sp.has_dna_stash

        clear_species_dna_stash(sp)
        assert not sp.has_dna_stash
        assert sp.dna_stash is None
