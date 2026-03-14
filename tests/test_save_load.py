"""Tests for save/load functionality."""

import json
import os
import tempfile

import numpy as np
import pytest

from pangea.dna import DNA
from pangea.save_load import load_species, save_species


class TestSaveLoad:
    def test_save_creates_file(self):
        """save_species should create a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_species.json")
            dna_list = [DNA.random() for _ in range(5)]
            save_species(dna_list, filepath, species_name="test", generation=10)
            assert os.path.exists(filepath)

    def test_save_valid_json(self):
        """Saved file should contain valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            save_species([DNA.random()], filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert "creatures" in data
            assert "species_name" in data
            assert "generation" in data

    def test_round_trip_preserves_data(self):
        """Save -> load should preserve all DNA data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "roundtrip.json")
            original = [DNA.random() for _ in range(3)]
            save_species(original, filepath, species_name="test", generation=42)

            loaded, metadata = load_species(filepath)

            assert len(loaded) == len(original)
            assert metadata["species_name"] == "test"
            assert metadata["generation"] == 42

            for orig, rest in zip(original, loaded):
                assert orig.speed == rest.speed
                assert orig.size == rest.size
                assert orig.vision == rest.vision
                assert orig.efficiency == rest.efficiency
                for ow, rw in zip(orig.weights, rest.weights):
                    np.testing.assert_array_almost_equal(ow, rw)

    def test_round_trip_preserves_budget(self):
        """Loaded DNA should still have valid budgets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "budget.json")
            original = [DNA.random() for _ in range(10)]
            save_species(original, filepath)

            loaded, _ = load_species(filepath)
            for dna in loaded:
                assert dna.validate_budget()

    def test_save_creates_directory(self):
        """save_species should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "nested", "test.json")
            save_species([DNA.random()], filepath)
            assert os.path.exists(filepath)
