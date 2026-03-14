"""Tests for the DNA data structure."""

import numpy as np
import pytest

from pangea.config import EVOLUTION_POINTS, NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE
from pangea.dna import DNA


class TestDNA:
    def _make_dna(self, speed=20, size=20, vision=20, efficiency=20, lifespan=20):
        """Helper to create a DNA with default weights."""
        weights = [
            np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
            np.zeros(NN_HIDDEN_SIZE),
            np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
            np.zeros(NN_OUTPUT_SIZE),
        ]
        return DNA(weights=weights, speed=speed, size=size, vision=vision,
                   efficiency=efficiency, lifespan=lifespan)

    def test_validate_budget_valid(self):
        """Budget should be valid when traits sum to EVOLUTION_POINTS."""
        dna = self._make_dna(20, 20, 20, 20, 20)
        assert dna.validate_budget()

    def test_validate_budget_invalid(self):
        """Budget should be invalid when traits don't sum correctly."""
        dna = self._make_dna(30, 30, 30, 30, 30)
        assert not dna.validate_budget()

    def test_effective_speed(self):
        """Effective speed should scale linearly with speed points."""
        dna = self._make_dna(speed=50)
        assert dna.effective_speed == pytest.approx(5.0)

    def test_effective_radius(self):
        """Effective radius should include base + scaled size."""
        dna = self._make_dna(size=20)
        assert dna.effective_radius == pytest.approx(3.0 + 20 * 0.15)

    def test_effective_vision(self):
        """Effective vision should include base + scaled vision."""
        dna = self._make_dna(vision=25)
        assert dna.effective_vision == pytest.approx(50.0 + 25 * 4.0)

    def test_effective_lifespan(self):
        """Effective lifespan should include base + scaled lifespan."""
        dna = self._make_dna(lifespan=20)
        assert dna.effective_lifespan == pytest.approx(10.0 + 20 * 0.5)

    def test_max_speed_size_penalty(self):
        """Larger creatures should have reduced max speed."""
        small = self._make_dna(speed=50, size=1)
        big = self._make_dna(speed=50, size=50)
        assert big.max_speed < small.max_speed

    def test_to_dict_from_dict_round_trip(self):
        """Serialization round trip should preserve all data."""
        original = self._make_dna(25, 20, 20, 20, 15)
        data = original.to_dict()
        restored = DNA.from_dict(data)

        assert restored.speed == original.speed
        assert restored.size == original.size
        assert restored.vision == original.vision
        assert restored.efficiency == original.efficiency
        assert restored.lifespan == original.lifespan

        for orig_w, rest_w in zip(original.weights, restored.weights):
            np.testing.assert_array_almost_equal(orig_w, rest_w)

    def test_to_dict_structure(self):
        """to_dict should produce expected JSON-compatible structure."""
        dna = self._make_dna()
        d = dna.to_dict()
        assert "weights" in d
        assert "W1" in d["weights"]
        assert "b1" in d["weights"]
        assert "W2" in d["weights"]
        assert "b2" in d["weights"]
        assert d["speed"] == 20
        assert d["lifespan"] == 20
        assert isinstance(d["weights"]["W1"], list)

    def test_random_creates_valid_budget(self):
        """Random DNA should always have a valid budget."""
        for _ in range(50):
            dna = DNA.random()
            total = dna.speed + dna.size + dna.vision + dna.efficiency + dna.lifespan
            assert total == EVOLUTION_POINTS, f"Budget was {total}, expected {EVOLUTION_POINTS}"

    def test_random_has_correct_weight_shapes(self):
        """Random DNA weights should have correct shapes."""
        dna = DNA.random()
        assert dna.weights[0].shape == (NN_INPUT_SIZE, NN_HIDDEN_SIZE)
        assert dna.weights[1].shape == (NN_HIDDEN_SIZE,)
        assert dna.weights[2].shape == (NN_HIDDEN_SIZE, NN_OUTPUT_SIZE)
        assert dna.weights[3].shape == (NN_OUTPUT_SIZE,)

    def test_from_dict_lifespan_fallback(self):
        """from_dict should use DEFAULT_LIFESPAN when lifespan key is missing."""
        from pangea.config import DEFAULT_LIFESPAN
        dna = self._make_dna()
        d = dna.to_dict()
        del d["lifespan"]
        restored = DNA.from_dict(d)
        assert restored.lifespan == DEFAULT_LIFESPAN
