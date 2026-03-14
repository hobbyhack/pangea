"""Tests for the NeuralNetwork brain."""

import numpy as np
import pytest

from pangea.brain import NeuralNetwork
from pangea.config import NN_HIDDEN_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE


class TestNeuralNetwork:
    def test_forward_output_shape(self):
        """Forward pass should return array of shape (NN_OUTPUT_SIZE,)."""
        nn = NeuralNetwork()
        inputs = np.array([0.5, 0.3, 0.8, 0.6, 0.5, 0.0, 0.3])
        output = nn.forward(inputs)
        assert output.shape == (NN_OUTPUT_SIZE,)

    def test_forward_output_range(self):
        """Outputs should be in [-1, 1] due to tanh activation."""
        nn = NeuralNetwork()
        for _ in range(100):
            inputs = np.random.randn(NN_INPUT_SIZE)
            output = nn.forward(inputs)
            assert np.all(output >= -1.0)
            assert np.all(output <= 1.0)

    def test_get_weights_returns_four_arrays(self):
        """get_weights should return [W1, b1, W2, b2]."""
        nn = NeuralNetwork()
        weights = nn.get_weights()
        assert len(weights) == 4
        assert weights[0].shape == (NN_INPUT_SIZE, NN_HIDDEN_SIZE)
        assert weights[1].shape == (NN_HIDDEN_SIZE,)
        assert weights[2].shape == (NN_HIDDEN_SIZE, NN_OUTPUT_SIZE)
        assert weights[3].shape == (NN_OUTPUT_SIZE,)

    def test_get_weights_returns_copies(self):
        """Modifying returned weights should not affect the network."""
        nn = NeuralNetwork()
        weights = nn.get_weights()
        original_W1 = nn.W1.copy()
        weights[0] *= 0  # Zero out the returned copy
        np.testing.assert_array_equal(nn.W1, original_W1)

    def test_set_weights_round_trip(self):
        """get_weights → set_weights should produce identical network."""
        nn1 = NeuralNetwork()
        weights = nn1.get_weights()

        nn2 = NeuralNetwork()
        nn2.set_weights(weights)

        np.testing.assert_array_equal(nn1.W1, nn2.W1)
        np.testing.assert_array_equal(nn1.b1, nn2.b1)
        np.testing.assert_array_equal(nn1.W2, nn2.W2)
        np.testing.assert_array_equal(nn1.b2, nn2.b2)

    def test_copy_independence(self):
        """Copied network should be independent of original."""
        nn1 = NeuralNetwork()
        nn2 = nn1.copy()

        # Modify original
        nn1.W1 *= 0

        # Copy should be unchanged
        assert not np.array_equal(nn1.W1, nn2.W1)

    def test_copy_produces_same_output(self):
        """Copied network should produce same output for same input."""
        nn1 = NeuralNetwork()
        nn2 = nn1.copy()

        inputs = np.array([0.5, -0.3, 0.8, 0.1, 0.5, 0.0, 0.3])
        np.testing.assert_array_equal(nn1.forward(inputs), nn2.forward(inputs))

    def test_deterministic_forward(self):
        """Same input should always produce same output."""
        nn = NeuralNetwork()
        inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.3])
        out1 = nn.forward(inputs)
        out2 = nn.forward(inputs)
        np.testing.assert_array_equal(out1, out2)
