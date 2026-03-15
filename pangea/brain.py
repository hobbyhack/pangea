"""
Neural Network "Brain" for creatures.
============================================================
A simple 3-layer feedforward neural network using numpy.
No training / backpropagation — weights are set by evolution.

Architecture:
    Input (12) → Hidden (8, tanh) → Output (2, tanh)

Inputs:  [food_distance, food_angle, wall_distance, energy_level,
          nearest_creature_distance, nearest_creature_angle, own_speed,
          threat_distance, threat_angle, under_attack,
          biome_speed, biome_danger]
Outputs: [turn_angle, thrust]
"""

import numpy as np

from pangea.config import NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE


class NeuralNetwork:
    """A small feedforward neural network used as a creature's brain."""

    def __init__(self) -> None:
        # Xavier-like initialization scaled to 0.5 for moderate initial outputs
        self.W1 = np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5
        self.b1 = np.zeros(NN_HIDDEN_SIZE)
        self.W2 = np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5
        self.b2 = np.zeros(NN_OUTPUT_SIZE)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run a forward pass through the network.

        Args:
            inputs: Array of shape (12,) with normalized sensor values.

        Returns:
            Array of shape (2,) with output values in [-1, 1].
            output[0] → turn angle (will be scaled to [-pi, pi])
            output[1] → thrust    (will be rescaled to [0, 1])
        """
        hidden = np.tanh(inputs @ self.W1 + self.b1)
        output = np.tanh(hidden @ self.W2 + self.b2)
        return output

    def get_weights(self) -> list[np.ndarray]:
        """Return a list of all weight/bias arrays (copies)."""
        return [w.copy() for w in [self.W1, self.b1, self.W2, self.b2]]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """
        Set all weight/bias arrays from a list.

        Args:
            weights: List of 4 arrays [W1, b1, W2, b2].
        """
        self.W1 = weights[0].copy()
        self.b1 = weights[1].copy()
        self.W2 = weights[2].copy()
        self.b2 = weights[3].copy()

    def copy(self) -> "NeuralNetwork":
        """Create an independent copy of this network."""
        nn = NeuralNetwork.__new__(NeuralNetwork)
        nn.W1 = self.W1.copy()
        nn.b1 = self.b1.copy()
        nn.W2 = self.W2.copy()
        nn.b2 = self.b2.copy()
        return nn
