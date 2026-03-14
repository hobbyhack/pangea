"""
DNA data structure for creatures.
============================================================
Bundles neural network weights with physical trait allocations.
Enforces the "Genetic Budget" — traits must sum to EVOLUTION_POINTS.

Traits and what they control:
    speed      -> How fast the creature can move
    size       -> How large the creature is (bigger = slower)
    vision     -> How far the creature can detect food
    efficiency -> How efficiently the creature uses energy
    lifespan   -> How long the creature can live (max age)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pangea.config import (
    DEFAULT_LIFESPAN,
    DIET_HERBIVORE,
    EFFICIENCY_BASE,
    EFFICIENCY_SCALE,
    EVOLUTION_POINTS,
    LIFESPAN_BASE,
    LIFESPAN_SCALE,
    NN_HIDDEN_SIZE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    RADIUS_BASE,
    RADIUS_SCALE,
    SIZE_SPEED_PENALTY,
    SPEED_SCALE,
    VISION_BASE,
    VISION_SCALE,
)


@dataclass
class DNA:
    """
    A creature's complete genetic code.

    Attributes:
        weights: List of numpy arrays [W1, b1, W2, b2] for the neural network.
        speed:   Evolution points allocated to speed (higher = faster).
        size:    Evolution points allocated to size (higher = bigger, but slower).
        vision:  Evolution points allocated to vision range (higher = sees farther).
        efficiency: Evolution points allocated to energy efficiency (higher = less drain).
        lifespan: Evolution points allocated to lifespan (higher = lives longer).
    """

    weights: list[np.ndarray]
    speed: int
    size: int
    vision: int
    efficiency: int
    lifespan: int
    diet: int = DIET_HERBIVORE  # 0=herbivore, 1=carnivore, 2=scavenger

    # ── Trait Scaling ────────────────────────────────────────

    @property
    def effective_speed(self) -> float:
        """Maximum velocity in pixels/frame (before size penalty)."""
        return self.speed * SPEED_SCALE

    @property
    def effective_radius(self) -> float:
        """Creature body radius in pixels."""
        return RADIUS_BASE + self.size * RADIUS_SCALE

    @property
    def effective_vision(self) -> float:
        """Vision range in pixels."""
        return VISION_BASE + self.vision * VISION_SCALE

    @property
    def effective_efficiency(self) -> float:
        """Energy efficiency multiplier (higher = less energy drain)."""
        return EFFICIENCY_BASE + self.efficiency * EFFICIENCY_SCALE

    @property
    def effective_lifespan(self) -> float:
        """Maximum age in seconds before the creature dies of old age."""
        return LIFESPAN_BASE + self.lifespan * LIFESPAN_SCALE

    @property
    def max_speed(self) -> float:
        """Actual max speed after size penalty."""
        return max(0.5, self.effective_speed - self.effective_radius * SIZE_SPEED_PENALTY)

    # ── Budget Validation ────────────────────────────────────

    def validate_budget(self) -> bool:
        """Check that trait points sum to EVOLUTION_POINTS."""
        return (
            self.speed + self.size + self.vision + self.efficiency + self.lifespan
            == EVOLUTION_POINTS
        )

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert DNA to a JSON-serializable dictionary."""
        return {
            "weights": {
                "W1": self.weights[0].tolist(),
                "b1": self.weights[1].tolist(),
                "W2": self.weights[2].tolist(),
                "b2": self.weights[3].tolist(),
            },
            "speed": int(self.speed),
            "size": int(self.size),
            "vision": int(self.vision),
            "efficiency": int(self.efficiency),
            "lifespan": int(self.lifespan),
            "diet": int(self.diet),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DNA:
        """Reconstruct DNA from a dictionary (loaded from JSON)."""
        weights = [
            np.array(data["weights"]["W1"]),
            np.array(data["weights"]["b1"]),
            np.array(data["weights"]["W2"]),
            np.array(data["weights"]["b2"]),
        ]
        return cls(
            weights=weights,
            speed=data["speed"],
            size=data["size"],
            vision=data["vision"],
            efficiency=data["efficiency"],
            lifespan=data.get("lifespan", DEFAULT_LIFESPAN),
            diet=data.get("diet", DIET_HERBIVORE),
        )

    @classmethod
    def random(cls) -> DNA:
        """Create a DNA with random weights and randomized trait allocation."""
        # Random trait allocation that sums to EVOLUTION_POINTS
        # Generate 4 random cut points, then compute segment lengths for 5 traits
        cuts = sorted(np.random.randint(1, EVOLUTION_POINTS, size=4))
        traits = [
            cuts[0],
            cuts[1] - cuts[0],
            cuts[2] - cuts[1],
            cuts[3] - cuts[2],
            EVOLUTION_POINTS - cuts[3],
        ]
        # Ensure minimum of 1 per trait
        traits = [max(1, t) for t in traits]
        total = sum(traits)
        if total != EVOLUTION_POINTS:
            traits[0] += EVOLUTION_POINTS - total

        # Random neural network weights
        weights = [
            np.random.randn(NN_INPUT_SIZE, NN_HIDDEN_SIZE) * 0.5,
            np.zeros(NN_HIDDEN_SIZE),
            np.random.randn(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE) * 0.5,
            np.zeros(NN_OUTPUT_SIZE),
        ]

        import random as _rng
        diet = _rng.choice([DIET_HERBIVORE, DIET_HERBIVORE, DIET_HERBIVORE,
                            1, 2])  # 60% herbivore, 20% carnivore, 20% scavenger

        return cls(
            weights=weights,
            speed=traits[0],
            size=traits[1],
            vision=traits[2],
            efficiency=traits[3],
            lifespan=traits[4],
            diet=diet,
        )
