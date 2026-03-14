"""
Creature class — the evolving organisms in the simulation.
============================================================
Each creature has a position, velocity, energy, and a neural network brain.
Every frame it senses the world, thinks, acts, and updates its physics.

Sensor inputs (all normalized):
    [0] Distance to nearest food      → 0.0 (touching) to 1.0 (at vision limit / none)
    [1] Angle to nearest food         → -1.0 (hard left) to 1.0 (hard right), 0 if none
    [2] Distance to nearest wall      → 0.0 (touching) to 1.0 (far away)
    [3] Energy level                  → 0.0 (empty) to 1.0 (full or above)
    [4] Distance to nearest creature  → 0.0 (touching) to 1.0 (at vision limit / none)
    [5] Angle to nearest creature     → -1.0 (hard left) to 1.0 (hard right), 0 if none
    [6] Own speed                     → 0.0 (stopped) to 1.0 (max speed)

Brain outputs:
    [0] Turn angle → mapped to [-pi, pi]
    [1] Thrust    → mapped to [0, 1]
"""

from __future__ import annotations

import math

import numpy as np

from pangea.brain import NeuralNetwork
from pangea.config import BASE_ENERGY, ENERGY_COST_PER_THRUST
from pangea.dna import DNA


class Creature:
    """A single organism in the simulation."""

    def __init__(self, dna: DNA, x: float, y: float, lineage: str = "") -> None:
        self.dna = dna
        self.x = x
        self.y = y
        self.heading = np.random.uniform(0, 2 * math.pi)  # facing direction in radians
        self.speed = 0.0  # current speed (0 to max_speed)

        self.energy = BASE_ENERGY
        self.food_eaten = 0
        self.age = 0.0  # seconds alive this generation
        self.alive = True
        self.lineage = lineage  # "" for isolation, "A" or "B" for convergence

        # Build brain from DNA weights
        self.brain = NeuralNetwork()
        self.brain.set_weights(dna.weights)

    # ── Sensors ──────────────────────────────────────────────

    def _find_nearest(
        self,
        targets: list,
        vision: float,
        world_width: float,
        world_height: float,
        wrap: bool,
        skip_self: bool = False,
    ) -> tuple[float, float]:
        """
        Find the nearest target's normalized distance and angle.

        Args:
            targets:      Objects with .x, .y attributes (and .alive if skip_self).
            vision:       Vision range in pixels.
            world_width:  World width for wrap-around.
            world_height: World height for wrap-around.
            wrap:         Whether the world wraps around.
            skip_self:    If True, skip targets that are self or not alive.

        Returns:
            Tuple of (normalized_distance, normalized_angle).
            Distance: 0.0 (touching) to 1.0 (at vision limit or none found).
            Angle: -1.0 to 1.0, 0.0 if none found.
        """
        best_dist = float("inf")
        best_angle = 0.0

        for target in targets:
            if skip_self and (target is self or not target.alive):
                continue

            dx = target.x - self.x
            dy = target.y - self.y

            if wrap:
                if abs(dx) > world_width / 2:
                    dx -= math.copysign(world_width, dx)
                if abs(dy) > world_height / 2:
                    dy -= math.copysign(world_height, dy)

            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist and dist <= vision:
                best_dist = dist
                angle = math.atan2(dy, dx) - self.heading
                best_angle = (angle + math.pi) % (2 * math.pi) - math.pi

        if best_dist <= vision:
            return best_dist / vision, best_angle / math.pi
        return 1.0, 0.0

    def sense(
        self,
        food_list: list,
        world_width: float,
        world_height: float,
        wrap: bool = False,
        vision_multiplier: float = 1.0,
        creatures: list | None = None,
    ) -> np.ndarray:
        """
        Compute the 7 normalized sensor inputs.

        Args:
            food_list:         List of food objects with .x, .y attributes.
            world_width:       Width of the world in pixels.
            world_height:      Height of the world in pixels.
            wrap:              Whether the world wraps around.
            vision_multiplier: Scales effective vision (e.g. for day/night cycle).
            creatures:         List of all creatures for neighbor detection.

        Returns:
            numpy array of shape (7,) with sensor values.
        """
        vision = self.dna.effective_vision * vision_multiplier

        # ─ Nearest food ─
        food_dist_normalized, food_angle_normalized = self._find_nearest(
            food_list, vision, world_width, world_height, wrap,
        )

        # ─ Wall distance ─
        if wrap:
            wall_dist_normalized = 1.0  # No walls in wrap mode
        else:
            dist_left = self.x
            dist_right = world_width - self.x
            dist_top = self.y
            dist_bottom = world_height - self.y
            min_wall_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            wall_dist_normalized = min(min_wall_dist / vision, 1.0)

        # ─ Energy level ─
        energy_normalized = min(self.energy / BASE_ENERGY, 1.0)

        # ─ Nearest creature ─
        if creatures is not None:
            creature_dist_normalized, creature_angle_normalized = self._find_nearest(
                creatures, vision, world_width, world_height, wrap, skip_self=True,
            )
        else:
            creature_dist_normalized = 1.0
            creature_angle_normalized = 0.0

        # ─ Own speed ─
        speed_normalized = min(self.speed / max(self.dna.max_speed, 0.01), 1.0)

        return np.array([
            food_dist_normalized,
            food_angle_normalized,
            wall_dist_normalized,
            energy_normalized,
            creature_dist_normalized,
            creature_angle_normalized,
            speed_normalized,
        ])

    # ── Actions ──────────────────────────────────────────────

    def think_and_act(self, inputs: np.ndarray) -> None:
        """
        Run the brain and apply its outputs to movement.

        Args:
            inputs: Sensor array of shape (7,).
        """
        outputs = self.brain.forward(inputs)

        # Output[0]: turn angle in [-pi, pi]
        turn = outputs[0] * math.pi
        self.heading += turn
        # Keep heading in [0, 2*pi]
        self.heading %= 2 * math.pi

        # Output[1]: thrust in [0, 1]
        thrust = (outputs[1] + 1.0) / 2.0
        self.speed = thrust * self.dna.max_speed

    # ── Physics Update ───────────────────────────────────────

    def update(self, dt: float) -> None:
        """
        Update position and drain energy for one frame.

        Args:
            dt: Time step in seconds.
        """
        if not self.alive:
            return

        # Move
        self.x += math.cos(self.heading) * self.speed * dt * 60  # normalize to 60fps
        self.y += math.sin(self.heading) * self.speed * dt * 60

        # Drain energy — faster and bigger creatures use more energy
        energy_cost = (
            self.speed
            * (1.0 / self.dna.effective_efficiency)
            * ENERGY_COST_PER_THRUST
            * dt
            * 60
        )
        # Small idle cost so creatures can't survive by standing still forever
        idle_cost = 0.05 * dt * 60
        self.energy -= energy_cost + idle_cost

        # Track age
        self.age += dt

        # Check death
        if self.energy <= 0:
            self.energy = 0
            self.alive = False

    # ── Eating ───────────────────────────────────────────────

    def eat(self, food_energy: float) -> None:
        """Gain energy from eating food."""
        self.energy += food_energy
        self.food_eaten += 1
