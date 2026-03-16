"""
Creature class — the evolving organisms in the simulation.
============================================================
Each creature has a position, velocity, energy, and a neural network brain.
Every frame it senses the world, thinks, acts, and updates its physics.

Sensor inputs (all normalized):
    [0] Distance to nearest food      -> 0.0 (touching) to 1.0 (at vision limit / none)
    [1] Angle to nearest food         -> -1.0 (hard left) to 1.0 (hard right), 0 if none
    [2] Distance to nearest wall      -> 0.0 (touching) to 1.0 (far away)
    [3] Energy level                  -> 0.0 (empty) to 1.0 (full or above)
    [4] Distance to nearest creature  -> 0.0 (touching) to 1.0 (at vision limit / none)
    [5] Angle to nearest creature     -> -1.0 (hard left) to 1.0 (hard right), 0 if none
    [6] Own speed                     -> 0.0 (stopped) to 1.0 (max speed)
    [7] Distance to nearest threat    -> 0.0 (touching) to 1.0 (at vision limit / none)
    [8] Angle to nearest threat       -> -1.0 (hard left) to 1.0 (hard right), 0 if none
    [9] Under attack                  -> 0.0 (safe) to 1.0 (taking damage)
    [10] Biome speed                   -> 0.0 (very slow terrain) to 1.0 (fast terrain)
    [11] Biome danger                  -> 0.0 (safe/beneficial) to 1.0 (high energy drain)

Brain outputs:
    [0] Turn angle -> mapped to [-pi, pi]
    [1] Thrust    -> mapped to [0, 1]
"""

from __future__ import annotations

import math

import numpy as np

from pangea.brain import NeuralNetwork
from pangea.config import (
    BASE_ENERGY,
    ENERGY_COST_PER_THRUST,
    MIN_THRUST_FRACTION,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pangea.settings import SimSettings
    from pangea.species import Species
from pangea.dna import DNA


class Creature:
    """A single organism in the simulation."""

    def __init__(self, dna: DNA, x: float, y: float,
                 species: Species | None = None) -> None:
        self.dna = dna
        self.species = species  # Species reference for diet behavior
        self.x = x
        self.y = y
        self.heading = np.random.uniform(0, 2 * math.pi)  # facing direction in radians
        self.speed = 0.0  # current speed (0 to max_speed)

        self.energy = (species.settings.base_energy if species else BASE_ENERGY)
        self.food_eaten = 0
        self.age = 0.0  # seconds alive this generation
        self.alive = True
        self.last_turn = 0.0  # absolute radians turned last frame
        self.under_attack = 0.0  # 0.0 = safe, 1.0 = taking damage this frame
        self.death_processed = False  # True once scavenger rewards have been given
        self.territory_cells: set[tuple[int, int]] = set()  # visited grid cells
        self.distance_traveled: float = 0.0  # total pixels moved (for fitness)

        # Freeplay breeding state
        self.breed_cooldown: float = 0.0   # seconds until next breed allowed
        self.offspring_count: int = 0       # total children produced
        self.generation: int = 0            # 0 = founding population
        self.time_to_first_food: float = -1.0  # seconds until first food eaten
        self.energy_at_death: float = -1.0     # energy when creature died

        # Build brain from DNA weights
        self.brain = NeuralNetwork()
        self.brain.set_weights(dna.weights)

    # ── Breeding ─────────────────────────────────────────────

    def can_breed(self, settings) -> bool:
        """Check if this creature meets all freeplay breeding criteria.

        Args:
            settings: SimSettings or SpeciesSettings — reads freeplay_breed_*
                      fields which exist on both.
        """
        return (
            self.alive
            and self.age >= settings.freeplay_breed_min_age
            and self.food_eaten >= settings.freeplay_breed_min_food
            and self.energy >= settings.freeplay_breed_energy_threshold * settings.base_energy
            and self.breed_cooldown <= 0
        )

    # ── Sensors ──────────────────────────────────────────────

    def _find_threats(self, creatures: list) -> list:
        """Return the subset of creatures that are hostile threats to this one.

        A creature is a threat if it belongs to a species that can attack
        and is permitted to attack this creature's species (same vs other).
        """
        my_species_id = self.dna.species_id
        threats = []
        for other in creatures:
            if other is self or not other.alive:
                continue
            sp = other.species
            if sp is None or not sp.can_attack:
                continue
            same_species = other.dna.species_id == my_species_id
            if same_species and not sp.can_attack_own_species:
                continue
            if not same_species and not sp.can_attack_other_species:
                continue
            threats.append(other)
        return threats

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
        half_w = world_width * 0.5
        half_h = world_height * 0.5

        for target in targets:
            if skip_self and (target is self or not target.alive):
                continue

            dx = target.x - self.x
            dy = target.y - self.y

            if wrap:
                if abs(dx) > half_w:
                    dx -= math.copysign(world_width, dx)
                if abs(dy) > half_h:
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
        biome_speed: float = 1.0,
        biome_danger: float = 0.0,
    ) -> np.ndarray:
        """
        Compute the 12 normalized sensor inputs.

        Args:
            food_list:         List of food objects with .x, .y attributes.
            world_width:       Width of the world in pixels.
            world_height:      Height of the world in pixels.
            wrap:              Whether the world wraps around.
            vision_multiplier: Scales effective vision (e.g. for day/night cycle).
            creatures:         List of all creatures for neighbor detection.
            biome_speed:       Speed multiplier at creature's position (0.3–1.5).
            biome_danger:      Energy drain rate at creature's position (0.0–1.0).

        Returns:
            numpy array of shape (12,) with sensor values.
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
        _base = self.species.settings.base_energy if self.species else BASE_ENERGY
        energy_normalized = min(self.energy / _base, 1.0)

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

        # ─ Nearest threat (hostile creature that can attack this one) ─
        threat_dist_normalized = 1.0
        threat_angle_normalized = 0.0
        if creatures is not None:
            threats = self._find_threats(creatures)
            if threats:
                threat_dist_normalized, threat_angle_normalized = self._find_nearest(
                    threats, vision, world_width, world_height, wrap,
                )

        # ─ Biome sensors ─
        # Normalize speed: 0.3–1.5 range → 0.0–1.0
        biome_speed_normalized = min(max((biome_speed - 0.3) / 1.2, 0.0), 1.0)
        biome_danger_normalized = min(biome_danger, 1.0)

        return np.array([
            food_dist_normalized,
            food_angle_normalized,
            wall_dist_normalized,
            energy_normalized,
            creature_dist_normalized,
            creature_angle_normalized,
            speed_normalized,
            threat_dist_normalized,
            threat_angle_normalized,
            min(self.under_attack, 1.0),
            biome_speed_normalized,
            biome_danger_normalized,
        ])

    # ── Actions ──────────────────────────────────────────────

    def think_and_act(self, inputs: np.ndarray) -> None:
        """
        Run the brain and apply its outputs to movement.

        Args:
            inputs: Sensor array of shape (12,).
        """
        outputs = self.brain.forward(inputs)

        # Output[0]: turn angle in [-pi, pi]
        turn = outputs[0] * math.pi
        self.last_turn = abs(turn)
        self.heading += turn
        # Keep heading in [0, 2*pi]
        self.heading %= 2 * math.pi

        # Output[1]: thrust in [MIN_THRUST_FRACTION, 1]
        # Creatures always have a minimum forward speed so spinning in place
        # is not a viable strategy — they must navigate, not idle.
        raw_thrust = (outputs[1] + 1.0) / 2.0
        thrust = MIN_THRUST_FRACTION + (1.0 - MIN_THRUST_FRACTION) * raw_thrust
        self.speed = thrust * self.dna.max_speed

    # ── Physics Update ───────────────────────────────────────

    def update(self, dt: float, speed_multiplier: float = 1.0) -> None:
        """
        Update position and drain energy for one frame.

        Args:
            dt:               Time step in seconds.
            speed_multiplier: Biome speed multiplier (default 1.0 for normal terrain).
                              Values < 1.0 slow movement (e.g. water),
                              values > 1.0 speed movement (e.g. road).
                              Energy cost is NOT affected by this multiplier.
        """
        if not self.alive:
            return

        # Move (apply biome speed multiplier to movement only)
        dx = math.cos(self.heading) * self.speed * speed_multiplier * dt * 60
        dy = math.sin(self.heading) * self.speed * speed_multiplier * dt * 60
        self.x += dx
        self.y += dy
        self.distance_traveled += math.sqrt(dx * dx + dy * dy)

        # Drain energy — faster and bigger creatures use more energy
        # Energy cost does NOT scale with biome multiplier
        _ecpt = (self.species.settings.energy_cost_per_thrust
                 if self.species else ENERGY_COST_PER_THRUST)
        energy_cost = (
            self.speed
            * (1.0 / self.dna.effective_efficiency)
            * _ecpt
            * dt
            * 60
        )
        # Idle cost scales with turning — spinning in place is expensive.
        # Base idle cost + extra proportional to turn angle (last_turn is 0..pi).
        idle_cost = 0.05 * (1.0 + self.last_turn) * dt * 60
        self.energy -= energy_cost + idle_cost

        # Track age
        self.age += dt

        # Tick breeding cooldown
        if self.breed_cooldown > 0:
            self.breed_cooldown -= dt

        # Check death — energy depletion
        if self.energy <= 0:
            self.energy = 0
            self.energy_at_death = self.energy
            self.alive = False

        # Check death — lifespan exceeded
        if self.alive and self.age >= self.dna.effective_lifespan:
            self.energy_at_death = self.energy
            self.alive = False

    # ── Eating ───────────────────────────────────────────────

    def eat(self, food_energy: float, lifespan_heal: float = 0.0) -> None:
        """Gain energy from eating food, scaled by species plant_food_multiplier."""
        if self.time_to_first_food < 0:
            self.time_to_first_food = self.age
        if self.species is not None:
            food_energy *= self.species.plant_food_multiplier

        self.energy += food_energy
        self.food_eaten += 1
        if lifespan_heal > 0:
            self.age = max(0, self.age - lifespan_heal)

    def gain_energy(self, amount: float) -> None:
        """Gain energy from non-food sources (carnivore attacks, scavenger bonus)."""
        self.energy += amount
