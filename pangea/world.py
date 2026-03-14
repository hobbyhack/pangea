"""
World — the simulation environment.
============================================================
Manages the 2D arena, food spawning, creature-food collisions,
boundary enforcement, and player tool effects (zones, barriers).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pangea.config import (
    FOOD_ENERGY,
    FOOD_RADIUS,
    FOOD_SPAWN_RATE,
    GENERATION_TIME_LIMIT,
    HAZARD_DAMAGE,
    HAZARD_MAX_RADIUS,
    HAZARD_MIN_RADIUS,
    INITIAL_FOOD_COUNT,
    PREDATOR_DAMAGE,
    PREDATOR_RADIUS,
    PREDATOR_SPEED,
    PREDATOR_VISION,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WORLD_WRAP,
)
from pangea.creature import Creature
from pangea.settings import SimSettings
from pangea.tools import PlayerTools


@dataclass
class Food:
    """A single food item in the world."""

    x: float
    y: float
    energy: float = FOOD_ENERGY
    radius: float = FOOD_RADIUS


@dataclass
class Hazard:
    """A static danger zone that drains energy from creatures on contact."""

    x: float
    y: float
    radius: float
    damage_rate: float = HAZARD_DAMAGE
    hazard_type: str = "lava"  # "lava" or "cold"


class Predator:
    """An NPC predator that chases and damages creatures on contact."""

    def __init__(
        self,
        x: float,
        y: float,
        speed: float = PREDATOR_SPEED,
        vision: float = PREDATOR_VISION,
        damage: float = PREDATOR_DAMAGE,
        radius: float = PREDATOR_RADIUS,
    ) -> None:
        self.x = x
        self.y = y
        self.speed = speed
        self.vision = vision
        self.damage = damage
        self.radius = radius
        self.heading = random.uniform(0, 2 * math.pi)

    def update(
        self,
        creatures: list[Creature],
        dt: float,
        width: float,
        height: float,
    ) -> None:
        """Move toward the nearest alive creature, or wander randomly."""
        # Find nearest alive creature within vision
        nearest_dist = float("inf")
        target: Creature | None = None
        for creature in creatures:
            if not creature.alive:
                continue
            dx = creature.x - self.x
            dy = creature.y - self.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.vision and dist < nearest_dist:
                nearest_dist = dist
                target = creature

        if target is not None:
            # Steer toward target
            desired = math.atan2(target.y - self.y, target.x - self.x)
            self.heading = desired
        else:
            # Wander: slight random heading change
            self.heading += random.gauss(0, 0.2)

        # Move
        self.x += math.cos(self.heading) * self.speed * dt * 60
        self.y += math.sin(self.heading) * self.speed * dt * 60

        # Clamp to bounds
        self.x = max(self.radius, min(width - self.radius, self.x))
        self.y = max(self.radius, min(height - self.radius, self.y))


class World:
    """
    The simulation environment containing creatures and food.

    Attributes:
        width:        Arena width in pixels.
        height:       Arena height in pixels.
        creatures:    List of living (and dead) creatures this generation.
        food:         List of available food items.
        elapsed_time: Seconds elapsed in the current generation.
        generation:   Current generation number (starts at 1).
        settings:     Runtime simulation settings.
        tools:        Player interaction tools (zones, barriers, etc.).
    """

    def __init__(
        self,
        creatures: list[Creature],
        width: float = WINDOW_WIDTH,
        height: float = WINDOW_HEIGHT,
        settings: SimSettings | None = None,
        tools: PlayerTools | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.creatures = creatures
        self.food: list[Food] = []
        self.elapsed_time = 0.0
        self.generation = 1
        self.settings = settings or SimSettings()
        self.tools = tools

        # Accumulator for fractional food spawning
        self._food_spawn_accum = 0.0

        # Day/night cycle timer
        self.day_night_time = 0.0

        # Generate hazard zones
        self.hazards: list[Hazard] = []
        for _ in range(self.settings.hazard_count):
            self.hazards.append(self._random_hazard())

        # Generate predators
        self.predators: list[Predator] = []
        for _ in range(self.settings.predator_count):
            px = random.uniform(50, self.width - 50)
            py = random.uniform(50, self.height - 50)
            self.predators.append(Predator(x=px, y=py))

        # Spawn initial food
        initial = self.settings.initial_food_count
        for _ in range(initial):
            self.food.append(self._random_food())

    # ── Day/Night Cycle ───────────────────────────────────────

    @property
    def daylight_factor(self) -> float:
        """Return a value from 0.0 (full night) to 1.0 (full day)."""
        cycle = self.settings.day_night_cycle_length
        return 0.5 + 0.5 * math.sin(2 * math.pi * self.day_night_time / cycle)

    # ── Hazard Generation ─────────────────────────────────────

    def _random_hazard(self) -> Hazard:
        """Create a hazard zone at a random position, avoiding edges."""
        margin = 50.0
        x = random.uniform(margin, self.width - margin)
        y = random.uniform(margin, self.height - margin)
        radius = random.uniform(HAZARD_MIN_RADIUS, HAZARD_MAX_RADIUS)
        hazard_type = random.choice(["lava", "cold"])
        return Hazard(x=x, y=y, radius=radius, damage_rate=HAZARD_DAMAGE, hazard_type=hazard_type)

    # ── Food Spawning ────────────────────────────────────────

    def _random_food(self) -> Food:
        """Create a food item at a random position."""
        margin = FOOD_RADIUS
        x = random.uniform(margin, self.width - margin)
        y = random.uniform(margin, self.height - margin)
        return Food(x=x, y=y, energy=self.settings.food_energy)

    def spawn_food(self, dt: float) -> None:
        """Spawn food probabilistically based on settings and tool state."""
        multiplier = 1.0
        if self.tools:
            multiplier = self.tools.get_food_spawn_multiplier()

        self._food_spawn_accum += self.settings.food_spawn_rate * multiplier * dt
        while self._food_spawn_accum >= 1.0:
            self._food_spawn_accum -= 1.0
            self.food.append(self._random_food())

        # Bounty zones spawn extra food nearby
        if self.tools:
            for zone in self.tools.zones:
                if zone.zone_type == "bounty" and random.random() < 0.05 * zone.opacity:
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(0, zone.radius * 0.8)
                    fx = zone.x + math.cos(angle) * dist
                    fy = zone.y + math.sin(angle) * dist
                    fx = max(FOOD_RADIUS, min(self.width - FOOD_RADIUS, fx))
                    fy = max(FOOD_RADIUS, min(self.height - FOOD_RADIUS, fy))
                    self.food.append(Food(x=fx, y=fy, energy=self.settings.food_energy))

    def add_food_at(self, x: float, y: float) -> None:
        """Add a food item at a specific position (from player tool)."""
        x = max(FOOD_RADIUS, min(self.width - FOOD_RADIUS, x))
        y = max(FOOD_RADIUS, min(self.height - FOOD_RADIUS, y))
        self.food.append(Food(x=x, y=y, energy=self.settings.food_energy))

    # ── Collision Detection ──────────────────────────────────

    def check_collisions(self) -> None:
        """Check and handle creature-food collisions."""
        wrap = self.settings.world_wrap
        for creature in self.creatures:
            if not creature.alive:
                continue

            cr = creature.dna.effective_radius
            eaten_indices = []

            for i, food in enumerate(self.food):
                dx = creature.x - food.x
                dy = creature.y - food.y

                if wrap:
                    if abs(dx) > self.width / 2:
                        dx -= math.copysign(self.width, dx)
                    if abs(dy) > self.height / 2:
                        dy -= math.copysign(self.height, dy)

                dist = math.sqrt(dx * dx + dy * dy)
                if dist < cr + food.radius:
                    creature.eat(food.energy)
                    eaten_indices.append(i)

            for i in reversed(eaten_indices):
                self.food.pop(i)

    # ── Boundary Enforcement ─────────────────────────────────

    def enforce_boundaries(self, creature: Creature) -> None:
        """Keep creature inside the world (clamp or wrap)."""
        r = creature.dna.effective_radius

        if self.settings.world_wrap:
            if creature.x < 0:
                creature.x += self.width
            elif creature.x > self.width:
                creature.x -= self.width
            if creature.y < 0:
                creature.y += self.height
            elif creature.y > self.height:
                creature.y -= self.height
        else:
            creature.x = max(r, min(self.width - r, creature.x))
            creature.y = max(r, min(self.height - r, creature.y))

    # ── Hazard Effects ────────────────────────────────────────

    def _apply_hazard_effects(self, creature: Creature, dt: float) -> None:
        """Drain energy from a creature that overlaps with hazard zones."""
        for hazard in self.hazards:
            dx = creature.x - hazard.x
            dy = creature.y - hazard.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < hazard.radius:
                # Damage scales linearly: full at center, zero at edge
                intensity = (1.0 - dist / hazard.radius) * hazard.damage_rate
                creature.energy -= intensity * dt * 60

    # ── Predator Collisions ──────────────────────────────────

    def _check_predator_collisions(self, dt: float) -> None:
        """Drain energy from creatures that overlap with predators."""
        for predator in self.predators:
            for creature in self.creatures:
                if not creature.alive:
                    continue
                dx = predator.x - creature.x
                dy = predator.y - creature.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < predator.radius + creature.dna.effective_radius:
                    creature.energy -= predator.damage * dt * 60

    # ── Player Tool Effects ──────────────────────────────────

    def apply_tool_effects(self, creature: Creature, dt: float) -> None:
        """Apply zone effects and barrier collisions from player tools."""
        if not self.tools:
            return

        # Zone energy effects
        modifier = self.tools.get_energy_modifier(creature.x, creature.y)
        if modifier != 0.0:
            creature.energy += modifier * dt * 60

        # Barrier collisions
        push = self.tools.check_barrier_collision(
            creature.x, creature.y, creature.dna.effective_radius
        )
        if push is not None:
            creature.x += push[0]
            creature.y += push[1]

    # ── Per-Frame Update ─────────────────────────────────────

    def update(self, dt: float) -> None:
        """
        Advance the simulation by one time step.

        For each living creature: sense → think → act → move →
        apply tool effects → check collisions.
        """
        self.elapsed_time += dt
        self.day_night_time += dt
        wrap = self.settings.world_wrap

        # Spawn food
        self.spawn_food(dt)

        # Update player tools (age zones/barriers)
        if self.tools:
            self.tools.update(dt)

        # Compute vision multiplier from day/night cycle
        night_mult = self.settings.night_vision_multiplier
        vision_multiplier = night_mult + (1 - night_mult) * self.daylight_factor

        # Update each creature
        for creature in self.creatures:
            if not creature.alive:
                continue

            # Sense the environment
            inputs = creature.sense(
                self.food, self.width, self.height, wrap,
                vision_multiplier=vision_multiplier,
            )

            # Think and act
            creature.think_and_act(inputs)

            # Update physics
            creature.update(dt)

            # Apply player tool effects (zones, barriers)
            self.apply_tool_effects(creature, dt)

            # Apply hazard zone damage
            self._apply_hazard_effects(creature, dt)

            # Enforce boundaries
            self.enforce_boundaries(creature)

        # Update predators
        for predator in self.predators:
            predator.update(self.creatures, dt, self.width, self.height)
        self._check_predator_collisions(dt)

        # Check for eating
        self.check_collisions()

    # ── Generation Status ────────────────────────────────────

    def is_generation_over(self) -> bool:
        """Check if the current generation has ended."""
        all_dead = all(not c.alive for c in self.creatures)
        time_up = self.elapsed_time >= self.settings.generation_time_limit
        return all_dead or time_up

    def alive_count(self) -> int:
        """Return the number of living creatures."""
        return sum(1 for c in self.creatures if c.alive)

    def alive_count_by_lineage(self, lineage: str) -> int:
        """Return the number of living creatures in a specific lineage."""
        return sum(1 for c in self.creatures if c.alive and c.lineage == lineage)

    def food_eaten_by_lineage(self, lineage: str) -> int:
        """Return total food eaten by a specific lineage."""
        return sum(c.food_eaten for c in self.creatures if c.lineage == lineage)
