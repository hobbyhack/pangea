"""
World -- the simulation environment.
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
    INITIAL_FOOD_COUNT,
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
        season_time:  Accumulated time for seasonal food oscillation.
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
        self.season_time = 0.0

        # Accumulator for fractional food spawning
        self._food_spawn_accum = 0.0

        # Spawn initial food
        initial = self.settings.initial_food_count
        for _ in range(initial):
            self.food.append(self._random_food())

    # -- Food Spawning ----------------------------------------------------

    def _random_food(self) -> Food:
        """Create a food item at a random position."""
        margin = FOOD_RADIUS
        x = random.uniform(margin, self.width - margin)
        y = random.uniform(margin, self.height - margin)
        return Food(x=x, y=y, energy=self.settings.food_energy)

    def _spawn_cluster(self, center_x: float, center_y: float) -> None:
        """Spawn a cluster of food items around a center point."""
        cluster_size = self.settings.food_cluster_size
        margin = FOOD_RADIUS
        for _ in range(cluster_size):
            fx = center_x + random.gauss(0, 30)
            fy = center_y + random.gauss(0, 30)
            fx = max(margin, min(self.width - margin, fx))
            fy = max(margin, min(self.height - margin, fy))
            self.food.append(Food(x=fx, y=fy, energy=self.settings.food_energy))

    def seasonal_multiplier(self) -> float:
        """Compute the current seasonal food spawn multiplier."""
        season_length = self.settings.season_length
        min_rate = self.settings.season_min_rate
        return min_rate + (1 - min_rate) * (
            0.5 + 0.5 * math.sin(2 * math.pi * self.season_time / season_length)
        )

    def spawn_food(self, dt: float) -> None:
        """Spawn food probabilistically based on settings and tool state."""
        multiplier = 1.0
        if self.tools:
            multiplier = self.tools.get_food_spawn_multiplier()

        # Apply seasonal multiplier
        multiplier *= self.seasonal_multiplier()

        self._food_spawn_accum += self.settings.food_spawn_rate * multiplier * dt
        while self._food_spawn_accum >= 1.0:
            self._food_spawn_accum -= 1.0
            # Spawn a cluster instead of a single food item
            margin = FOOD_RADIUS
            cx = random.uniform(margin, self.width - margin)
            cy = random.uniform(margin, self.height - margin)
            self._spawn_cluster(cx, cy)

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

    # -- Collision Detection ----------------------------------------------

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

    # -- Boundary Enforcement ---------------------------------------------

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

    # -- Player Tool Effects ----------------------------------------------

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

    # -- Per-Frame Update -------------------------------------------------

    def update(self, dt: float) -> None:
        """
        Advance the simulation by one time step.

        For each living creature: sense -> think -> act -> move ->
        apply tool effects -> check collisions.
        """
        self.elapsed_time += dt
        self.season_time += dt
        wrap = self.settings.world_wrap

        # Spawn food
        self.spawn_food(dt)

        # Update player tools (age zones/barriers)
        if self.tools:
            self.tools.update(dt)

        # Update each creature
        for creature in self.creatures:
            if not creature.alive:
                continue

            # Sense the environment
            inputs = creature.sense(self.food, self.width, self.height, wrap)

            # Think and act
            creature.think_and_act(inputs)

            # Update physics
            creature.update(dt)

            # Apply player tool effects (zones, barriers)
            self.apply_tool_effects(creature, dt)

            # Enforce boundaries
            self.enforce_boundaries(creature)

        # Check for eating
        self.check_collisions()

    # -- Generation Status ------------------------------------------------

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
