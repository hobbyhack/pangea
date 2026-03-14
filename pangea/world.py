"""
World — the simulation environment.
============================================================
Manages the 2D arena, food spawning/decay, creature-food collisions,
boundary enforcement, biome regions, hazard zones, predators,
and player tool effects (zones, barriers).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pangea.config as config
from pangea.config import (
    BIOME_MAX_RADIUS,
    BIOME_MIN_RADIUS,
    BIOME_SPEED_MULTIPLIERS,
    FOOD_ENERGY,
    FOOD_RADIUS,
    HAZARD_DAMAGE,
    HAZARD_MAX_RADIUS,
    HAZARD_MIN_RADIUS,
    PREDATOR_DAMAGE,
    PREDATOR_RADIUS,
    PREDATOR_SPEED,
    PREDATOR_VISION,
    SIZE_ARMOR_SCALE,
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
    age: float = 0.0
    lifetime: float = 0.0  # 0 means no decay (set by World from settings)


@dataclass
class Hazard:
    """A static danger zone that drains energy from creatures on contact."""

    x: float
    y: float
    radius: float
    damage_rate: float = HAZARD_DAMAGE
    hazard_type: str = "lava"  # "lava" or "cold"


@dataclass
class Biome:
    """A circular terrain region with a movement speed multiplier."""

    x: float
    y: float
    radius: float
    biome_type: str          # "water" or "road"
    speed_multiplier: float  # from BIOME_SPEED_MULTIPLIERS


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
        biomes:       List of terrain biome regions.
        elapsed_time: Seconds elapsed in the current generation.
        generation:   Current generation number (starts at 1).
        settings:     Runtime simulation settings.
        tools:        Player interaction tools (zones, barriers, etc.).
        season_time:  Accumulated time for seasonal food oscillation.
    """

    def __init__(
        self,
        creatures: list[Creature],
        width: float | None = None,
        height: float | None = None,
        settings: SimSettings | None = None,
        tools: PlayerTools | None = None,
    ) -> None:
        self.width = width if width is not None else config.WINDOW_WIDTH
        self.height = height if height is not None else config.WINDOW_HEIGHT
        self.creatures = creatures
        self.food: list[Food] = []
        self.elapsed_time = 0.0
        self.generation = 1
        self.settings = settings or SimSettings()
        self.tools = tools
        self.season_time = 0.0

        # Accumulator for fractional food spawning
        self._food_spawn_accum = 0.0

        # Day/night cycle timer
        self.day_night_time = 0.0

        # Generate hazard zones
        self.hazards: list[Hazard] = []
        for _ in range(self.settings.hazard_count):
            self.hazards.append(self._random_hazard())

        # Generate biome regions
        self.biomes: list[Biome] = []
        for _ in range(self.settings.biome_count):
            self.biomes.append(self._random_biome())

        # Generate predators
        self.predators: list[Predator] = []
        for _ in range(self.settings.predator_count):
            px = random.uniform(50, self.width - 50)
            py = random.uniform(50, self.height - 50)
            self.predators.append(Predator(
                x=px, y=py,
                speed=self.settings.predator_speed,
                vision=self.settings.predator_vision,
                damage=self.settings.predator_damage,
                radius=self.settings.predator_radius,
            ))

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

    # ── Biome Generation ──────────────────────────────────────

    def _random_biome(self) -> Biome:
        """Create a biome at a random position with a random type."""
        margin = BIOME_MAX_RADIUS
        x = random.uniform(margin, self.width - margin)
        y = random.uniform(margin, self.height - margin)
        radius = random.uniform(BIOME_MIN_RADIUS, BIOME_MAX_RADIUS)
        biome_type = random.choice(["water", "road"])
        speed_multiplier = BIOME_SPEED_MULTIPLIERS[biome_type]
        return Biome(
            x=x, y=y, radius=radius,
            biome_type=biome_type,
            speed_multiplier=speed_multiplier,
        )

    def get_speed_multiplier(self, x: float, y: float) -> float:
        """
        Get the movement speed multiplier at a given position.

        Checks each biome region; if the point is inside, returns
        that biome's speed multiplier. If inside multiple biomes,
        the first match wins. Returns 1.0 if outside all biomes.
        """
        for biome in self.biomes:
            dx = x - biome.x
            dy = y - biome.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < biome.radius:
                return biome.speed_multiplier
        return 1.0

    # ── Food Spawning ────────────────────────────────────────

    def _make_food(self, x: float, y: float) -> Food:
        """Create a food item with current settings for energy and decay."""
        return Food(
            x=x, y=y,
            energy=self.settings.food_energy,
            lifetime=self.settings.food_decay_time,
        )

    def _random_food(self) -> Food:
        """Create a food item at a random position."""
        margin = FOOD_RADIUS
        x = random.uniform(margin, self.width - margin)
        y = random.uniform(margin, self.height - margin)
        return self._make_food(x, y)

    def _spawn_cluster(self, center_x: float, center_y: float) -> None:
        """Spawn a cluster of food items around a center point."""
        cluster_size = self.settings.food_cluster_size
        margin = FOOD_RADIUS
        for _ in range(cluster_size):
            fx = center_x + random.gauss(0, 30)
            fy = center_y + random.gauss(0, 30)
            fx = max(margin, min(self.width - margin, fx))
            fy = max(margin, min(self.height - margin, fy))
            self.food.append(self._make_food(fx, fy))

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
                    self.food.append(self._make_food(fx, fy))

    def add_food_at(self, x: float, y: float) -> None:
        """Add a food item at a specific position (from player tool)."""
        x = max(FOOD_RADIUS, min(self.width - FOOD_RADIUS, x))
        y = max(FOOD_RADIUS, min(self.height - FOOD_RADIUS, y))
        self.food.append(self._make_food(x, y))

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
        """Drain energy from creatures that overlap with predators.

        Larger creatures take reduced damage — effective_radius acts as armor,
        reducing damage by SIZE_ARMOR_SCALE per pixel of radius.
        """
        for predator in self.predators:
            for creature in self.creatures:
                if not creature.alive:
                    continue
                dx = predator.x - creature.x
                dy = predator.y - creature.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < predator.radius + creature.dna.effective_radius:
                    armor = creature.dna.effective_radius * SIZE_ARMOR_SCALE
                    damage = predator.damage * max(0.1, 1.0 - armor) * dt * 60
                    creature.energy -= damage

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

        For each living creature: sense -> think -> act -> move ->
        apply tool effects -> check collisions.
        """
        self.elapsed_time += dt
        self.day_night_time += dt
        self.season_time += dt
        wrap = self.settings.world_wrap

        # Spawn food
        self.spawn_food(dt)

        # Age food and remove expired items
        for food in self.food:
            food.age += dt
        self.food = [f for f in self.food if f.lifetime <= 0 or f.age < f.lifetime]

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

            # Sense the environment (including predators)
            inputs = creature.sense(
                self.food, self.width, self.height, wrap,
                vision_multiplier=vision_multiplier,
                creatures=self.creatures,
                predators=self.predators,
            )

            # Think and act
            creature.think_and_act(inputs)

            # Compute biome speed multiplier at creature's position
            speed_mult = self.get_speed_multiplier(creature.x, creature.y)

            # Update physics (with biome speed multiplier)
            creature.update(dt, speed_multiplier=speed_mult)

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
