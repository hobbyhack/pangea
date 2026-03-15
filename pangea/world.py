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
    BIOME_ENERGY_DRAIN,
    BIOME_FOOD_MULTIPLIER,
    BIOME_MAX_RADIUS,
    BIOME_MIN_RADIUS,
    BIOME_PREDATOR_BLOCKED,
    BIOME_SPEED_MULTIPLIERS,
    CARNIVORE_ATTACK_DAMAGE,
    CARNIVORE_ATTACK_RANGE,
    CARNIVORE_ENERGY_STEAL,
    DIET_CARNIVORE,
    DIET_SCAVENGER,
    FOOD_ENERGY,
    FOOD_RADIUS,
    HAZARD_DAMAGE,
    HAZARD_MAX_RADIUS,
    HAZARD_MIN_RADIUS,
    PREDATOR_DAMAGE,
    PREDATOR_RADIUS,
    PREDATOR_SPEED,
    PREDATOR_VISION,
    CORPSE_ENERGY,
    CORPSE_RADIUS,
    SCAVENGER_DEATH_ENERGY,
    SCAVENGER_DEATH_RADIUS,
    SIZE_ARMOR_SCALE,
    TERRITORY_GRID_SIZE,
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
    is_corpse: bool = False  # True = scavenger-only corpse food


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
        stamina: float = 0.0,
    ) -> None:
        self.x = x
        self.y = y
        self.speed = speed
        self.vision = vision
        self.damage = damage
        self.radius = radius
        self.heading = random.uniform(0, 2 * math.pi)
        self.stamina = stamina  # max chase seconds (0 = infinite)
        self.chase_time = 0.0
        self.rest_time = 0.0
        self.resting = False

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

        if self.resting:
            # Wander while resting
            self.heading += random.gauss(0, 0.2)
            self.rest_time += dt
            # Recover after resting for half the stamina duration
            if self.stamina > 0 and self.rest_time >= self.stamina * 0.5:
                self.resting = False
                self.rest_time = 0.0
                self.chase_time = 0.0
        elif target is not None:
            # Steer toward target
            desired = math.atan2(target.y - self.y, target.x - self.x)
            self.heading = desired
            # Track chase fatigue
            if self.stamina > 0:
                self.chase_time += dt
                if self.chase_time >= self.stamina:
                    self.resting = True
        else:
            # Wander: slight random heading change
            self.heading += random.gauss(0, 0.2)
            # Recover chase time when not chasing
            if self.chase_time > 0:
                self.chase_time = max(0, self.chase_time - dt * 0.5)

        # Move (slower when resting)
        move_speed = self.speed * (0.3 if self.resting else 1.0)
        self.x += math.cos(self.heading) * move_speed * dt * 60
        self.y += math.sin(self.heading) * move_speed * dt * 60

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
        self.settings = settings or SimSettings()
        self.width = width if width is not None else self.settings.world_width
        self.height = height if height is not None else self.settings.world_height
        self.creatures = creatures
        self.food: list[Food] = []
        self.elapsed_time = 0.0
        self.generation = 1
        self.tools = tools
        self.season_time = 0.0
        self.freeplay = False  # Set True for continuous breeding mode
        self.total_births = 0  # Freeplay birth counter
        self.total_deaths = 0  # Freeplay death counter

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
        if self.settings.biomes_enabled:
            for _ in range(self.settings.biome_count):
                self.biomes.append(self._random_biome())

        # Generate predators
        self.predators: list[Predator] = []
        for _ in range(self.settings.predator_count):
            self.predators.append(self._spawn_predator())
        self._predator_respawn_timer = 0.0

        # Spawn initial food
        initial = self.settings.initial_food_count
        for _ in range(initial):
            self.food.append(self._random_food())

    # ── Dynamic Resize ─────────────────────────────────────────

    def resize(self, new_width: float, new_height: float) -> None:
        """Update world bounds to match new window size, clamping entities."""
        self.width = new_width
        self.height = new_height
        self.settings.world_width = int(new_width)
        self.settings.world_height = int(new_height)

        # Clamp living creatures
        for creature in self.creatures:
            if creature.alive:
                r = creature.dna.effective_radius
                creature.x = max(r, min(new_width - r, creature.x))
                creature.y = max(r, min(new_height - r, creature.y))

        # Clamp food
        for food in self.food:
            food.x = max(food.radius, min(new_width - food.radius, food.x))
            food.y = max(food.radius, min(new_height - food.radius, food.y))

        # Clamp predators
        for pred in self.predators:
            pred.x = max(pred.radius, min(new_width - pred.radius, pred.x))
            pred.y = max(pred.radius, min(new_height - pred.radius, pred.y))

        # Clamp hazards
        for hazard in self.hazards:
            hazard.x = max(hazard.radius, min(new_width - hazard.radius, hazard.x))
            hazard.y = max(hazard.radius, min(new_height - hazard.radius, hazard.y))

        # Clamp biomes
        for biome in self.biomes:
            biome.x = max(biome.radius, min(new_width - biome.radius, biome.x))
            biome.y = max(biome.radius, min(new_height - biome.radius, biome.y))

    # ── Day/Night Cycle ───────────────────────────────────────

    @property
    def daylight_factor(self) -> float:
        """Return a value from 0.0 (full night) to 1.0 (full day)."""
        if not self.settings.day_night_enabled:
            return 1.0
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
        biome_type = random.choice(
            ["water", "road", "forest", "desert", "swamp", "mountain"]
        )
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

    def get_biome_at(self, x: float, y: float) -> Biome | None:
        """Return the biome at a position, or None if outside all biomes."""
        for biome in self.biomes:
            dx = x - biome.x
            dy = y - biome.y
            if math.sqrt(dx * dx + dy * dy) < biome.radius:
                return biome
        return None

    def is_in_biome_type(self, x: float, y: float, biome_type: str) -> bool:
        """Check if a position is inside a biome of the given type."""
        biome = self.get_biome_at(x, y)
        return biome is not None and biome.biome_type == biome_type

    # ── Biome Effects ─────────────────────────────────────────

    def _apply_biome_effects(self, creature: Creature, dt: float) -> None:
        """Apply special biome effects (energy drain) to a creature."""
        biome = self.get_biome_at(creature.x, creature.y)
        if biome is None:
            return
        drain = BIOME_ENERGY_DRAIN.get(biome.biome_type, 0.0)
        if drain > 0:
            creature.energy -= drain * dt * 60

    # ── Predator Spawning ──────────────────────────────────────

    def _spawn_predator(self) -> Predator:
        """Create a predator at a random position with current settings."""
        px = random.uniform(50, self.width - 50)
        py = random.uniform(50, self.height - 50)
        return Predator(
            x=px, y=py,
            speed=self.settings.predator_speed,
            vision=self.settings.predator_vision,
            damage=self.settings.predator_damage,
            radius=self.settings.predator_radius,
            stamina=self.settings.predator_stamina,
        )

    # ── Food Spawning ────────────────────────────────────────

    def _make_food(self, x: float, y: float) -> Food:
        """Create a food item with current settings for energy and decay."""
        return Food(
            x=x, y=y,
            energy=self.settings.food_energy,
            lifetime=self.settings.food_decay_time,
        )

    def _at_food_max(self) -> bool:
        """Return True if food count is at or above the configured maximum."""
        food_max = self.settings.food_max
        return food_max > 0 and len(self.food) >= food_max

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
        if not self.settings.season_enabled:
            return 1.0
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

        # Freeplay: reduce food when above total carrying capacity
        if self.freeplay:
            alive = self.alive_count()
            cap = self.settings.total_freeplay_carrying_capacity()
            if alive > cap:
                multiplier *= self.settings.freeplay_overcapacity_food_penalty

        # Enforce food_max: skip spawning if already at or above max
        if self._at_food_max():
            self._food_spawn_accum = 0.0
        else:
            self._food_spawn_accum += self.settings.food_spawn_rate * multiplier * dt
            while self._food_spawn_accum >= 1.0:
                self._food_spawn_accum -= 1.0
                if self._at_food_max():
                    self._food_spawn_accum = 0.0
                    break
                # Spawn a cluster instead of a single food item
                margin = FOOD_RADIUS
                cx = random.uniform(margin, self.width - margin)
                cy = random.uniform(margin, self.height - margin)
                self._spawn_cluster(cx, cy)

        # Bounty zones spawn extra food nearby
        if self.tools:
            for zone in self.tools.zones:
                if zone.zone_type == "bounty" and random.random() < 0.05 * zone.opacity:
                    if self._at_food_max():
                        break
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(0, zone.radius * 0.8)
                    fx = zone.x + math.cos(angle) * dist
                    fy = zone.y + math.sin(angle) * dist
                    fx = max(FOOD_RADIUS, min(self.width - FOOD_RADIUS, fx))
                    fy = max(FOOD_RADIUS, min(self.height - FOOD_RADIUS, fy))
                    self.food.append(self._make_food(fx, fy))

        # Enforce food_min: top up if below minimum (cap per frame to avoid spikes)
        food_min = self.settings.food_min
        if food_min > 0 and len(self.food) < food_min:
            deficit = food_min - len(self.food)
            for _ in range(min(deficit, self.settings.food_cluster_size)):
                self.food.append(self._random_food())

    def add_food_at(self, x: float, y: float) -> None:
        """Add a food item at a specific position (from player tool)."""
        x = max(FOOD_RADIUS, min(self.width - FOOD_RADIUS, x))
        y = max(FOOD_RADIUS, min(self.height - FOOD_RADIUS, y))
        self.food.append(self._make_food(x, y))

    # ── Collision Detection ──────────────────────────────────

    def check_collisions(self) -> None:
        """Check and handle creature-food collisions."""
        wrap = self.settings.world_wrap
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        for creature in self.creatures:
            if not creature.alive:
                continue

            cr = creature.dna.effective_radius
            eaten_indices = []

            is_scavenger = creature.dna.diet == DIET_SCAVENGER
            for i, food in enumerate(self.food):
                # Only scavengers can eat corpses
                if food.is_corpse and not is_scavenger:
                    continue

                dx = creature.x - food.x
                dy = creature.y - food.y

                if wrap:
                    if abs(dx) > half_w:
                        dx -= math.copysign(self.width, dx)
                    if abs(dy) > half_h:
                        dy -= math.copysign(self.height, dy)

                dist = math.sqrt(dx * dx + dy * dy)
                if dist < cr + food.radius:
                    creature.eat(food.energy, self.settings.food_heal)
                    eaten_indices.append(i)

            for i in reversed(eaten_indices):
                self.food.pop(i)

            # Chance to spawn replacement food at a random location
            if self.settings.food_respawn_chance > 0:
                for _ in eaten_indices:
                    if random.random() < self.settings.food_respawn_chance:
                        self.food.append(self._random_food())

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
                    creature.under_attack = 1.0

    # ── Carnivore Combat ──────────────────────────────────────

    def _check_carnivore_attacks(self, dt: float) -> None:
        """Carnivores damage nearby creatures and steal energy."""
        for attacker in self.creatures:
            if not attacker.alive or attacker.dna.diet != DIET_CARNIVORE:
                continue
            attack_range = attacker.dna.effective_radius * CARNIVORE_ATTACK_RANGE
            for victim in self.creatures:
                if victim is attacker or not victim.alive:
                    continue
                dx = attacker.x - victim.x
                dy = attacker.y - victim.y
                dist = math.sqrt(dx * dx + dy * dy)
                contact = attack_range + victim.dna.effective_radius
                if dist < contact:
                    damage = CARNIVORE_ATTACK_DAMAGE * dt * 60
                    victim.energy -= damage
                    victim.under_attack = 1.0
                    attacker.gain_energy(damage * CARNIVORE_ENERGY_STEAL)

    # ── Scavenger Death Detection ────────────────────────────

    def _reward_scavengers(self, newly_dead: list[Creature]) -> None:
        """Give energy to scavengers near recently dead creatures and spawn corpses."""
        for dead in newly_dead:
            # Scavengers can't eat other scavengers
            if dead.dna.diet == DIET_SCAVENGER:
                continue
            # Instant proximity bonus for scavengers already nearby
            for creature in self.creatures:
                if not creature.alive or creature.dna.diet != DIET_SCAVENGER:
                    continue
                dx = creature.x - dead.x
                dy = creature.y - dead.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < SCAVENGER_DEATH_RADIUS:
                    creature.gain_energy(SCAVENGER_DEATH_ENERGY)
            # Spawn a corpse food item so scavengers can find it later
            self.food.append(Food(
                x=dead.x,
                y=dead.y,
                energy=CORPSE_ENERGY,
                radius=CORPSE_RADIUS,
                lifetime=self.settings.corpse_decay_time,
                is_corpse=True,
            ))

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

        # Age food and remove expired items (only rebuild list when needed)
        has_expiry = False
        for food in self.food:
            food.age += dt
            if food.lifetime > 0 and food.age >= food.lifetime:
                has_expiry = True
        if has_expiry:
            self.food = [f for f in self.food if f.lifetime <= 0 or f.age < f.lifetime]

        # Update player tools (age zones/barriers)
        if self.tools:
            self.tools.update(dt)

        # Compute vision multiplier from day/night cycle
        night_mult = self.settings.night_vision_multiplier
        vision_multiplier = night_mult + (1 - night_mult) * self.daylight_factor

        # Reset under_attack flag for all creatures
        for creature in self.creatures:
            creature.under_attack = 0.0

        # Update each creature
        for creature in self.creatures:
            if not creature.alive:
                continue

            # Compute biome info at creature's position (single lookup)
            biome = self.get_biome_at(creature.x, creature.y)
            if biome is not None:
                speed_mult = biome.speed_multiplier
                biome_danger = BIOME_ENERGY_DRAIN.get(biome.biome_type, 0.0)
            else:
                speed_mult = 1.0
                biome_danger = 0.0

            # Sense the environment (including predators and biome)
            inputs = creature.sense(
                self.food, self.width, self.height, wrap,
                vision_multiplier=vision_multiplier,
                creatures=self.creatures,
                predators=self.predators,
                biome_speed=speed_mult,
                biome_danger=biome_danger,
            )

            # Think and act
            creature.think_and_act(inputs)

            # Update physics (with biome speed multiplier)
            creature.update(dt, speed_multiplier=speed_mult)

            # Turn cost — energy penalty proportional to turning
            if self.settings.turn_cost > 0:
                creature.energy -= (
                    creature.last_turn * self.settings.turn_cost * dt * 60
                )

            # Territory tracking
            if TERRITORY_GRID_SIZE > 0:
                cell = (
                    int(creature.x // TERRITORY_GRID_SIZE),
                    int(creature.y // TERRITORY_GRID_SIZE),
                )
                creature.territory_cells.add(cell)

            # Apply player tool effects (zones, barriers)
            self.apply_tool_effects(creature, dt)

            # Apply hazard zone damage
            self._apply_hazard_effects(creature, dt)

            # Apply biome special effects (desert/swamp energy drain)
            self._apply_biome_effects(creature, dt)

            # Enforce boundaries
            self.enforce_boundaries(creature)

        # Carnivore attacks
        self._check_carnivore_attacks(dt)

        # Update predators (blocked from mountain biomes)
        blocked_biomes = [b for b in self.biomes if b.biome_type in BIOME_PREDATOR_BLOCKED]
        for predator in self.predators:
            predator.update(self.creatures, dt, self.width, self.height)
            # Push predators out of blocked biomes
            for biome in blocked_biomes:
                    dx = predator.x - biome.x
                    dy = predator.y - biome.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < biome.radius:
                        # Push predator to edge of biome
                        if dist > 0:
                            nx, ny = dx / dist, dy / dist
                        else:
                            nx, ny = 1.0, 0.0
                        predator.x = biome.x + nx * biome.radius
                        predator.y = biome.y + ny * biome.radius
        self._check_predator_collisions(dt)

        # Sync predator count with settings
        desired = self.settings.predator_count
        while len(self.predators) < desired:
            self.predators.append(self._spawn_predator())
        while len(self.predators) > desired:
            self.predators.pop()

        # Predator respawn (replace killed predators over time)
        if self.settings.predator_respawn_interval > 0:
            self._predator_respawn_timer += dt
            if self._predator_respawn_timer >= self.settings.predator_respawn_interval:
                self._predator_respawn_timer = 0.0
                if len(self.predators) < desired:
                    self.predators.append(self._spawn_predator())

        # Reward scavengers near deaths
        # Collect newly dead after all damage (predator + carnivore + hazard)
        frame_dead = [
            c for c in self.creatures
            if not c.alive and not c.death_processed
        ]
        self._reward_scavengers(frame_dead)
        for c in frame_dead:
            c.death_processed = True

        # Bonus food spawning in forest biomes
        for biome in self.biomes:
            food_mult = BIOME_FOOD_MULTIPLIER.get(biome.biome_type, 0.0)
            if food_mult > 1.0 and random.random() < 0.02 * (food_mult - 1.0):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0, biome.radius * 0.8)
                fx = biome.x + math.cos(angle) * dist
                fy = biome.y + math.sin(angle) * dist
                fx = max(FOOD_RADIUS, min(self.width - FOOD_RADIUS, fx))
                fy = max(FOOD_RADIUS, min(self.height - FOOD_RADIUS, fy))
                self.food.append(self._make_food(fx, fy))

        # Check for eating
        self.check_collisions()

    # ── Generation Status ────────────────────────────────────

    def is_generation_over(self) -> bool:
        """Check if the current generation has ended."""
        alive = self.alive_count()
        all_dead = alive == 0
        time_up = self.elapsed_time >= self.settings.generation_time_limit
        threshold = self.settings.extinction_threshold
        below_threshold = (
            threshold > 0 and alive > 0 and alive <= threshold
        )
        return all_dead or time_up or below_threshold

    # ── Freeplay Breeding ─────────────────────────────────────

    def check_breeding(self) -> list[Creature]:
        """
        Check all living creatures for breeding eligibility.

        Uses per-diet settings for breeding thresholds and population caps.
        Eligible creatures produce one mutated offspring nearby.
        Returns list of newly spawned children.
        """
        from pangea.evolution import breed_creature
        from pangea.config import DIET_HERBIVORE, DIET_CARNIVORE, DIET_SCAVENGER

        # Pre-compute per-diet alive counts
        diet_alive = {
            DIET_HERBIVORE: self.alive_count_by_diet(DIET_HERBIVORE),
            DIET_CARNIVORE: self.alive_count_by_diet(DIET_CARNIVORE),
            DIET_SCAVENGER: self.alive_count_by_diet(DIET_SCAVENGER),
        }
        # Track new children per diet for cap enforcement
        diet_new = {DIET_HERBIVORE: 0, DIET_CARNIVORE: 0, DIET_SCAVENGER: 0}

        new_children: list[Creature] = []
        for creature in list(self.creatures):
            diet = creature.dna.diet
            ds = self.settings.diet_settings(diet)

            if not creature.can_breed(ds):
                continue

            # Per-diet hard cap check
            if diet_alive[diet] + diet_new[diet] >= ds.freeplay_hard_cap:
                continue

            child_dna = breed_creature(
                creature,
                mutation_rate=ds.mutation_rate,
                mutation_strength=ds.mutation_strength,
                weight_clamp=ds.weight_clamp,
                trait_mutation_range=ds.trait_mutation_range,
                diet_mutation_rate=ds.diet_mutation_rate,
            )

            # Spawn child near parent
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(5, self.settings.freeplay_child_spawn_radius)
            cx = creature.x + math.cos(angle) * dist
            cy = creature.y + math.sin(angle) * dist
            cx = max(10, min(self.width - 10, cx))
            cy = max(10, min(self.height - 10, cy))

            child = Creature(child_dna, cx, cy, lineage=creature.lineage)
            child.energy = ds.freeplay_child_energy
            child.generation = creature.generation + 1

            # Deduct cost and set cooldown on parent
            creature.energy -= ds.freeplay_breed_energy_cost
            creature.breed_cooldown = ds.freeplay_breed_cooldown
            creature.offspring_count += 1

            # Track new child's diet (may differ from parent if diet mutated)
            child_diet = child_dna.diet
            if child_diet in diet_new:
                diet_new[child_diet] += 1
            new_children.append(child)

        self.creatures.extend(new_children)
        self.total_births += len(new_children)
        return new_children

    def remove_dead_creatures(self, min_dead_age: float = 3.0) -> None:
        """
        Remove creatures that have been dead for a while.

        Keeps recently dead creatures so scavenger/corpse logic can process them.
        """
        kept: list[Creature] = []
        removed = 0
        for c in self.creatures:
            if not c.alive:
                # Keep if died recently (for scavenger detection)
                time_dead = self.elapsed_time - c.age
                if time_dead < min_dead_age:
                    kept.append(c)
                else:
                    removed += 1
            else:
                kept.append(c)
        self.total_deaths += removed
        self.creatures = kept

    def alive_count(self) -> int:
        """Return the number of living creatures."""
        return sum(1 for c in self.creatures if c.alive)

    def alive_count_by_diet(self, diet: int) -> int:
        """Return the number of living creatures with a specific diet."""
        return sum(1 for c in self.creatures if c.alive and c.dna.diet == diet)

    def alive_count_by_lineage(self, lineage: str) -> int:
        """Return the number of living creatures in a specific lineage."""
        return sum(1 for c in self.creatures if c.alive and c.lineage == lineage)

    def food_eaten_by_lineage(self, lineage: str) -> int:
        """Return total food eaten by a specific lineage."""
        return sum(c.food_eaten for c in self.creatures if c.lineage == lineage)
