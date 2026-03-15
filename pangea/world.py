"""
World — the simulation environment.
============================================================
Manages the 2D arena, food spawning/decay, creature-food collisions,
boundary enforcement, biome regions, hazard zones,
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
    BIOME_SPEED_MULTIPLIERS,
    CARNIVORE_ATTACK_RANGE,
    FOOD_ENERGY,
    FOOD_RADIUS,
    HAZARD_DAMAGE,
    HAZARD_MAX_RADIUS,
    HAZARD_MIN_RADIUS,
    CORPSE_ENERGY,
    CORPSE_RADIUS,
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
    is_corpse: bool = False  # True = corpse food (diet flags determine who eats it)
    species_id: str = ""     # species of the creature that died (for own/other corpse logic)


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

            sp = creature.species
            for i, food in enumerate(self.food):
                # Corpse eating gated by species diet flags
                if food.is_corpse:
                    if sp is None:
                        continue  # no species = can't eat corpses
                    same_species = food.species_id == creature.dna.species_id
                    if same_species and not sp.can_eat_own_corpse:
                        continue
                    if not same_species and not sp.can_eat_other_corpse:
                        continue
                else:
                    # Plant food gated by can_eat_plants
                    if sp is not None and not sp.can_eat_plants:
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
                    sp_heal = (sp.settings.food_heal if sp else
                               self.settings.food_heal)
                    creature.eat(food.energy, sp_heal)
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

    # ── Creature Combat ───────────────────────────────────────

    def _check_creature_attacks(self, dt: float) -> None:
        """Species with attack flags damage nearby creatures and steal energy."""
        for attacker in self.creatures:
            if not attacker.alive:
                continue
            sp = attacker.species
            if sp is None or not sp.can_attack:
                continue
            attack_range = attacker.dna.effective_radius * CARNIVORE_ATTACK_RANGE
            for victim in self.creatures:
                if victim is attacker or not victim.alive:
                    continue
                # Check own-vs-other species attack permission
                same_species = attacker.dna.species_id == victim.dna.species_id
                if same_species and not sp.can_attack_own_species:
                    continue
                if not same_species and not sp.can_attack_other_species:
                    continue
                dx = attacker.x - victim.x
                dy = attacker.y - victim.y
                dist = math.sqrt(dx * dx + dy * dy)
                contact = attack_range + victim.dna.effective_radius
                if dist < contact:
                    damage = sp.attack_damage * dt * 60
                    victim.energy -= damage
                    victim.under_attack = 1.0
                    attacker.gain_energy(damage * sp.energy_steal_fraction)

    # ── Scavenger Death Detection ────────────────────────────

    def _reward_scavengers(self, newly_dead: list[Creature]) -> None:
        """Give energy to scavengers near recently dead creatures and spawn corpses."""
        for dead in newly_dead:
            dead_species_id = dead.dna.species_id
            # Instant proximity bonus for species that can scavenge
            for creature in self.creatures:
                if not creature.alive:
                    continue
                sp = creature.species
                if sp is None or not sp.can_scavenge:
                    continue
                # Check own-vs-other corpse permission
                same_species = creature.dna.species_id == dead_species_id
                if same_species and not sp.can_eat_own_corpse:
                    continue
                if not same_species and not sp.can_eat_other_corpse:
                    continue
                dx = creature.x - dead.x
                dy = creature.y - dead.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < sp.scavenge_death_radius:
                    creature.gain_energy(sp.scavenge_death_energy)
            # Spawn a corpse food item so scavengers can find it later
            self.food.append(Food(
                x=dead.x,
                y=dead.y,
                energy=CORPSE_ENERGY,
                radius=CORPSE_RADIUS,
                lifetime=self.settings.corpse_decay_time,
                is_corpse=True,
                species_id=dead_species_id,
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

        # Daylight factor for day/night vision
        daylight = self.daylight_factor

        # Reset under_attack flag for all creatures
        for creature in self.creatures:
            creature.under_attack = 0.0

        # Update each creature
        for creature in self.creatures:
            if not creature.alive:
                continue

            # Per-species night vision multiplier
            sp = creature.species
            night_mult = (sp.settings.night_vision_multiplier if sp
                          else self.settings.night_vision_multiplier)
            vision_multiplier = night_mult + (1 - night_mult) * daylight

            # Compute biome info at creature's position (single lookup)
            biome = self.get_biome_at(creature.x, creature.y)
            if biome is not None:
                speed_mult = biome.speed_multiplier
                biome_danger = BIOME_ENERGY_DRAIN.get(biome.biome_type, 0.0)
            else:
                speed_mult = 1.0
                biome_danger = 0.0

            # Sense the environment (including threats and biome)
            inputs = creature.sense(
                self.food, self.width, self.height, wrap,
                vision_multiplier=vision_multiplier,
                creatures=self.creatures,
                biome_speed=speed_mult,
                biome_danger=biome_danger,
            )

            # Think and act
            creature.think_and_act(inputs)

            # Update physics (with biome speed multiplier)
            creature.update(dt, speed_multiplier=speed_mult)

            # Turn cost — energy penalty proportional to turning (per-species)
            tc = sp.settings.turn_cost if sp else self.settings.turn_cost
            if tc > 0:
                creature.energy -= (
                    creature.last_turn * tc * dt * 60
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

        # Creature attacks (species with attack flags)
        self._check_creature_attacks(dt)

        # Reward scavengers near deaths
        # Collect newly dead after all damage (carnivore + hazard)
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

        Uses per-species settings for breeding thresholds and population caps.
        Eligible creatures produce one mutated offspring nearby.
        Returns list of newly spawned children.
        """
        from pangea.evolution import breed_creature

        registry = self.settings.species_registry

        # Pre-compute per-species alive counts
        species_alive: dict[str, int] = {}
        for sp in registry.all():
            species_alive[sp.id] = self.alive_count_by_species(sp.id)
        # Track new children per species for cap enforcement
        species_new: dict[str, int] = {sp.id: 0 for sp in registry.all()}

        new_children: list[Creature] = []
        for creature in list(self.creatures):
            sid = creature.dna.species_id
            sp = registry.get(sid)
            if sp is None:
                continue
            if not sp.enabled:
                continue
            ss = sp.settings

            if not creature.can_breed(ss):
                continue

            # Per-species hard cap check
            alive_count = species_alive.get(sid, 0)
            new_count = species_new.get(sid, 0)
            if alive_count + new_count >= ss.freeplay_hard_cap:
                continue

            child_dna = breed_creature(
                creature,
                mutation_rate=ss.mutation_rate,
                mutation_strength=ss.mutation_strength,
                weight_clamp=ss.weight_clamp,
                trait_mutation_range=ss.trait_mutation_range,
            )

            # Spawn child near parent
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(5, self.settings.freeplay_child_spawn_radius)
            cx = creature.x + math.cos(angle) * dist
            cy = creature.y + math.sin(angle) * dist
            cx = max(10, min(self.width - 10, cx))
            cy = max(10, min(self.height - 10, cy))

            child = Creature(child_dna, cx, cy,
                             species=creature.species)
            child.energy = ss.freeplay_child_energy
            child.generation = creature.generation + 1

            # Deduct cost and set cooldown on parent
            creature.energy -= ss.freeplay_breed_energy_cost
            creature.breed_cooldown = ss.freeplay_breed_cooldown
            creature.offspring_count += 1

            # Track new child's species
            if sid in species_new:
                species_new[sid] += 1
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

    def alive_count_by_species(self, species_id: str) -> int:
        """Return the number of living creatures of a specific species."""
        return sum(1 for c in self.creatures if c.alive and c.dna.species_id == species_id)

