"""
Runtime settings — mutable configuration for in-app tuning.
============================================================
Unlike config.py (compile-time defaults), SimSettings can be
changed live through the in-app settings panel. Each simulation
gets a SimSettings instance that flows through World, Evolution, etc.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, field
from pathlib import Path

from pangea import config


@dataclass
class SimSettings:
    """
    All tunable simulation parameters, adjustable at runtime.

    Initialized from config.py defaults. Modify these through the
    in-app settings panel to steer evolution during Isolation Mode.
    """

    # ── Population & Generations ─────────────────────────────
    population_size: int = config.POPULATION_SIZE
    generation_time_limit: float = config.GENERATION_TIME_LIMIT
    top_performers_count: int = config.TOP_PERFORMERS_COUNT
    min_population: int = 0
    extinction_threshold: int = 0
    max_generations: int = 0

    # ── Mutation ─────────────────────────────────────────────
    mutation_rate: float = config.MUTATION_RATE
    mutation_strength: float = config.MUTATION_STRENGTH
    crossover_rate: float = 0.0
    trait_mutation_range: int = config.TRAIT_MUTATION_RANGE
    weight_clamp: float = 0.0

    # ── World / Food ─────────────────────────────────────────
    food_spawn_rate: float = config.FOOD_SPAWN_RATE
    food_energy: float = config.FOOD_ENERGY
    initial_food_count: int = config.INITIAL_FOOD_COUNT
    world_wrap: bool = config.WORLD_WRAP
    food_decay_time: float = config.FOOD_DECAY_TIME
    corpse_decay_time: float = config.CORPSE_DECAY_TIME
    food_cluster_size: int = config.FOOD_CLUSTER_SIZE
    season_length: float = config.SEASON_LENGTH
    season_min_rate: float = config.SEASON_MIN_RATE

    # ── Biomes / Terrain ─────────────────────────────────────
    biome_count: int = config.BIOME_COUNT

    # ── Creature Physics ─────────────────────────────────────
    base_energy: float = config.BASE_ENERGY
    energy_cost_per_thrust: float = config.ENERGY_COST_PER_THRUST
    turn_cost: float = 0.0

    # ── Food Healing ──────────────────────────────────────────
    food_heal: float = 0.0

    # ── Fitness Weights ────────────────────────────────────────
    fitness_food_weight: float = config.FITNESS_FOOD_WEIGHT
    fitness_time_weight: float = config.FITNESS_TIME_WEIGHT
    fitness_energy_weight: float = config.FITNESS_ENERGY_WEIGHT
    territory_fitness_weight: float = 0.0

    # ── Convergence ──────────────────────────────────────────
    convergence_max_generations: int = config.CONVERGENCE_MAX_GENERATIONS

    # ── Day/Night Cycle ───────────────────────────────────────
    day_night_cycle_length: float = config.DAY_NIGHT_CYCLE_LENGTH
    night_vision_multiplier: float = config.NIGHT_VISION_MULTIPLIER

    # ── Hazards ────────────────────────────────────────────────
    hazard_count: int = config.HAZARD_COUNT

    # ── Predators ──────────────────────────────────────────────
    predator_count: int = config.PREDATOR_COUNT
    predator_speed: float = config.PREDATOR_SPEED
    predator_vision: float = config.PREDATOR_VISION
    predator_damage: float = config.PREDATOR_DAMAGE
    predator_radius: float = config.PREDATOR_RADIUS
    predator_stamina: float = 0.0
    predator_respawn_interval: float = 0.0

    def to_dict(self) -> dict:
        """Serialize all settings to a plain dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> SimSettings:
        """Create SimSettings from a dict, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save_to_file(self, filepath: str) -> None:
        """Save settings to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> SimSettings:
        """Load settings from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def list_settings_files(directory: str = "settings") -> list[str]:
        """List all .json settings files in a directory."""
        path = Path(directory)
        if not path.exists():
            return []
        return sorted(f.name for f in path.glob("*.json"))

    def copy(self) -> SimSettings:
        """Create an independent copy of these settings."""
        return SimSettings(
            population_size=self.population_size,
            generation_time_limit=self.generation_time_limit,
            top_performers_count=self.top_performers_count,
            min_population=self.min_population,
            extinction_threshold=self.extinction_threshold,
            max_generations=self.max_generations,
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength,
            crossover_rate=self.crossover_rate,
            trait_mutation_range=self.trait_mutation_range,
            weight_clamp=self.weight_clamp,
            food_spawn_rate=self.food_spawn_rate,
            food_energy=self.food_energy,
            initial_food_count=self.initial_food_count,
            world_wrap=self.world_wrap,
            food_decay_time=self.food_decay_time,
            corpse_decay_time=self.corpse_decay_time,
            food_cluster_size=self.food_cluster_size,
            season_length=self.season_length,
            season_min_rate=self.season_min_rate,
            biome_count=self.biome_count,
            base_energy=self.base_energy,
            energy_cost_per_thrust=self.energy_cost_per_thrust,
            turn_cost=self.turn_cost,
            food_heal=self.food_heal,
            fitness_food_weight=self.fitness_food_weight,
            fitness_time_weight=self.fitness_time_weight,
            fitness_energy_weight=self.fitness_energy_weight,
            territory_fitness_weight=self.territory_fitness_weight,
            convergence_max_generations=self.convergence_max_generations,
            day_night_cycle_length=self.day_night_cycle_length,
            night_vision_multiplier=self.night_vision_multiplier,
            hazard_count=self.hazard_count,
            predator_count=self.predator_count,
            predator_speed=self.predator_speed,
            predator_vision=self.predator_vision,
            predator_damage=self.predator_damage,
            predator_radius=self.predator_radius,
            predator_stamina=self.predator_stamina,
            predator_respawn_interval=self.predator_respawn_interval,
        )


# ── Setting Definitions for the UI ──────────────────────────

@dataclass
class SettingDef:
    """Describes one setting for the settings panel UI."""

    key: str          # attribute name on SimSettings
    label: str        # display name
    min_val: float    # minimum allowed value
    max_val: float    # maximum allowed value
    step: float       # increment per click/drag
    fmt: str = ".1f"  # format string for display
    category: str = ""
    widget_type: str = "slider"  # "slider" or "toggle"
    tooltip: str = "" # hover description


# Settings organized by category for the UI panel
SETTING_DEFS: list[SettingDef] = [
    # Population
    SettingDef("population_size", "Population Size", 10, 200, 10, ".0f", "Population",
               tooltip="Number of creatures spawned each generation"),
    SettingDef("generation_time_limit", "Gen Time (sec)", 10, 120, 5, ".0f", "Population",
               tooltip="Max seconds a generation runs before selection occurs"),
    SettingDef("top_performers_count", "Top Survivors", 2, 50, 1, ".0f", "Population",
               tooltip="How many top-fitness creatures survive to breed the next generation"),
    SettingDef("min_population", "Min Parents", 0, 50, 1, ".0f", "Population",
               tooltip="Minimum parent count; pads survivors with random DNA if too few qualify (0 = off)"),
    SettingDef("extinction_threshold", "Extinction Threshold", 0, 50, 1, ".0f", "Population",
               tooltip="If fewer than this many survive, the run ends as an extinction event (0 = off)"),
    SettingDef("max_generations", "Max Generations", 0, 500, 10, ".0f", "Population",
               tooltip="Stop the simulation after this many generations (0 = unlimited)"),
    # Mutation
    SettingDef("mutation_rate", "Mutation Rate", 0.01, 1.0, 0.05, ".2f", "Mutation",
               tooltip="Probability each neural-network weight is mutated per offspring"),
    SettingDef("mutation_strength", "Mutation Strength", 0.05, 2.0, 0.05, ".2f", "Mutation",
               tooltip="Standard deviation of Gaussian noise added to mutated weights"),
    SettingDef("crossover_rate", "Crossover Rate", 0.0, 1.0, 0.05, ".2f", "Mutation",
               tooltip="Chance two parents swap gene segments when breeding (0 = cloning only)"),
    SettingDef("trait_mutation_range", "Trait Mut. Range", 0, 20, 1, ".0f", "Mutation",
               tooltip="Max points a trait (speed/size/vision/efficiency) can shift per generation"),
    SettingDef("weight_clamp", "Weight Clamp", 0.0, 10.0, 0.5, ".1f", "Mutation",
               tooltip="Clamp NN weights to [-value, +value] after mutation (0 = no clamping)"),
    # Environment
    SettingDef("food_spawn_rate", "Food Spawn Rate", 0.0, 5.0, 0.1, ".1f", "Environment",
               tooltip="New food items spawned per second during a generation"),
    SettingDef("food_energy", "Food Energy", 5, 100, 5, ".0f", "Environment",
               tooltip="Energy a creature gains from eating one food item"),
    SettingDef("initial_food_count", "Initial Food", 0, 100, 5, ".0f", "Environment",
               tooltip="Food items placed on the map at the start of each generation"),
    SettingDef("base_energy", "Start Energy", 20, 500, 10, ".0f", "Environment",
               tooltip="Energy each creature starts with; reaching 0 means death"),
    SettingDef("energy_cost_per_thrust", "Move Cost", 0.01, 0.5, 0.01, ".2f", "Environment",
               tooltip="Energy drained per unit of forward thrust (scaled by creature size)"),
    SettingDef("turn_cost", "Turn Cost", 0.0, 0.5, 0.01, ".2f", "Environment",
               tooltip="Extra energy cost for turning (0 = turning is free)"),
    SettingDef("food_decay_time", "Food Decay (sec)", 5, 60, 5, ".0f", "Environment",
               tooltip="Seconds before an uneaten food item disappears"),
    SettingDef("corpse_decay_time", "Corpse Decay (sec)", 3, 60, 1, ".0f", "Environment",
               tooltip="Seconds before a dead creature's corpse disappears (scavenger food)"),
    SettingDef("food_cluster_size", "Cluster Size", 1, 10, 1, ".0f", "Environment",
               tooltip="Food spawns in clusters of this many items"),
    SettingDef("food_heal", "Food Heal (sec)", 0.0, 10.0, 0.5, ".1f", "Environment",
               tooltip="Seconds of predator-damage immunity after eating (0 = none)"),
    SettingDef("season_length", "Season Length (s)", 10, 300, 10, ".0f", "Environment",
               tooltip="Duration of one full season cycle (food rate oscillates over this period)"),
    SettingDef("season_min_rate", "Season Min Rate", 0.0, 1.0, 0.05, ".2f", "Environment",
               tooltip="Food spawn rate multiplier at the trough of the season cycle"),
    SettingDef("biome_count", "Biome Count", 0, 10, 1, ".0f", "Environment",
               tooltip="Number of distinct terrain biomes on the map (0 = uniform terrain)"),
    SettingDef("world_wrap", "World Wrap", 0, 1, 1, ".0f", "Environment", widget_type="toggle",
               tooltip="Creatures exiting one edge reappear on the opposite side"),
    # Day/Night
    SettingDef("day_night_cycle_length", "Day/Night Cycle (s)", 10, 300, 10, ".0f", "Environment",
               tooltip="Length of a full day/night cycle in seconds"),
    SettingDef("night_vision_multiplier", "Night Vision", 0.0, 1.0, 0.05, ".2f", "Environment",
               tooltip="Vision range multiplier during nighttime (1.0 = no reduction)"),
    SettingDef("hazard_count", "Hazard Zones", 0, 10, 1, ".0f", "Environment",
               tooltip="Number of damaging hazard zones placed on the map"),
    SettingDef("predator_count", "Predators", 0, 10, 1, ".0f", "Predators",
               tooltip="Number of AI-controlled predators that hunt creatures"),
    SettingDef("predator_speed", "Predator Speed", 0.5, 5.0, 0.5, ".1f", "Predators",
               tooltip="Movement speed of predators (higher = harder to outrun)"),
    SettingDef("predator_vision", "Predator Vision", 50, 400, 25, ".0f", "Predators",
               tooltip="How far predators can detect creatures (in pixels)"),
    SettingDef("predator_damage", "Predator Damage", 1.0, 20.0, 1.0, ".1f", "Predators",
               tooltip="Energy drained from a creature per predator hit"),
    SettingDef("predator_radius", "Predator Size", 4.0, 20.0, 1.0, ".0f", "Predators",
               tooltip="Collision radius of predators (larger = harder to dodge)"),
    SettingDef("predator_stamina", "Predator Stamina", 0.0, 30.0, 1.0, ".0f", "Predators",
               tooltip="Seconds a predator can chase before resting (0 = infinite stamina)"),
    SettingDef("predator_respawn_interval", "Predator Respawn", 0.0, 60.0, 5.0, ".0f", "Predators",
               tooltip="Seconds before a killed predator respawns (0 = predators are immortal)"),
    # Fitness
    SettingDef("fitness_food_weight", "Food Weight", 0.0, 50.0, 1.0, ".1f", "Fitness",
               tooltip="How much food eaten contributes to a creature's fitness score"),
    SettingDef("fitness_time_weight", "Survival Weight", 0.0, 5.0, 0.05, ".2f", "Fitness",
               tooltip="How much time alive contributes to fitness (rewards longevity)"),
    SettingDef("fitness_energy_weight", "Energy Weight", 0.0, 5.0, 0.05, ".2f", "Fitness",
               tooltip="How much remaining energy at death contributes to fitness"),
    SettingDef("territory_fitness_weight", "Territory Weight", 0.0, 5.0, 0.1, ".1f", "Fitness",
               tooltip="How much area explored contributes to fitness (rewards exploration)"),
]
