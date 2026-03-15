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

    # ── World / Play Area ─────────────────────────────────────
    world_width: int = config.WORLD_WIDTH
    world_height: int = config.WORLD_HEIGHT

    # ── Food ───────────────────────────────────────────────────
    food_spawn_rate: float = config.FOOD_SPAWN_RATE
    food_energy: float = config.FOOD_ENERGY
    initial_food_count: int = config.INITIAL_FOOD_COUNT
    world_wrap: bool = config.WORLD_WRAP
    food_decay_time: float = config.FOOD_DECAY_TIME
    corpse_decay_time: float = config.CORPSE_DECAY_TIME
    food_cluster_size: int = config.FOOD_CLUSTER_SIZE
    food_respawn_chance: float = config.FOOD_RESPAWN_CHANCE
    food_min: int = config.FOOD_MIN
    food_max: int = config.FOOD_MAX
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
    fitness_offspring_weight: float = config.FITNESS_OFFSPRING_WEIGHT

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

    # ── Freeplay Mode ──────────────────────────────────────────
    freeplay_initial_population: int = config.FREEPLAY_INITIAL_POPULATION
    freeplay_carrying_capacity: int = config.FREEPLAY_CARRYING_CAPACITY
    freeplay_hard_cap: int = config.FREEPLAY_HARD_CAP
    freeplay_breed_min_age: float = config.FREEPLAY_BREED_MIN_AGE
    freeplay_breed_min_food: int = config.FREEPLAY_BREED_MIN_FOOD
    freeplay_breed_energy_threshold: float = config.FREEPLAY_BREED_ENERGY_THRESHOLD
    freeplay_breed_cooldown: float = config.FREEPLAY_BREED_COOLDOWN
    freeplay_breed_energy_cost: float = config.FREEPLAY_BREED_ENERGY_COST
    freeplay_child_energy: float = config.FREEPLAY_CHILD_ENERGY
    freeplay_child_spawn_radius: float = config.FREEPLAY_CHILD_SPAWN_RADIUS
    freeplay_overcapacity_food_penalty: float = config.FREEPLAY_OVERCAPACITY_FOOD_PENALTY

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
        return SimSettings.from_dict(self.to_dict())


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
    SettingDef("food_respawn_chance", "Food Respawn %", 0.0, 1.0, 0.05, ".0%", "Environment",
               tooltip="Chance a new food spawns at a random location when one is eaten"),
    SettingDef("food_min", "Food Min", 0, 200, 5, ".0f", "Environment",
               tooltip="Minimum food items on the map — spawns extra if below this (0 = off)"),
    SettingDef("food_max", "Food Max", 0, 500, 10, ".0f", "Environment",
               tooltip="Maximum food items on the map — stops spawning above this (0 = no limit)"),
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
    SettingDef("fitness_offspring_weight", "Offspring Weight", 0.0, 20.0, 0.5, ".1f", "Fitness",
               tooltip="How much breeding success contributes to fitness (rewards creatures that reproduced)"),
    # Freeplay
    SettingDef("freeplay_initial_population", "Initial Population", 10, 100, 5, ".0f", "Freeplay",
               tooltip="Number of random creatures at the start of freeplay"),
    SettingDef("freeplay_carrying_capacity", "Carrying Capacity", 20, 200, 10, ".0f", "Freeplay",
               tooltip="Soft population cap — food spawns slower above this"),
    SettingDef("freeplay_hard_cap", "Hard Cap", 30, 300, 10, ".0f", "Freeplay",
               tooltip="Absolute max population — no births above this"),
    SettingDef("freeplay_breed_min_age", "Breed Min Age (s)", 1, 30, 1, ".0f", "Freeplay",
               tooltip="Minimum seconds alive before a creature can breed"),
    SettingDef("freeplay_breed_min_food", "Breed Min Food", 1, 20, 1, ".0f", "Freeplay",
               tooltip="Minimum food items eaten before breeding is allowed"),
    SettingDef("freeplay_breed_energy_threshold", "Breed Energy %", 0.1, 1.0, 0.05, ".2f", "Freeplay",
               tooltip="Energy must be above this fraction of base energy to breed"),
    SettingDef("freeplay_breed_cooldown", "Breed Cooldown (s)", 1, 60, 1, ".0f", "Freeplay",
               tooltip="Seconds between successive breeding attempts"),
    SettingDef("freeplay_breed_energy_cost", "Breed Energy Cost", 5, 100, 5, ".0f", "Freeplay",
               tooltip="Energy deducted from parent when offspring is produced"),
    SettingDef("freeplay_child_energy", "Child Start Energy", 10, 200, 10, ".0f", "Freeplay",
               tooltip="Starting energy for newborn creatures"),
]
