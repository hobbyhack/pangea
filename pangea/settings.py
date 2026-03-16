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
from pangea.species import (
    Species,
    SpeciesRegistry,
    SpeciesSettings,
    default_registry,
    species_id_from_legacy_diet,
    EXTINCTION_RESPAWN_BEST,
    EXTINCTION_RESPAWN_RANDOM,
    EXTINCTION_PERMANENT,
    EXTINCTION_MODES,
)



@dataclass
class SimSettings:
    """
    All tunable simulation parameters, adjustable at runtime.

    Initialized from config.py defaults. Modify these through the
    in-app settings panel.
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

    # ── World ─────────────────────────────────────────────────
    world_width: int = config.WORLD_WIDTH
    world_height: int = config.WORLD_HEIGHT
    world_wrap: bool = config.WORLD_WRAP
    biomes_enabled: bool = config.BIOMES_ENABLED
    biome_count: int = config.BIOME_COUNT

    # ── Food ───────────────────────────────────────────────────
    initial_food_count: int = config.INITIAL_FOOD_COUNT
    food_spawn_rate: float = config.FOOD_SPAWN_RATE
    food_energy: float = config.FOOD_ENERGY
    food_decay_time: float = config.FOOD_DECAY_TIME
    food_cluster_size: int = config.FOOD_CLUSTER_SIZE
    food_respawn_chance: float = config.FOOD_RESPAWN_CHANCE
    food_min: int = config.FOOD_MIN
    food_max: int = config.FOOD_MAX
    corpse_decay_time: float = config.CORPSE_DECAY_TIME
    season_enabled: bool = config.SEASON_ENABLED
    season_length: float = config.SEASON_LENGTH
    season_min_rate: float = config.SEASON_MIN_RATE

    # ── Creatures ─────────────────────────────────────────────
    base_energy: float = config.BASE_ENERGY
    energy_cost_per_thrust: float = config.ENERGY_COST_PER_THRUST
    turn_cost: float = 0.0
    food_heal: float = 0.0

    # ── Day/Night Cycle ───────────────────────────────────────
    day_night_enabled: bool = config.DAY_NIGHT_ENABLED
    day_night_cycle_length: float = config.DAY_NIGHT_CYCLE_LENGTH
    night_vision_multiplier: float = config.NIGHT_VISION_MULTIPLIER

    # ── Threats (Hazards) ──────────────────────────────────────
    hazard_count: int = config.HAZARD_COUNT

    # ── Fitness Weights ────────────────────────────────────────
    fitness_food_weight: float = config.FITNESS_FOOD_WEIGHT
    fitness_time_weight: float = config.FITNESS_TIME_WEIGHT
    fitness_energy_weight: float = config.FITNESS_ENERGY_WEIGHT
    territory_fitness_weight: float = 0.0
    fitness_offspring_weight: float = config.FITNESS_OFFSPRING_WEIGHT
    fitness_distance_weight: float = config.FITNESS_DISTANCE_WEIGHT

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

    # ── Species Registry (replaces fixed herbivore/carnivore/scavenger) ──
    species_registry: SpeciesRegistry = field(default_factory=default_registry)

    def total_freeplay_carrying_capacity(self) -> int:
        """Sum of enabled per-species carrying capacities (for food overcapacity calc)."""
        return sum(sp.settings.freeplay_carrying_capacity for sp in self.species_registry if sp.enabled)

    def to_dict(self) -> dict:
        """Serialize all settings to a plain dict."""
        d = {}
        for f in fields(self):
            if f.name == "species_registry":
                continue
            d[f.name] = getattr(self, f.name)
        d["species"] = self.species_registry.to_list()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> SimSettings:
        """Create SimSettings from a dict, ignoring unknown keys.

        Backward compatible: old saves with diet_herbivore/diet_carnivore/
        diet_scavenger keys are migrated to species_registry.
        """
        valid = {f.name for f in fields(cls)} - {"species_registry"}
        kwargs = {k: v for k, v in data.items() if k in valid}

        if "species" in data:
            # New format: species list
            kwargs["species_registry"] = SpeciesRegistry.from_list(data["species"])
        elif "diet_herbivore" in data or "diet_carnivore" in data or "diet_scavenger" in data:
            # Legacy format: migrate from fixed diet settings
            registry = default_registry()
            if "diet_herbivore" in data:
                sp = registry.get("herbivore")
                if sp:
                    sp.settings = SpeciesSettings.from_dict(data["diet_herbivore"])
            if "diet_carnivore" in data:
                sp = registry.get("carnivore")
                if sp:
                    sp.settings = SpeciesSettings.from_dict(data["diet_carnivore"])
            if "diet_scavenger" in data:
                sp = registry.get("scavenger")
                if sp:
                    sp.settings = SpeciesSettings.from_dict(data["diet_scavenger"])
            kwargs["species_registry"] = registry

        return cls(**kwargs)

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

    key: str          # attribute name on SimSettings (or SpeciesSettings attr for per-species)
    label: str        # display name
    min_val: float    # minimum allowed value
    max_val: float    # maximum allowed value
    step: float       # increment per click/drag
    fmt: str = ".1f"  # format string for display
    category: str = ""
    widget_type: str = "slider"  # "slider", "toggle", or "select"
    tooltip: str = "" # hover description
    species_id: str | None = None  # None=global, string=per-species


# Settings organized by category for the UI panel
SETTING_DEFS: list[SettingDef] = [
    # ── World ────────────────────────────────────────────────
    SettingDef("world_wrap", "World Wrap", 0, 1, 1, ".0f", "World", widget_type="toggle",
               tooltip="Creatures exiting one edge reappear on the opposite side"),
    SettingDef("biomes_enabled", "Biomes", 0, 1, 1, ".0f", "World", widget_type="toggle",
               tooltip="Enable terrain biome regions (water, desert, forest, etc.)"),
    SettingDef("biome_count", "Biome Count", 0, 10, 1, ".0f", "World",
               tooltip="Number of distinct terrain biomes on the map (0 = uniform terrain)"),
    # ── Food ─────────────────────────────────────────────────
    SettingDef("initial_food_count", "Initial Food", 0, 100, 5, ".0f", "Food",
               tooltip="Food items placed on the map at the start of each generation"),
    SettingDef("food_spawn_rate", "Food Spawn Rate", 0.0, 5.0, 0.1, ".1f", "Food",
               tooltip="New food items spawned per second during a generation"),
    SettingDef("food_energy", "Food Energy", 5, 100, 5, ".0f", "Food",
               tooltip="Energy a creature gains from eating one food item"),
    SettingDef("food_decay_time", "Food Decay (sec)", 5, 60, 5, ".0f", "Food",
               tooltip="Seconds before an uneaten food item disappears"),
    SettingDef("food_cluster_size", "Cluster Size", 1, 10, 1, ".0f", "Food",
               tooltip="Food spawns in clusters of this many items"),
    SettingDef("food_respawn_chance", "Food Respawn %", 0.0, 1.0, 0.05, ".0%", "Food",
               tooltip="Chance a new food spawns at a random location when one is eaten"),
    SettingDef("food_min", "Food Min", 0, 200, 5, ".0f", "Food",
               tooltip="Minimum food items on the map — spawns extra if below this (0 = off)"),
    SettingDef("food_max", "Food Max", 0, 500, 10, ".0f", "Food",
               tooltip="Maximum food items on the map — stops spawning above this (0 = no limit)"),
    SettingDef("corpse_decay_time", "Corpse Decay (sec)", 3, 60, 1, ".0f", "Food",
               tooltip="Seconds before a dead creature's corpse disappears (scavenger food)"),
    # ── Seasons ─────────────────────────────────────────────
    SettingDef("season_enabled", "Seasons", 0, 1, 1, ".0f", "Seasons", widget_type="toggle",
               tooltip="Enable seasonal food spawn oscillation"),
    SettingDef("season_length", "Season Length (s)", 10, 300, 10, ".0f", "Seasons",
               tooltip="Duration of one full season cycle (food rate oscillates over this period)"),
    SettingDef("season_min_rate", "Season Min Rate", 0.0, 1.0, 0.05, ".2f", "Seasons",
               tooltip="Food spawn rate multiplier at the trough of the season cycle"),
    # ── Day/Night ────────────────────────────────────────────
    SettingDef("day_night_enabled", "Day/Night Cycle", 0, 1, 1, ".0f", "Day/Night", widget_type="toggle",
               tooltip="Enable the day/night cycle (darkness overlay and vision reduction)"),
    SettingDef("day_night_cycle_length", "Day/Night Cycle (s)", 10, 300, 10, ".0f", "Day/Night",
               tooltip="Length of a full day/night cycle in seconds"),
    # ── Threats ──────────────────────────────────────────────
    SettingDef("hazard_count", "Hazard Zones", 0, 10, 1, ".0f", "Threats",
               tooltip="Number of damaging hazard zones placed on the map"),
]


def _species_setting_defs(species_id: str, category: str) -> list[SettingDef]:
    """Generate per-species SettingDef entries."""
    return [
        # Population
        SettingDef("freeplay_initial_population", "Initial Population", 1, 100, 1, ".0f",
                   category, tooltip="Starting population for this species", species_id=species_id),
        SettingDef("freeplay_carrying_capacity", "Carrying Capacity", 5, 200, 5, ".0f",
                   category, tooltip="Soft cap — food spawns slower above this", species_id=species_id),
        SettingDef("freeplay_hard_cap", "Hard Cap", 10, 300, 5, ".0f",
                   category, tooltip="Absolute max — no births above this", species_id=species_id),
        # Mutation
        SettingDef("mutation_rate", "Mutation Rate", 0.01, 1.0, 0.05, ".2f",
                   category, tooltip="NN weight mutation probability per offspring", species_id=species_id),
        SettingDef("mutation_strength", "Mutation Strength", 0.05, 2.0, 0.05, ".2f",
                   category, tooltip="Gaussian noise std dev for weight mutation", species_id=species_id),
        SettingDef("crossover_rate", "Crossover Rate", 0.0, 1.0, 0.05, ".2f",
                   category, tooltip="Chance two parents swap genes (0 = cloning)", species_id=species_id),
        SettingDef("trait_mutation_range", "Trait Mut. Range", 0, 20, 1, ".0f",
                   category, tooltip="Max trait point shift per generation", species_id=species_id),
        SettingDef("weight_clamp", "Weight Clamp", 0.0, 10.0, 0.5, ".1f",
                   category, tooltip="Clamp NN weights after mutation (0 = off)", species_id=species_id),
        SettingDef("top_performers_count", "Top Survivors", 1, 50, 1, ".0f",
                   category, tooltip="Top-fitness creatures used for extinction respawn", species_id=species_id),
        SettingDef("min_population", "Min Parents", 0, 50, 1, ".0f",
                   category, tooltip="Pad parents with random DNA if too few (0 = off)", species_id=species_id),
        # Breeding
        SettingDef("freeplay_breed_min_age", "Breed Min Age (s)", 1, 30, 1, ".0f",
                   category, tooltip="Min seconds alive before breeding", species_id=species_id),
        SettingDef("freeplay_breed_min_food", "Breed Min Food", 1, 20, 1, ".0f",
                   category, tooltip="Min food eaten before breeding", species_id=species_id),
        SettingDef("freeplay_breed_energy_threshold", "Breed Energy %", 0.1, 1.0, 0.05, ".2f",
                   category, tooltip="Energy fraction required to breed", species_id=species_id),
        SettingDef("freeplay_breed_cooldown", "Breed Cooldown (s)", 1, 60, 1, ".0f",
                   category, tooltip="Seconds between breedings", species_id=species_id),
        SettingDef("freeplay_breed_energy_cost", "Breed Energy Cost", 5, 100, 5, ".0f",
                   category, tooltip="Energy deducted from parent", species_id=species_id),
        SettingDef("freeplay_child_energy", "Child Start Energy", 10, 200, 10, ".0f",
                   category, tooltip="Starting energy for offspring", species_id=species_id),
        # Creatures
        SettingDef("base_energy", "Start Energy", 20, 500, 10, ".0f",
                   category, tooltip="Energy each creature starts with", species_id=species_id),
        SettingDef("energy_cost_per_thrust", "Move Cost", 0.01, 0.5, 0.01, ".2f",
                   category, tooltip="Energy drained per unit of thrust", species_id=species_id),
        SettingDef("turn_cost", "Turn Cost", 0.0, 0.5, 0.01, ".2f",
                   category, tooltip="Extra energy cost for turning (0 = free)", species_id=species_id),
        SettingDef("food_heal", "Food Heal (sec)", 0.0, 10.0, 0.5, ".1f",
                   category, tooltip="Lifespan seconds restored per food eaten", species_id=species_id),
        # Night Vision
        SettingDef("night_vision_multiplier", "Night Vision", 0.0, 1.0, 0.05, ".2f",
                   category, tooltip="Vision multiplier at night (1.0 = no reduction)", species_id=species_id),
        # Fitness
        SettingDef("fitness_food_weight", "Food Weight", 0.0, 50.0, 1.0, ".1f",
                   category, tooltip="Food eaten contribution to fitness", species_id=species_id),
        SettingDef("fitness_time_weight", "Survival Weight", 0.0, 5.0, 0.05, ".2f",
                   category, tooltip="Time alive contribution to fitness", species_id=species_id),
        SettingDef("fitness_energy_weight", "Energy Weight", 0.0, 5.0, 0.05, ".2f",
                   category, tooltip="Remaining energy contribution to fitness", species_id=species_id),
        SettingDef("territory_fitness_weight", "Territory Weight", 0.0, 5.0, 0.1, ".1f",
                   category, tooltip="Area explored contribution to fitness", species_id=species_id),
        SettingDef("fitness_offspring_weight", "Offspring Weight", 0.0, 20.0, 0.5, ".1f",
                   category, tooltip="Breeding success contribution to fitness", species_id=species_id),
        SettingDef("fitness_distance_weight", "Distance Weight", 0.0, 2.0, 0.05, ".2f",
                   category, tooltip="Distance traveled contribution to fitness (rewards movement)", species_id=species_id),
        # Extinction
        SettingDef("extinction_mode", "Extinction Mode", 0, 2, 1, ".0f",
                   category, widget_type="select",
                   tooltip="0=Respawn Best, 1=Respawn Random, 2=Permanent", species_id=species_id),
    ]


def build_species_setting_defs(registry: SpeciesRegistry) -> list[SettingDef]:
    """Build per-species setting definitions dynamically from the registry."""
    result: list[SettingDef] = []
    for sp in registry.all():
        result.extend(_species_setting_defs(sp.id, sp.name))
    return result
