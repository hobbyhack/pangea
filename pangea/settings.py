"""
Runtime settings — mutable configuration for in-app tuning.
============================================================
Unlike config.py (compile-time defaults), SimSettings can be
changed live through the in-app settings panel. Each simulation
gets a SimSettings instance that flows through World, Evolution, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

    # ── Mutation ─────────────────────────────────────────────
    mutation_rate: float = config.MUTATION_RATE
    mutation_strength: float = config.MUTATION_STRENGTH
    crossover_enabled: bool = False

    # ── World / Food ─────────────────────────────────────────
    food_spawn_rate: float = config.FOOD_SPAWN_RATE
    food_energy: float = config.FOOD_ENERGY
    initial_food_count: int = config.INITIAL_FOOD_COUNT
    world_wrap: bool = config.WORLD_WRAP
    food_decay_time: float = config.FOOD_DECAY_TIME
    food_cluster_size: int = config.FOOD_CLUSTER_SIZE
    season_length: float = config.SEASON_LENGTH
    season_min_rate: float = config.SEASON_MIN_RATE

    # ── Biomes / Terrain ─────────────────────────────────────
    biome_count: int = config.BIOME_COUNT

    # ── Creature Physics ─────────────────────────────────────
    base_energy: float = config.BASE_ENERGY
    energy_cost_per_thrust: float = config.ENERGY_COST_PER_THRUST

    # ── Fitness Weights ────────────────────────────────────────
    fitness_food_weight: float = config.FITNESS_FOOD_WEIGHT
    fitness_time_weight: float = config.FITNESS_TIME_WEIGHT
    fitness_energy_weight: float = config.FITNESS_ENERGY_WEIGHT

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

    def copy(self) -> SimSettings:
        """Create an independent copy of these settings."""
        return SimSettings(
            population_size=self.population_size,
            generation_time_limit=self.generation_time_limit,
            top_performers_count=self.top_performers_count,
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength,
            crossover_enabled=self.crossover_enabled,
            food_spawn_rate=self.food_spawn_rate,
            food_energy=self.food_energy,
            initial_food_count=self.initial_food_count,
            world_wrap=self.world_wrap,
            food_decay_time=self.food_decay_time,
            food_cluster_size=self.food_cluster_size,
            season_length=self.season_length,
            season_min_rate=self.season_min_rate,
            biome_count=self.biome_count,
            base_energy=self.base_energy,
            energy_cost_per_thrust=self.energy_cost_per_thrust,
            fitness_food_weight=self.fitness_food_weight,
            fitness_time_weight=self.fitness_time_weight,
            fitness_energy_weight=self.fitness_energy_weight,
            convergence_max_generations=self.convergence_max_generations,
            day_night_cycle_length=self.day_night_cycle_length,
            night_vision_multiplier=self.night_vision_multiplier,
            hazard_count=self.hazard_count,
            predator_count=self.predator_count,
            predator_speed=self.predator_speed,
            predator_vision=self.predator_vision,
            predator_damage=self.predator_damage,
            predator_radius=self.predator_radius,
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


# Settings organized by category for the UI panel
SETTING_DEFS: list[SettingDef] = [
    # Population
    SettingDef("population_size", "Population Size", 10, 200, 10, ".0f", "Population"),
    SettingDef("generation_time_limit", "Gen Time (sec)", 10, 120, 5, ".0f", "Population"),
    SettingDef("top_performers_count", "Top Survivors", 2, 50, 1, ".0f", "Population"),
    # Mutation
    SettingDef("mutation_rate", "Mutation Rate", 0.01, 1.0, 0.05, ".2f", "Mutation"),
    SettingDef("mutation_strength", "Mutation Strength", 0.05, 2.0, 0.05, ".2f", "Mutation"),
    SettingDef("crossover_enabled", "Crossover", 0, 1, 1, ".0f", "Mutation", widget_type="toggle"),
    # Environment
    SettingDef("food_spawn_rate", "Food Spawn Rate", 0.0, 5.0, 0.1, ".1f", "Environment"),
    SettingDef("food_energy", "Food Energy", 5, 100, 5, ".0f", "Environment"),
    SettingDef("initial_food_count", "Initial Food", 0, 100, 5, ".0f", "Environment"),
    SettingDef("base_energy", "Start Energy", 20, 500, 10, ".0f", "Environment"),
    SettingDef("energy_cost_per_thrust", "Move Cost", 0.01, 0.5, 0.01, ".2f", "Environment"),
    SettingDef("food_decay_time", "Food Decay (sec)", 5, 60, 5, ".0f", "Environment"),
    SettingDef("food_cluster_size", "Cluster Size", 1, 10, 1, ".0f", "Environment"),
    SettingDef("season_length", "Season Length (s)", 10, 300, 10, ".0f", "Environment"),
    SettingDef("season_min_rate", "Season Min Rate", 0.0, 1.0, 0.05, ".2f", "Environment"),
    SettingDef("biome_count", "Biome Count", 0, 10, 1, ".0f", "Environment"),
    SettingDef("world_wrap", "World Wrap", 0, 1, 1, ".0f", "Environment", widget_type="toggle"),
    # Day/Night
    SettingDef("day_night_cycle_length", "Day/Night Cycle (s)", 10, 300, 10, ".0f", "Environment"),
    SettingDef("night_vision_multiplier", "Night Vision", 0.0, 1.0, 0.05, ".2f", "Environment"),
    SettingDef("hazard_count", "Hazard Zones", 0, 10, 1, ".0f", "Environment"),
    SettingDef("predator_count", "Predators", 0, 10, 1, ".0f", "Predators"),
    SettingDef("predator_speed", "Predator Speed", 0.5, 5.0, 0.5, ".1f", "Predators"),
    SettingDef("predator_vision", "Predator Vision", 50, 400, 25, ".0f", "Predators"),
    SettingDef("predator_damage", "Predator Damage", 1.0, 20.0, 1.0, ".1f", "Predators"),
    SettingDef("predator_radius", "Predator Size", 4.0, 20.0, 1.0, ".0f", "Predators"),
    # Fitness
    SettingDef("fitness_food_weight", "Food Weight", 0.0, 50.0, 1.0, ".1f", "Fitness"),
    SettingDef("fitness_time_weight", "Survival Weight", 0.0, 5.0, 0.05, ".2f", "Fitness"),
    SettingDef("fitness_energy_weight", "Energy Weight", 0.0, 5.0, 0.05, ".2f", "Fitness"),
]
