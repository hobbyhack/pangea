"""
Species — first-class species model for creatures.
============================================================
Each species defines a named group of creatures sharing diet behavior,
color, and population/breeding settings. Replaces the old hardcoded
diet integer system (DIET_HERBIVORE=0, DIET_CARNIVORE=1, DIET_SCAVENGER=2).

Species instances are registered in a SpeciesRegistry which lives
on SimSettings and flows through the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, field
from typing import Iterator

from pangea import config


# ── Per-Species Settings ─────────────────────────────────────

EXTINCTION_RESPAWN_BEST = "respawn_best"
EXTINCTION_RESPAWN_RANDOM = "respawn_random"
EXTINCTION_PERMANENT = "permanent"
EXTINCTION_MODES = [EXTINCTION_RESPAWN_BEST, EXTINCTION_RESPAWN_RANDOM, EXTINCTION_PERMANENT]


@dataclass
class SpeciesSettings:
    """Per-species settings for population, breeding, mutation, and extinction."""

    # ── Population ────────────────────────────────────────────
    freeplay_initial_population: int = 13
    freeplay_carrying_capacity: int = 27
    freeplay_hard_cap: int = 40

    # ── Mutation / Evolution ──────────────────────────────────
    mutation_rate: float = config.MUTATION_RATE
    mutation_strength: float = config.MUTATION_STRENGTH
    crossover_rate: float = 0.0
    trait_mutation_range: int = config.TRAIT_MUTATION_RANGE
    weight_clamp: float = 0.0
    top_performers_count: int = config.TOP_PERFORMERS_COUNT
    min_population: int = 0

    # ── Breeding ──────────────────────────────────────────────
    freeplay_breed_min_age: float = config.FREEPLAY_BREED_MIN_AGE
    freeplay_breed_min_food: int = config.FREEPLAY_BREED_MIN_FOOD
    freeplay_breed_energy_threshold: float = config.FREEPLAY_BREED_ENERGY_THRESHOLD
    freeplay_breed_cooldown: float = config.FREEPLAY_BREED_COOLDOWN
    freeplay_breed_energy_cost: float = config.FREEPLAY_BREED_ENERGY_COST
    freeplay_child_energy: float = config.FREEPLAY_CHILD_ENERGY

    # ── Creatures ──────────────────────────────────────────────
    base_energy: float = config.BASE_ENERGY
    energy_cost_per_thrust: float = config.ENERGY_COST_PER_THRUST
    turn_cost: float = 0.0
    food_heal: float = 0.0

    # ── Night Vision ───────────────────────────────────────────
    night_vision_multiplier: float = config.NIGHT_VISION_MULTIPLIER

    # ── Fitness Weights ────────────────────────────────────────
    fitness_food_weight: float = config.FITNESS_FOOD_WEIGHT
    fitness_time_weight: float = config.FITNESS_TIME_WEIGHT
    fitness_energy_weight: float = config.FITNESS_ENERGY_WEIGHT
    territory_fitness_weight: float = 0.0
    fitness_offspring_weight: float = config.FITNESS_OFFSPRING_WEIGHT

    # ── Extinction ────────────────────────────────────────────
    extinction_mode: str = EXTINCTION_RESPAWN_BEST

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> SpeciesSettings:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def copy(self) -> SpeciesSettings:
        return SpeciesSettings.from_dict(self.to_dict())


# ── Species ──────────────────────────────────────────────────


@dataclass
class Species:
    """
    A named group of creatures sharing diet behavior and settings.

    Diet behavior is defined by boolean flags and numeric tuning
    values, replacing the old hardcoded diet integer system.
    """

    id: str                              # unique slug ("herbivore", "carnivore", or user-defined)
    name: str                            # display name
    color: tuple[int, int, int]          # RGB display color

    # ── Diet behavior flags ───────────────────────────────────
    can_eat_plants: bool = True          # can eat normal food items
    plant_food_multiplier: float = 1.0   # energy multiplier when eating plants
    can_attack_other_species: bool = False  # can attack creatures of OTHER species
    can_attack_own_species: bool = False    # can attack creatures of SAME species
    can_eat_other_corpse: bool = False      # can eat corpses of OTHER species
    can_eat_own_corpse: bool = False        # can eat corpses of SAME species

    # ── Combat tuning ─────────────────────────────────────────
    attack_damage: float = config.CARNIVORE_ATTACK_DAMAGE
    energy_steal_fraction: float = config.CARNIVORE_ENERGY_STEAL

    # ── Scavenge tuning ───────────────────────────────────────
    scavenge_death_radius: float = config.SCAVENGER_DEATH_RADIUS
    scavenge_death_energy: float = config.SCAVENGER_DEATH_ENERGY

    # ── Per-species settings ──────────────────────────────────
    settings: SpeciesSettings = field(default_factory=SpeciesSettings)

    # ── Active state ─────────────────────────────────────────
    enabled: bool = True  # False = paused (no breeding, no extinction respawn)

    @property
    def can_attack(self) -> bool:
        """True if this species can attack any creature."""
        return self.can_attack_other_species or self.can_attack_own_species

    @property
    def can_eat_corpses(self) -> bool:
        """True if this species can eat any type of corpse."""
        return self.can_eat_other_corpse or self.can_eat_own_corpse

    @property
    def can_scavenge(self) -> bool:
        """True if this species benefits from nearby deaths."""
        return self.can_eat_corpses and self.scavenge_death_energy > 0

    def to_dict(self) -> dict:
        """Serialize species definition to a JSON-compatible dict."""
        return {
            "id": self.id,
            "name": self.name,
            "color": list(self.color),
            "can_eat_plants": self.can_eat_plants,
            "plant_food_multiplier": self.plant_food_multiplier,
            "can_attack_other_species": self.can_attack_other_species,
            "can_attack_own_species": self.can_attack_own_species,
            "can_eat_other_corpse": self.can_eat_other_corpse,
            "can_eat_own_corpse": self.can_eat_own_corpse,
            "attack_damage": self.attack_damage,
            "energy_steal_fraction": self.energy_steal_fraction,
            "scavenge_death_radius": self.scavenge_death_radius,
            "scavenge_death_energy": self.scavenge_death_energy,
            "settings": self.settings.to_dict(),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Species:
        """Reconstruct a Species from a saved dict."""
        settings_data = data.get("settings", {})
        return cls(
            id=data["id"],
            name=data["name"],
            color=tuple(data["color"]),
            can_eat_plants=data.get("can_eat_plants", True),
            plant_food_multiplier=data.get("plant_food_multiplier", 1.0),
            can_attack_other_species=data.get("can_attack_other_species", False),
            can_attack_own_species=data.get("can_attack_own_species", False),
            can_eat_other_corpse=data.get("can_eat_other_corpse", False),
            can_eat_own_corpse=data.get("can_eat_own_corpse", False),
            attack_damage=data.get("attack_damage", config.CARNIVORE_ATTACK_DAMAGE),
            energy_steal_fraction=data.get("energy_steal_fraction", config.CARNIVORE_ENERGY_STEAL),
            scavenge_death_radius=data.get("scavenge_death_radius", config.SCAVENGER_DEATH_RADIUS),
            scavenge_death_energy=data.get("scavenge_death_energy", config.SCAVENGER_DEATH_ENERGY),
            settings=SpeciesSettings.from_dict(settings_data) if settings_data else SpeciesSettings(),
            enabled=data.get("enabled", True),
        )

    def copy(self) -> Species:
        """Create an independent copy of this species."""
        return Species.from_dict(self.to_dict())


# ── Preset Factories ─────────────────────────────────────────


def default_herbivore() -> Species:
    """Create the default herbivore species (plant specialist)."""
    return Species(
        id="herbivore",
        name="Herbivore",
        color=config.COLOR_HERBIVORE,
        can_eat_plants=True,
        plant_food_multiplier=config.HERBIVORE_FOOD_BONUS,
        can_attack_other_species=False,
        can_attack_own_species=False,
        can_eat_other_corpse=False,
        can_eat_own_corpse=False,
        settings=SpeciesSettings(
            freeplay_initial_population=24,
            freeplay_carrying_capacity=48,
            freeplay_hard_cap=72,
        ),
    )


def default_carnivore() -> Species:
    """Create the default carnivore species (attacks other species)."""
    return Species(
        id="carnivore",
        name="Carnivore",
        color=config.COLOR_CARNIVORE,
        can_eat_plants=True,
        plant_food_multiplier=config.CARNIVORE_FOOD_PENALTY,
        can_attack_other_species=True,
        can_attack_own_species=False,
        can_eat_other_corpse=False,
        can_eat_own_corpse=False,
        attack_damage=config.CARNIVORE_ATTACK_DAMAGE,
        energy_steal_fraction=config.CARNIVORE_ENERGY_STEAL,
        settings=SpeciesSettings(
            freeplay_initial_population=8,
            freeplay_carrying_capacity=16,
            freeplay_hard_cap=24,
        ),
    )


def default_scavenger() -> Species:
    """Create the default scavenger species (eats other species' corpses)."""
    return Species(
        id="scavenger",
        name="Scavenger",
        color=config.COLOR_SCAVENGER,
        can_eat_plants=True,
        plant_food_multiplier=config.SCAVENGER_FOOD_PENALTY,
        can_attack_other_species=False,
        can_attack_own_species=False,
        can_eat_other_corpse=True,
        can_eat_own_corpse=False,
        scavenge_death_radius=config.SCAVENGER_DEATH_RADIUS,
        scavenge_death_energy=config.SCAVENGER_DEATH_ENERGY,
        settings=SpeciesSettings(
            freeplay_initial_population=8,
            freeplay_carrying_capacity=16,
            freeplay_hard_cap=24,
        ),
    )


# ── Legacy Migration ─────────────────────────────────────────

# Well-known species IDs for the 3 classic types
LEGACY_DIET_TO_SPECIES_ID = {
    config.DIET_HERBIVORE: "herbivore",
    config.DIET_CARNIVORE: "carnivore",
    config.DIET_SCAVENGER: "scavenger",
}


def species_id_from_legacy_diet(diet: int) -> str:
    """Convert a legacy diet integer to a species ID string."""
    return LEGACY_DIET_TO_SPECIES_ID.get(diet, "herbivore")


# ── Species Registry ─────────────────────────────────────────


class SpeciesRegistry:
    """
    A registry of all species in the simulation.

    Dict-like container mapping species_id -> Species.
    Supports unlimited species. Preserves insertion order.
    """

    def __init__(self) -> None:
        self._species: dict[str, Species] = {}

    def register(self, species: Species) -> str:
        """Add a species to the registry. Returns its id."""
        self._species[species.id] = species
        return species.id

    def get(self, species_id: str) -> Species | None:
        """Get a species by ID, or None if not found."""
        return self._species.get(species_id)

    def remove(self, species_id: str) -> Species | None:
        """Remove and return a species, or None if not found."""
        return self._species.pop(species_id, None)

    def all(self) -> list[Species]:
        """Return all registered species in insertion order."""
        return list(self._species.values())

    def ids(self) -> list[str]:
        """Return all registered species IDs."""
        return list(self._species.keys())

    def __len__(self) -> int:
        return len(self._species)

    def __contains__(self, species_id: str) -> bool:
        return species_id in self._species

    def __iter__(self) -> Iterator[Species]:
        return iter(self._species.values())

    def to_list(self) -> list[dict]:
        """Serialize all species to a list of dicts."""
        return [s.to_dict() for s in self._species.values()]

    @classmethod
    def from_list(cls, data: list[dict]) -> SpeciesRegistry:
        """Reconstruct a registry from a list of species dicts."""
        registry = cls()
        for item in data:
            registry.register(Species.from_dict(item))
        return registry

    def copy(self) -> SpeciesRegistry:
        """Create an independent copy of this registry."""
        return SpeciesRegistry.from_list(self.to_list())

    def generate_unique_id(self, base: str) -> str:
        """Generate a unique species ID based on a base string."""
        if base not in self._species:
            return base
        counter = 2
        while f"{base}_{counter}" in self._species:
            counter += 1
        return f"{base}_{counter}"


def default_registry() -> SpeciesRegistry:
    """Create a registry with the 3 classic species presets."""
    registry = SpeciesRegistry()
    registry.register(default_herbivore())
    registry.register(default_carnivore())
    registry.register(default_scavenger())
    return registry
