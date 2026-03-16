"""
Save/Load — JSON serialization for species DNA and full game saves.
============================================================
Export your top creatures to a .json file.

Also provides full game save/load so players can quit and resume later.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from pangea.dna import DNA


def save_species(
    dna_list: list[DNA],
    filepath: str,
    species_name: str = "",
    generation: int = 0,
) -> None:
    """
    Save a list of DNA to a JSON file.

    Args:
        dna_list:     List of DNA objects to save.
        filepath:     Path to the output .json file.
        species_name: Optional name for this species batch.
        generation:   Which generation these creatures came from.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    data = {
        "species_name": species_name or f"gen_{generation}_top{len(dna_list)}",
        "generation": generation,
        "creatures": [dna.to_dict() for dna in dna_list],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_species(filepath: str) -> tuple[list[DNA], dict]:
    """
    Load a list of DNA from a JSON file.

    Args:
        filepath: Path to the .json file to load.

    Returns:
        Tuple of (list of DNA objects, metadata dict with species_name and generation).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    dna_list = [DNA.from_dict(c) for c in data["creatures"]]
    metadata = {
        "species_name": data.get("species_name", "unknown"),
        "generation": data.get("generation", 0),
    }

    return dna_list, metadata


def list_species_files(directory: str = "species") -> list[str]:
    """
    List all .json species files in a directory.

    Args:
        directory: Path to the directory to scan.

    Returns:
        List of file paths (sorted alphabetically).
    """
    path = Path(directory)
    if not path.exists():
        return []
    return sorted(str(f) for f in path.glob("*.json"))


# ── DNA Stash Helpers ────────────────────────────────────────


def stash_species_dna(
    species,
    creatures: list,
    settings=None,
) -> int:
    """
    Select top performers and store their DNA on the species.

    Args:
        species:   Species object to stash DNA on.
        creatures: All creatures (alive or dead) to evaluate.
        settings:  Optional SimSettings for fitness weights.

    Returns:
        Number of DNA entries stashed.
    """
    from pangea.evolution import select_top

    species_creatures = [c for c in creatures if c.dna.species_id == species.id]
    if not species_creatures:
        return 0

    n = species.settings.top_performers_count
    top_dna = select_top(species_creatures, n=n, settings=settings)
    species.dna_stash = [dna.to_dict() for dna in top_dna]
    return len(species.dna_stash)


def clear_species_dna_stash(species) -> None:
    """Remove stashed DNA from a species."""
    species.dna_stash = None


# ── Full Game Saves ──────────────────────────────────────────

SAVES_DIR = "saves"


def save_game(
    dna_list: list[DNA],
    generation: int,
    settings_dict: dict,
    save_name: str = "",
) -> str:
    """
    Save a full game state to a JSON file in the saves/ directory.

    Args:
        dna_list:      DNA of the living creatures.
        generation:    Current generation number.
        settings_dict: SimSettings serialized via to_dict().
        save_name:     Optional display name for this save.

    Returns:
        The filepath that was written.
    """
    os.makedirs(SAVES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not save_name:
        save_name = f"freeplay gen {generation}"

    filename = f"freeplay_{timestamp}.json"
    filepath = os.path.join(SAVES_DIR, filename)

    data: dict = {
        "save_name": save_name,
        "mode": "freeplay",
        "generation": generation,
        "timestamp": timestamp,
        "settings": settings_dict,
        "creatures": [dna.to_dict() for dna in dna_list],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filepath


def load_game(filepath: str) -> dict:
    """
    Load a full game save.

    Returns:
        Dict with keys: save_name, mode, generation, settings (dict),
        creatures (list[DNA]), extra (dict or None).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "save_name": data.get("save_name", ""),
        "mode": data.get("mode", "freeplay"),
        "generation": data.get("generation", 0),
        "timestamp": data.get("timestamp", ""),
        "settings": data.get("settings", {}),
        "creatures": [DNA.from_dict(c) for c in data.get("creatures", [])],
    }


def list_saves() -> list[dict]:
    """
    List all game saves.

    Returns:
        List of dicts with keys: filepath, save_name, mode, generation, timestamp.
        Sorted newest-first.
    """
    path = Path(SAVES_DIR)
    if not path.exists():
        return []

    saves = []
    for f in path.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            saves.append({
                "filepath": str(f),
                "save_name": data.get("save_name", f.stem),
                "mode": data.get("mode", "freeplay"),
                "generation": data.get("generation", 0),
                "timestamp": data.get("timestamp", ""),
                "creature_count": len(data.get("creatures", [])),
                "snapshot": data.get("snapshot", False),
            })
        except Exception:
            continue

    saves.sort(key=lambda s: s["timestamp"], reverse=True)
    return saves


def delete_save(filepath: str) -> None:
    """Delete a game save file."""
    os.remove(filepath)


# ── Full Snapshot Saves (Freeplay) ──────────────────────────

def _creature_to_dict(creature) -> dict:
    """Serialize a Creature's full live state (DNA + dynamic fields)."""
    return {
        "dna": creature.dna.to_dict(),
        "x": creature.x,
        "y": creature.y,
        "heading": float(creature.heading),
        "speed": float(creature.speed),
        "energy": float(creature.energy),
        "food_eaten": creature.food_eaten,
        "age": creature.age,
        "alive": creature.alive,
        "last_turn": float(creature.last_turn),
        "under_attack": float(creature.under_attack),
        "death_processed": creature.death_processed,
        "territory_cells": [list(c) for c in creature.territory_cells],
        "breed_cooldown": creature.breed_cooldown,
        "offspring_count": creature.offspring_count,
        "generation": creature.generation,
        "time_to_first_food": creature.time_to_first_food,
        "energy_at_death": creature.energy_at_death,
    }


def _creature_from_dict(data: dict):
    """Reconstruct a Creature from a snapshot dict."""
    from pangea.creature import Creature

    dna = DNA.from_dict(data["dna"])
    creature = Creature(dna, data["x"], data["y"])
    creature.heading = data["heading"]
    creature.speed = data["speed"]
    creature.energy = data["energy"]
    creature.food_eaten = data["food_eaten"]
    creature.age = data["age"]
    creature.alive = data["alive"]
    creature.last_turn = data.get("last_turn", 0.0)
    creature.under_attack = data.get("under_attack", 0.0)
    creature.death_processed = data.get("death_processed", False)
    creature.territory_cells = {tuple(c) for c in data.get("territory_cells", [])}
    creature.breed_cooldown = data.get("breed_cooldown", 0.0)
    creature.offspring_count = data.get("offspring_count", 0)
    creature.generation = data.get("generation", 0)
    creature.time_to_first_food = data.get("time_to_first_food", -1.0)
    creature.energy_at_death = data.get("energy_at_death", -1.0)
    return creature


def _food_to_dict(food) -> dict:
    """Serialize a Food item."""
    return {
        "x": food.x, "y": food.y,
        "energy": food.energy, "radius": food.radius,
        "age": food.age, "lifetime": food.lifetime,
        "is_corpse": food.is_corpse,
        "species_id": food.species_id,
    }


def _hazard_to_dict(h) -> dict:
    """Serialize a Hazard."""
    return {"x": h.x, "y": h.y, "radius": h.radius,
            "damage_rate": h.damage_rate, "hazard_type": h.hazard_type}


def _biome_to_dict(b) -> dict:
    """Serialize a Biome."""
    return {"x": b.x, "y": b.y, "radius": b.radius,
            "biome_type": b.biome_type, "speed_multiplier": b.speed_multiplier}


def _zone_to_dict(z) -> dict:
    """Serialize a player Zone."""
    return {"x": z.x, "y": z.y, "radius": z.radius,
            "zone_type": z.zone_type, "strength": z.strength,
            "lifetime": z.lifetime, "age": z.age}


def _barrier_to_dict(b) -> dict:
    """Serialize a player Barrier."""
    return {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
            "thickness": b.thickness, "lifetime": b.lifetime, "age": b.age}


def save_snapshot(
    world,
    settings_dict: dict,
    freeplay_state: dict,
    tools=None,
    save_name: str = "",
) -> str:
    """
    Save a full simulation snapshot — every creature, food item, hazard, etc.

    Args:
        world:          The World object with all live state.
        settings_dict:  SimSettings serialized via to_dict().
        freeplay_state: Dict of freeplay tracking vars (elapsed, peak_pop, history, etc.).
        tools:          PlayerTools instance (optional).
        save_name:      Optional display name.

    Returns:
        The filepath that was written.
    """
    os.makedirs(SAVES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not save_name:
        alive = sum(1 for c in world.creatures if c.alive)
        save_name = f"freeplay snapshot ({alive} alive)"

    filename = f"freeplay_snapshot_{timestamp}.json"
    filepath = os.path.join(SAVES_DIR, filename)

    data: dict = {
        "save_name": save_name,
        "mode": "freeplay",
        "snapshot": True,
        "timestamp": timestamp,
        "settings": settings_dict,
        # World state
        "creatures": [_creature_to_dict(c) for c in world.creatures]
                      + [_creature_to_dict(c) for stash in world._stashed_creatures.values() for c in stash],
        "food": [_food_to_dict(f) for f in world.food],
        "hazards": [_hazard_to_dict(h) for h in world.hazards],
        "biomes": [_biome_to_dict(b) for b in world.biomes],
        "world_timers": {
            "elapsed_time": world.elapsed_time,
            "season_time": world.season_time,
            "day_night_time": world.day_night_time,
            "total_births": world.total_births,
            "total_deaths": world.total_deaths,
            "food_spawn_accum": world._food_spawn_accum,
        },
        # Freeplay tracking
        "freeplay_state": freeplay_state,
    }

    # Player tools
    if tools:
        data["tools"] = {
            "drought_active": tools.drought_active,
            "zones": [_zone_to_dict(z) for z in tools.zones],
            "barriers": [_barrier_to_dict(b) for b in tools.barriers],
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filepath


def load_snapshot(filepath: str) -> dict:
    """
    Load a full simulation snapshot.

    Returns:
        Dict with keys: save_name, settings, creatures (list[Creature]),
        food, hazards, biomes, world_timers, freeplay_state, tools.
    """
    from pangea.world import Food, Hazard, Biome
    from pangea.tools import Zone, Barrier

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    creatures = [_creature_from_dict(c) for c in data.get("creatures", [])]

    food = [
        Food(x=f["x"], y=f["y"], energy=f["energy"], radius=f["radius"],
             age=f["age"], lifetime=f["lifetime"], is_corpse=f.get("is_corpse", False),
             species_id=f.get("species_id", ""))
        for f in data.get("food", [])
    ]

    hazards = [
        Hazard(x=h["x"], y=h["y"], radius=h["radius"],
               damage_rate=h.get("damage_rate", 0), hazard_type=h.get("hazard_type", "lava"))
        for h in data.get("hazards", [])
    ]

    biomes = [
        Biome(x=b["x"], y=b["y"], radius=b["radius"],
              biome_type=b["biome_type"], speed_multiplier=b["speed_multiplier"])
        for b in data.get("biomes", [])
    ]

    tools_data = data.get("tools")
    zones = []
    barriers = []
    drought = False
    if tools_data:
        drought = tools_data.get("drought_active", False)
        zones = [
            Zone(x=z["x"], y=z["y"], radius=z["radius"], zone_type=z["zone_type"],
                 strength=z.get("strength", 1.0), lifetime=z["lifetime"], age=z["age"])
            for z in tools_data.get("zones", [])
        ]
        barriers = [
            Barrier(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                    thickness=b.get("thickness", 6.0), lifetime=b["lifetime"], age=b["age"])
            for b in tools_data.get("barriers", [])
        ]

    return {
        "save_name": data.get("save_name", ""),
        "snapshot": True,
        "settings": data.get("settings", {}),
        "creatures": creatures,
        "food": food,
        "hazards": hazards,
        "biomes": biomes,
        "world_timers": data.get("world_timers", {}),
        "freeplay_state": data.get("freeplay_state", {}),
        "tools_drought": drought,
        "tools_zones": zones,
        "tools_barriers": barriers,
    }


# ── Community / DNA Exchange ──────────────────────────────────

def upload_species(filepath: str, server_url: str) -> dict:
    """
    Upload a species file to the DNA Exchange API.

    Args:
        filepath:   Path to the local .json species file.
        server_url: Base URL of the API (e.g. "http://localhost:8000").

    Returns:
        Server response dict with keys: id, name, token.
    """
    import urllib.request
    import urllib.error

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    upload_data: dict = {
        "species_name": data.get("species_name", ""),
        "generation": data.get("generation", 0),
        "creatures": data.get("creatures", []),
    }
    # Include species config if the file was saved as a species settings file
    # (has "id" key from Species.to_dict())
    if "id" in data:
        # Strip dna_stash from config to avoid duplication (creatures IS the DNA)
        config = {k: v for k, v in data.items() if k not in ("dna_stash", "creatures")}
        upload_data["species_config"] = config
    payload = json.dumps(upload_data).encode("utf-8")

    url = f"{server_url.rstrip('/')}/species"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_species(
    species_id: int | str,
    server_url: str,
    save_dir: str = "species",
) -> str:
    """
    Download a species from the DNA Exchange API and save it locally.

    Args:
        species_id: The ID of the species to download.
        server_url: Base URL of the API.
        save_dir:   Local directory to save the file.

    Returns:
        The local filepath where the species was saved.
    """
    import urllib.request

    url = f"{server_url.rstrip('/')}/species/{species_id}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    os.makedirs(save_dir, exist_ok=True)
    name = data.get("species_name", f"species_{species_id}")
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filepath = os.path.join(save_dir, f"{safe_name}_{species_id}.json")

    # If server returned a species config, save as a full species settings file
    species_config = data.get("species_config")
    if species_config:
        save_data = dict(species_config)
        save_data["dna_stash"] = data.get("creatures", [])
    else:
        save_data = {
            "species_name": data.get("species_name", name),
            "generation": data.get("generation", 0),
            "creatures": data.get("creatures", []),
        }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    return filepath


def list_remote_species(server_url: str, page: int = 1) -> list[dict]:
    """
    Fetch the species list from the DNA Exchange API.

    Returns:
        List of dicts with keys: id, name, generation, wins, losses.
    """
    import urllib.request

    url = f"{server_url.rstrip('/')}/species?page={page}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("species", [])
