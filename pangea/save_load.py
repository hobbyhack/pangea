"""
Save/Load — JSON serialization for species DNA and full game saves.
============================================================
Export your top creatures to a .json file so another user can
load them into Convergence Mode on their machine.

Also provides full game save/load so players can quit and resume
any game mode (Isolation, Convergence, Freeplay) later.
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


# ── Full Game Saves ──────────────────────────────────────────

SAVES_DIR = "saves"


def save_game(
    mode: str,
    dna_list: list[DNA],
    generation: int,
    settings_dict: dict,
    save_name: str = "",
    extra: dict | None = None,
) -> str:
    """
    Save a full game state to a JSON file in the saves/ directory.

    Args:
        mode:          Game mode ("isolation", "convergence", "freeplay").
        dna_list:      DNA of the top performers / living creatures.
        generation:    Current generation number.
        settings_dict: SimSettings serialized via to_dict().
        save_name:     Optional display name for this save.
        extra:         Optional extra data (e.g. convergence file paths, stats).

    Returns:
        The filepath that was written.
    """
    os.makedirs(SAVES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not save_name:
        save_name = f"{mode} gen {generation}"

    filename = f"{mode}_{timestamp}.json"
    filepath = os.path.join(SAVES_DIR, filename)

    data: dict = {
        "save_name": save_name,
        "mode": mode,
        "generation": generation,
        "timestamp": timestamp,
        "settings": settings_dict,
        "creatures": [dna.to_dict() for dna in dna_list],
    }
    if extra:
        data["extra"] = extra

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
        "mode": data["mode"],
        "generation": data.get("generation", 0),
        "timestamp": data.get("timestamp", ""),
        "settings": data.get("settings", {}),
        "creatures": [DNA.from_dict(c) for c in data.get("creatures", [])],
        "extra": data.get("extra"),
    }


def list_saves(mode: str | None = None) -> list[dict]:
    """
    List all game saves, optionally filtered by mode.

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
            if mode and data.get("mode") != mode:
                continue
            saves.append({
                "filepath": str(f),
                "save_name": data.get("save_name", f.stem),
                "mode": data.get("mode", "unknown"),
                "generation": data.get("generation", 0),
                "timestamp": data.get("timestamp", ""),
                "creature_count": len(data.get("creatures", [])),
            })
        except Exception:
            continue

    saves.sort(key=lambda s: s["timestamp"], reverse=True)
    return saves


def delete_save(filepath: str) -> None:
    """Delete a game save file."""
    os.remove(filepath)
