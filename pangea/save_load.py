"""
Save/Load -- JSON serialization for species DNA.
============================================================
Export your top creatures to a .json file so another user can
load them into Convergence Mode on their machine.

JSON schema:
{
    "species_name": "gen_42_top10",
    "generation": 42,
    "creatures": [
        {
            "weights": {"W1": [[...]], "b1": [...], "W2": [[...]], "b2": [...]},
            "speed": 30, "size": 20, "vision": 25, "efficiency": 25
        },
        ...
    ]
}
"""

from __future__ import annotations

import json
import os
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
