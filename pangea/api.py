"""
DNA Exchange API — REST server for community species sharing.
=============================================================
Upload/download evolved species, request matchups, report results,
and view leaderboards.

Usage:
    python -m pangea.api               # default port 8000
    python -m pangea.api --port 9000   # custom port

Requires optional 'server' dependencies: fastapi, uvicorn, aiosqlite.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Pangea DNA Exchange", version="0.1.0")

DB_PATH = Path("pangea_community.db")


# ── Database ─────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS species (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            generation INTEGER DEFAULT 0,
            dna_json TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            upload_token TEXT NOT NULL,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species_a_id INTEGER NOT NULL,
            species_b_id INTEGER NOT NULL,
            winner TEXT,
            created_at TEXT NOT NULL,
            reported_at TEXT,
            FOREIGN KEY (species_a_id) REFERENCES species(id),
            FOREIGN KEY (species_b_id) REFERENCES species(id)
        )
    """)
    db.commit()
    return db


# ── Models ───────────────────────────────────────────────────

class SpeciesUpload(BaseModel):
    species_name: str = ""
    generation: int = 0
    creatures: list[dict]


class MatchResult(BaseModel):
    match_id: int
    winner: str  # "A" or "B"
    food_a: int = 0
    food_b: int = 0


# ── Endpoints ────────────────────────────────────────────────

@app.post("/species")
def upload_species(data: SpeciesUpload):
    """Upload a species. Returns the species ID and an upload token."""
    if not data.creatures:
        raise HTTPException(400, "No creatures in payload")

    token = secrets.token_urlsafe(16)
    now = datetime.now().isoformat()
    dna_json = json.dumps(data.creatures)
    name = data.species_name or f"species_{now[:10]}"

    db = _get_db()
    cursor = db.execute(
        "INSERT INTO species (name, generation, dna_json, uploaded_at, upload_token) "
        "VALUES (?, ?, ?, ?, ?)",
        (name, data.generation, dna_json, now, token),
    )
    db.commit()
    species_id = cursor.lastrowid
    db.close()

    return {"id": species_id, "name": name, "token": token}


@app.get("/species")
def list_species(page: int = 1, per_page: int = 20):
    """List available species, paginated."""
    db = _get_db()
    offset = (page - 1) * per_page
    rows = db.execute(
        "SELECT id, name, generation, wins, losses, uploaded_at "
        "FROM species ORDER BY uploaded_at DESC LIMIT ? OFFSET ?",
        (per_page, offset),
    ).fetchall()

    total = db.execute("SELECT COUNT(*) FROM species").fetchone()[0]
    db.close()

    return {
        "species": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@app.get("/species/{species_id}")
def get_species(species_id: int):
    """Download a species by ID."""
    db = _get_db()
    row = db.execute(
        "SELECT id, name, generation, dna_json, wins, losses "
        "FROM species WHERE id = ?",
        (species_id,),
    ).fetchone()
    db.close()

    if row is None:
        raise HTTPException(404, "Species not found")

    return {
        "id": row["id"],
        "species_name": row["name"],
        "generation": row["generation"],
        "creatures": json.loads(row["dna_json"]),
        "wins": row["wins"],
        "losses": row["losses"],
    }


@app.delete("/species/{species_id}")
def delete_species(species_id: int, token: str = ""):
    """Delete a species by ID (requires upload token)."""
    db = _get_db()
    row = db.execute(
        "SELECT upload_token FROM species WHERE id = ?", (species_id,),
    ).fetchone()

    if row is None:
        raise HTTPException(404, "Species not found")
    if row["upload_token"] != token:
        raise HTTPException(403, "Invalid token")

    db.execute("DELETE FROM species WHERE id = ?", (species_id,))
    db.commit()
    db.close()
    return {"ok": True}


@app.post("/match")
def create_match(species_id: int):
    """Request a matchup for the given species. Server picks an opponent."""
    db = _get_db()

    # Verify species exists
    own = db.execute("SELECT id FROM species WHERE id = ?", (species_id,)).fetchone()
    if own is None:
        db.close()
        raise HTTPException(404, "Species not found")

    # Pick a random opponent
    opponent = db.execute(
        "SELECT id, name, generation, dna_json FROM species "
        "WHERE id != ? ORDER BY RANDOM() LIMIT 1",
        (species_id,),
    ).fetchone()

    if opponent is None:
        db.close()
        raise HTTPException(400, "No other species available for matchup")

    now = datetime.now().isoformat()
    cursor = db.execute(
        "INSERT INTO matches (species_a_id, species_b_id, created_at) VALUES (?, ?, ?)",
        (species_id, opponent["id"], now),
    )
    db.commit()
    match_id = cursor.lastrowid
    db.close()

    return {
        "match_id": match_id,
        "opponent": {
            "id": opponent["id"],
            "name": opponent["name"],
            "generation": opponent["generation"],
            "creatures": json.loads(opponent["dna_json"]),
        },
    }


@app.post("/results")
def report_results(data: MatchResult):
    """Report the outcome of a match."""
    db = _get_db()

    match = db.execute(
        "SELECT species_a_id, species_b_id, winner FROM matches WHERE id = ?",
        (data.match_id,),
    ).fetchone()

    if match is None:
        db.close()
        raise HTTPException(404, "Match not found")
    if match["winner"] is not None:
        db.close()
        raise HTTPException(400, "Match already reported")

    now = datetime.now().isoformat()
    db.execute(
        "UPDATE matches SET winner = ?, reported_at = ? WHERE id = ?",
        (data.winner, now, data.match_id),
    )

    if data.winner == "A":
        db.execute("UPDATE species SET wins = wins + 1 WHERE id = ?", (match["species_a_id"],))
        db.execute("UPDATE species SET losses = losses + 1 WHERE id = ?", (match["species_b_id"],))
    elif data.winner == "B":
        db.execute("UPDATE species SET wins = wins + 1 WHERE id = ?", (match["species_b_id"],))
        db.execute("UPDATE species SET losses = losses + 1 WHERE id = ?", (match["species_a_id"],))

    db.commit()
    db.close()
    return {"ok": True}


@app.get("/leaderboard")
def leaderboard(limit: int = 20):
    """Get the top species ranked by win rate."""
    db = _get_db()
    rows = db.execute(
        "SELECT id, name, generation, wins, losses, "
        "CASE WHEN (wins + losses) > 0 THEN CAST(wins AS REAL) / (wins + losses) ELSE 0 END AS win_rate "
        "FROM species WHERE (wins + losses) > 0 "
        "ORDER BY win_rate DESC, wins DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    return {"leaderboard": [dict(r) for r in rows]}


# ── Entry Point ──────────────────────────────────────────────

def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description="Pangea DNA Exchange API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
