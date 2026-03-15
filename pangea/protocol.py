"""
Network Protocol — message types, serialization, and snapshot logic.
====================================================================
Pure-data module with no I/O. Defines the message format for
host-client communication and provides functions to convert between
World state and compact binary snapshots (via msgpack).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgpack

if TYPE_CHECKING:
    from pangea.settings import SimSettings
    from pangea.tools import PlayerTools
    from pangea.world import World


# ── Message Types (short strings for compact msgpack) ────────

class MsgType:
    """Constants for the ``"t"`` field in every network message."""

    SNAPSHOT = "s"
    FULL_STATE = "fs"
    TOOL_ACTION = "ta"
    SETTINGS_CHANGE = "sc"
    GENERATION_END = "ge"
    JOIN = "j"
    LEAVE = "l"
    ROOM_CREATE = "rc"
    HOST_LEFT = "hl"
    CLIENT_JOINED = "cj"


# ── Pack / Unpack ────────────────────────────────────────────

def pack(msg: dict) -> bytes:
    """Serialize a message dict to msgpack bytes."""
    return msgpack.packb(msg, use_bin_type=True)


def unpack(data: bytes) -> dict:
    """Deserialize msgpack bytes to a message dict."""
    return msgpack.unpackb(data, raw=False)


# ── Compact Per-Frame Snapshot ───────────────────────────────

def _creature_snapshot(c) -> list:
    """Minimal creature state for per-frame sync (list for compactness)."""
    return [
        float(c.x), float(c.y), float(c.heading), float(c.speed),
        float(c.energy), c.alive, float(c.dna.effective_radius),
        "", c.food_eaten, float(c.age), c.dna.species_id,
        c.death_processed, float(c.under_attack),
    ]


def _food_snapshot(f) -> list:
    """Minimal food state for per-frame sync."""
    return [float(f.x), float(f.y), float(f.energy), float(f.radius),
            float(f.age), float(f.lifetime), f.is_corpse, f.species_id]


def snapshot_from_world(world: World) -> dict:
    """Build a compact snapshot dict suitable for per-frame broadcast."""
    return {
        "t": MsgType.SNAPSHOT,
        "c": [_creature_snapshot(c) for c in world.creatures],
        "f": [_food_snapshot(f) for f in world.food],
        "et": world.elapsed_time,
        "st": world.season_time,
        "dnt": world.day_night_time,
        "gen": world.generation,
        "tb": world.total_births,
        "td": world.total_deaths,
    }


def apply_snapshot(world: World, data: dict) -> None:
    """
    Update a World object in-place from a compact snapshot.

    Creatures are matched by index (host maintains stable ordering).
    If the count changes, creatures are added or removed from the tail.
    """
    from pangea.creature import Creature
    from pangea.dna import DNA

    # ── Creatures ────────────────────────────────────────────
    snap_creatures = data["c"]
    # Sync list length
    while len(world.creatures) > len(snap_creatures):
        world.creatures.pop()
    while len(world.creatures) < len(snap_creatures):
        # Placeholder creature — will be overwritten below
        world.creatures.append(Creature(DNA.random(), 0, 0))

    for i, sc in enumerate(snap_creatures):
        cr = world.creatures[i]
        cr.x = sc[0]
        cr.y = sc[1]
        cr.heading = sc[2]
        cr.speed = sc[3]
        cr.energy = sc[4]
        cr.alive = sc[5]
        # sc[6] = effective_radius (read-only from DNA, skip)
        # sc[7] = reserved (unused)
        cr.food_eaten = sc[8]
        cr.age = sc[9]
        # sc[10] = species_id (read-only from DNA, skip)
        cr.death_processed = sc[11]
        cr.under_attack = sc[12]

    # ── Food ─────────────────────────────────────────────────
    from pangea.world import Food

    world.food = [
        Food(x=f[0], y=f[1], energy=f[2], radius=f[3],
             age=f[4], lifetime=f[5], is_corpse=f[6],
             species_id=f[7] if len(f) > 7 else "")
        for f in data["f"]
    ]

    # ── Timers ───────────────────────────────────────────────
    world.elapsed_time = data["et"]
    world.season_time = data["st"]
    world.day_night_time = data["dnt"]
    world.generation = data["gen"]
    world.total_births = data.get("tb", 0)
    world.total_deaths = data.get("td", 0)


# ── Full State (for client join / reconnect) ─────────────────

def full_state_from_world(
    world: World,
    settings: SimSettings,
    tools: PlayerTools,
    generation: int,
    generation_history: list[dict] | None = None,
) -> dict:
    """
    Build a complete state dict for sending to a newly joined client.

    Reuses the serialization helpers from save_load.py.
    """
    from pangea.save_load import (
        _barrier_to_dict,
        _biome_to_dict,
        _creature_to_dict,
        _food_to_dict,
        _hazard_to_dict,
        _zone_to_dict,
    )

    return {
        "t": MsgType.FULL_STATE,
        "mode": "freeplay",
        "generation": generation,
        "settings": settings.to_dict(),
        "creatures": [_creature_to_dict(c) for c in world.creatures],
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
        "tools": {
            "drought_active": tools.drought_active,
            "zones": [_zone_to_dict(z) for z in tools.zones],
            "barriers": [_barrier_to_dict(b) for b in tools.barriers],
        },
        "generation_history": generation_history or [],
        "freeplay": world.freeplay,
    }


def apply_full_state(data: dict):
    """
    Reconstruct World, SimSettings, and PlayerTools from a full-state message.

    Returns:
        Tuple of (world, settings, tools, mode, generation, generation_history).
    """
    from pangea.save_load import _creature_from_dict
    from pangea.settings import SimSettings
    from pangea.tools import Barrier, PlayerTools, Zone
    from pangea.world import Biome, Food, Hazard, World

    settings = SimSettings.from_dict(data["settings"])

    creatures = [_creature_from_dict(c) for c in data["creatures"]]

    # Prevent World.__init__ from auto-generating entities we're about to overwrite
    saved_food = settings.initial_food_count
    saved_hazards = settings.hazard_count
    saved_biomes = settings.biome_count
    settings.initial_food_count = 0
    settings.hazard_count = 0
    settings.biome_count = 0

    world = World(
        creatures,
        width=settings.world_width,
        height=settings.world_height,
        settings=settings,
    )

    # Restore original settings values
    settings.initial_food_count = saved_food
    settings.hazard_count = saved_hazards
    settings.biome_count = saved_biomes

    # Restore food
    world.food = [
        Food(x=f["x"], y=f["y"], energy=f["energy"], radius=f["radius"],
             age=f["age"], lifetime=f["lifetime"], is_corpse=f.get("is_corpse", False),
             species_id=f.get("species_id", ""))
        for f in data.get("food", [])
    ]

    # Restore hazards and biomes
    world.hazards = [
        Hazard(x=h["x"], y=h["y"], radius=h["radius"],
               damage_rate=h.get("damage_rate", 0), hazard_type=h.get("hazard_type", "lava"))
        for h in data.get("hazards", [])
    ]
    world.biomes = [
        Biome(x=b["x"], y=b["y"], radius=b["radius"],
              biome_type=b["biome_type"], speed_multiplier=b["speed_multiplier"])
        for b in data.get("biomes", [])
    ]

    # Restore timers
    timers = data.get("world_timers", {})
    world.elapsed_time = timers.get("elapsed_time", 0.0)
    world.season_time = timers.get("season_time", 0.0)
    world.day_night_time = timers.get("day_night_time", 0.0)
    world.total_births = timers.get("total_births", 0)
    world.total_deaths = timers.get("total_deaths", 0)
    world._food_spawn_accum = timers.get("food_spawn_accum", 0.0)
    world.freeplay = data.get("freeplay", False)

    # Restore tools
    tools = PlayerTools()
    tools_data = data.get("tools", {})
    tools.drought_active = tools_data.get("drought_active", False)
    tools.zones = [
        Zone(x=z["x"], y=z["y"], radius=z["radius"], zone_type=z["zone_type"],
             strength=z.get("strength", 1.0), lifetime=z["lifetime"], age=z["age"])
        for z in tools_data.get("zones", [])
    ]
    tools.barriers = [
        Barrier(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                thickness=b.get("thickness", 6.0), lifetime=b["lifetime"], age=b["age"])
        for b in tools_data.get("barriers", [])
    ]
    world.tools = tools

    return (
        world,
        settings,
        tools,
        "freeplay",
        data.get("generation", 1),
        data.get("generation_history", []),
    )


# ── Tool Action & Settings Messages ─────────────────────────

def tool_action_msg(
    tool_type: str,
    x: float,
    y: float,
    x2: float | None = None,
    y2: float | None = None,
) -> dict:
    """Create a tool action message."""
    msg: dict = {"t": MsgType.TOOL_ACTION, "tool": tool_type, "x": x, "y": y}
    if x2 is not None:
        msg["x2"] = x2
    if y2 is not None:
        msg["y2"] = y2
    return msg


def settings_change_msg(changes: dict) -> dict:
    """Create a settings change message with only the modified fields."""
    return {"t": MsgType.SETTINGS_CHANGE, "changes": changes}


def generation_end_msg(
    generation: int,
    top_dna_dicts: list[dict],
    stats: dict,
) -> dict:
    """Create a generation-end summary message."""
    return {
        "t": MsgType.GENERATION_END,
        "generation": generation,
        "top_dna": top_dna_dicts,
        "stats": stats,
    }
