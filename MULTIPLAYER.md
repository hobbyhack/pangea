# Multiplayer Design — Host-Authority + DNA Exchange

Multiplayer support for Pangea: LAN and internet. Simulation stays mostly local. A thin server relays data but runs no simulation logic. Two complementary modes:

## Mode 1: Host-Authority (Real-Time Shared Play)

One player runs the authoritative simulation. A thin WebSocket relay server forwards compressed state to other players, who render locally and send tool commands back.

### Roles

- **Host:** Runs the full simulation (World, Creatures, Evolution — everything that `simulation.py` does today). Broadcasts world state snapshots. Receives and applies remote player tool actions.
- **Client:** Receives snapshots, renders them locally via `renderer.py`. Sends tool actions (food placement, zones, barriers, drought) and settings changes back to host. Does NOT run `world.update()`.
- **Relay Server:** A lightweight WebSocket proxy (~50-100 lines). Forwards messages between host and clients. Handles lobby/room creation. No simulation logic. Deployable on any free tier.

### Data Flow

- **Host → Clients:** Delta-compressed snapshots every N frames. Contains creature positions/headings/energy/alive, food positions, scores, generation info. ~2-5 KB per update at 10 updates/sec with 50 creatures.
- **Clients → Host:** Tool actions (mouse clicks, zone placements, barrier drags). ~50-100 bytes per action. Settings changes as diffs.
- **Generation boundaries:** Host sends full results (top DNA summary, generation stats, lineage scores).

### Key Design Decisions

- Single authority eliminates floating-point determinism issues across platforms.
- Tool action latency (~50-200ms) is acceptable because players influence environment, not individual creatures.
- Host leaving ends the game (can mitigate with save-on-disconnect).
- Clients can join mid-game by receiving a full state snapshot on connect.

### What Changes in Existing Code

- `simulation.py` — Host mode wraps the existing loop, broadcasting state after each `world.update()`. Client mode replaces `world.update()` with snapshot application.
- `renderer.py` — No changes needed. It already reads World state to draw.
- `tools.py` — Tool actions become serializable commands (tool type + coordinates + parameters).
- `menu.py` — Add "Host Game" / "Join Game" options. Join needs IP/room code entry.
- `settings.py` — `SimSettings` changes from any player get relayed to host and applied next generation.
- `world.py` — Add snapshot export/import methods (creature list, food list, scores).

### New Files

- `pangea/network.py` — Host and Client classes. WebSocket connections, snapshot serialization (msgpack), message protocol.
- `pangea/server.py` — Standalone thin relay server. Room management, player connect/disconnect, message forwarding.

### Serialization

Use msgpack for compact binary snapshots. Creature state per frame: x, y, heading, speed, energy, alive, radius, color, lineage (~40 bytes each). Food: x, y, energy (~12 bytes each).

### Protocol Messages

- `join` / `leave` / `room_create` / `room_list` — lobby management
- `snapshot` — periodic world state from host
- `tool_action` — player tool command (type, coords, params)
- `settings_change` — settings diff from any player
- `generation_end` — results + new generation start signal
- `full_state` — sent on client join for catch-up

## Mode 2: DNA Exchange (Async Competitive)

Players evolve species locally in Isolation, then upload top DNA to a server for matchmaking and automated Convergence tournaments.

### Server Responsibilities

- Store species DNA files (JSON, ~10KB each, already produced by `save_load.py`).
- Match species for convergence battles (random, ranked, or challenge-based).
- Track leaderboards and win/loss records.

### What Stays Local

Everything. Full simulation on each client.

### Implementation

- Extend `save_load.py` with upload/download functions (HTTP POST/GET to server API).
- Add a "Community" or "Online" menu option to browse/submit species.
- Server is a simple REST API (FastAPI or Flask) with SQLite storage.
- Convergence battles can run on either player's machine or a lightweight server instance.

**Why:** This is a natural extension of the existing species export system and requires minimal new code. It provides multiplayer value even with zero server compute for simulation.

## Implementation Priority

1. **Network protocol + Host/Client classes** (`pangea/network.py`)
2. **World snapshot serialization** (methods on World class)
3. **Tool action serialization** (methods on PlayerTools)
4. **Relay server** (`pangea/server.py`)
5. **Menu integration** (Host/Join options)
6. **Simulation loop refactor** (host vs client mode branching)
7. **DNA Exchange API** (server + client upload/download)

## Dependencies to Add

- `websockets` — async WebSocket library for Python
- `msgpack` — compact binary serialization
- `fastapi` + `uvicorn` — for DNA Exchange REST API (server-side only)
