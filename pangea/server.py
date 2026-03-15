"""
Relay Server — lightweight WebSocket proxy for multiplayer rooms.
=================================================================
A standalone process that forwards messages between host and clients.
No simulation logic, no pygame dependency.

Usage:
    python -m pangea.server              # default port 8765
    python -m pangea.server --port 9000  # custom port
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import string
import threading

import websockets
from websockets.asyncio.server import ServerConnection

from pangea.protocol import MsgType, pack, unpack

logging.basicConfig(level=logging.INFO, format="%(asctime)s [relay] %(message)s")
log = logging.getLogger(__name__)


# ── Room Management ──────────────────────────────────────────

class Room:
    """A game room with one host and zero or more clients."""

    def __init__(self, code: str, host: ServerConnection) -> None:
        self.code = code
        self.host = host
        self.clients: list[ServerConnection] = []


rooms: dict[str, Room] = {}
ws_to_room: dict[ServerConnection, str] = {}
ws_role: dict[ServerConnection, str] = {}  # "host" or "client"


def _generate_code(length: int = 4) -> str:
    """Generate a unique room code."""
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
        if code not in rooms:
            return code


# ── Connection Handler ───────────────────────────────────────

async def handler(websocket: ServerConnection) -> None:
    """Handle a single WebSocket connection."""
    try:
        async for raw in websocket:
            msg = unpack(raw)
            msg_type = msg.get("t")

            if msg_type == MsgType.ROOM_CREATE:
                code = _generate_code()
                room = Room(code, websocket)
                rooms[code] = room
                ws_to_room[websocket] = code
                ws_role[websocket] = "host"
                await websocket.send(pack({"t": MsgType.ROOM_CREATE, "code": code}))
                log.info("Room %s created by host", code)

            elif msg_type == MsgType.JOIN:
                code = msg.get("code", "")
                room = rooms.get(code)
                if room is None:
                    await websocket.send(pack({"t": "error", "msg": "Room not found"}))
                    continue
                room.clients.append(websocket)
                ws_to_room[websocket] = code
                ws_role[websocket] = "client"
                # Notify host that a client joined
                try:
                    await room.host.send(pack({
                        "t": MsgType.CLIENT_JOINED,
                        "players": len(room.clients),
                    }))
                except websockets.ConnectionClosed:
                    pass
                # Confirm join to client
                await websocket.send(pack({"t": MsgType.JOIN, "code": code, "ok": True}))
                log.info("Client joined room %s (%d clients)", code, len(room.clients))

            else:
                # Route messages: host → all clients, client → host
                code = ws_to_room.get(websocket)
                if code is None:
                    continue
                room = rooms.get(code)
                if room is None:
                    continue

                role = ws_role.get(websocket, "")
                if role == "host":
                    # Broadcast to all clients
                    closed = []
                    for client in room.clients:
                        try:
                            await client.send(raw)
                        except websockets.ConnectionClosed:
                            closed.append(client)
                    for c in closed:
                        room.clients.remove(c)
                        ws_to_room.pop(c, None)
                        ws_role.pop(c, None)

                elif role == "client":
                    # Forward to host
                    try:
                        await room.host.send(raw)
                    except websockets.ConnectionClosed:
                        pass

    except websockets.ConnectionClosed:
        pass
    finally:
        await _handle_disconnect(websocket)


async def _handle_disconnect(websocket: ServerConnection) -> None:
    """Clean up when a connection drops."""
    code = ws_to_room.pop(websocket, None)
    role = ws_role.pop(websocket, None)
    if code is None:
        return

    room = rooms.get(code)
    if room is None:
        return

    if role == "host":
        # Notify all clients that host left
        for client in room.clients:
            try:
                await client.send(pack({"t": MsgType.HOST_LEFT}))
            except websockets.ConnectionClosed:
                pass
            ws_to_room.pop(client, None)
            ws_role.pop(client, None)
        del rooms[code]
        log.info("Host left, room %s closed", code)

    elif role == "client":
        if websocket in room.clients:
            room.clients.remove(websocket)
        # Notify host
        try:
            await room.host.send(pack({
                "t": MsgType.LEAVE,
                "players": len(room.clients),
            }))
        except websockets.ConnectionClosed:
            pass
        log.info("Client left room %s (%d clients remain)", code, len(room.clients))


# ── Entry Point ──────────────────────────────────────────────

async def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Start the relay server."""
    log.info("Relay server starting on %s:%d", host, port)
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever


# ── Embedded Server (for host-in-process) ────────────────────

class EmbeddedRelay:
    """
    Run the relay server in a background thread so the host doesn't
    need a separate process.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._error: str | None = None

    def start(self) -> None:
        """Start the relay in a background thread. Blocks until ready."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._error:
            raise RuntimeError(self._error)
        log.info("Embedded relay ready on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Stop the relay server."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        # Clear global room state so a fresh start is clean
        rooms.clear()
        ws_to_room.clear()
        ws_role.clear()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            self._error = str(exc)
            self._ready.set()

    async def _serve(self) -> None:
        try:
            async with websockets.serve(handler, self._host, self._port):
                self._ready.set()
                await asyncio.Future()  # run until stopped
        except asyncio.CancelledError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Pangea Multiplayer Relay Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port number")
    args = parser.parse_args()
    asyncio.run(run_server(args.host, args.port))


if __name__ == "__main__":
    main()
