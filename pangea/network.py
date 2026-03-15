"""
Network I/O — Host and Client classes for multiplayer communication.
====================================================================
Each class runs a WebSocket connection in a background thread with its
own asyncio event loop. Communication with the pygame main thread
happens through thread-safe queues.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

import websockets

from pangea.protocol import MsgType, pack, unpack

log = logging.getLogger(__name__)


class NetworkHost:
    """
    Manages the host side of a multiplayer session.

    The host creates a room on the relay server, broadcasts snapshots
    to connected clients, and receives tool actions / settings changes.
    """

    def __init__(self, relay_url: str) -> None:
        self._relay_url = relay_url
        self._outbound: queue.Queue[bytes] = queue.Queue()
        self._inbound: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()
        self._room_code: str = ""
        self._player_count = 0
        self._connected = False
        self._ready = threading.Event()
        self._error: str | None = None

    @property
    def room_code(self) -> str:
        return self._room_code

    @property
    def player_count(self) -> int:
        return self._player_count

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> str:
        """
        Start the network thread, connect to relay, and create a room.

        Returns:
            The room code assigned by the relay server.

        Raises:
            RuntimeError: If connection or room creation fails.
        """
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)
        if self._error:
            raise RuntimeError(f"Failed to start host: {self._error}")
        return self._room_code

    def stop(self) -> None:
        """Signal the network thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._connected = False

    def broadcast_snapshot(self, data: dict) -> None:
        """Enqueue a snapshot for broadcast to all clients."""
        self._outbound.put(pack(data))

    def broadcast_full_state(self, data: dict) -> None:
        """Enqueue a full state message for broadcast."""
        self._outbound.put(pack(data))

    def send_to_clients(self, data: dict) -> None:
        """Enqueue any message for broadcast."""
        self._outbound.put(pack(data))

    def poll_incoming(self) -> list[dict]:
        """Non-blocking read of all pending incoming messages."""
        msgs = []
        while True:
            try:
                msgs.append(self._inbound.get_nowait())
            except queue.Empty:
                break
        return msgs

    # ── Background Thread ────────────────────────────────────

    def _run_thread(self) -> None:
        """Entry point for the network thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_run())
        except Exception as e:
            self._error = str(e)
            self._ready.set()
        finally:
            self._loop.close()

    async def _async_run(self) -> None:
        """Async main loop: connect, create room, then send/recv."""
        try:
            async with websockets.connect(self._relay_url) as ws:
                # Create room
                await ws.send(pack({"t": MsgType.ROOM_CREATE}))
                resp = unpack(await ws.recv())
                if resp.get("t") != MsgType.ROOM_CREATE or "code" not in resp:
                    self._error = "Failed to create room"
                    self._ready.set()
                    return
                self._room_code = resp["code"]
                self._connected = True
                self._ready.set()

                # Send/recv loop
                recv_task = asyncio.create_task(self._recv_loop(ws))
                send_task = asyncio.create_task(self._send_loop(ws))
                stop_task = asyncio.create_task(self._wait_stop())

                done, pending = await asyncio.wait(
                    [recv_task, send_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            self._error = str(e)
            self._ready.set()
        finally:
            self._connected = False

    async def _recv_loop(self, ws) -> None:
        """Continuously receive messages from the relay."""
        try:
            async for raw in ws:
                msg = unpack(raw)
                msg_type = msg.get("t")
                if msg_type == MsgType.CLIENT_JOINED:
                    self._player_count = msg.get("players", self._player_count)
                elif msg_type == MsgType.LEAVE:
                    self._player_count = msg.get("players", max(0, self._player_count - 1))
                self._inbound.put(msg)
        except websockets.ConnectionClosed:
            pass

    async def _send_loop(self, ws) -> None:
        """Continuously drain the outbound queue and send messages."""
        while not self._stop_event.is_set():
            try:
                raw = self._outbound.get_nowait()
                await ws.send(raw)
            except queue.Empty:
                await asyncio.sleep(0.005)
            except websockets.ConnectionClosed:
                break

    async def _wait_stop(self) -> None:
        """Wait for the stop event."""
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)


class NetworkClient:
    """
    Manages the client side of a multiplayer session.

    The client joins a room on the relay server, receives snapshots
    from the host, and sends tool actions / settings changes.
    """

    def __init__(self, relay_url: str, room_code: str) -> None:
        self._relay_url = relay_url
        self._room_code = room_code
        self._outbound: queue.Queue[bytes] = queue.Queue()
        self._inbound: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._ready = threading.Event()
        self._error: str | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        """
        Start the network thread, connect to relay, and join the room.

        Raises:
            RuntimeError: If connection or room join fails.
        """
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)
        if self._error:
            raise RuntimeError(f"Failed to join: {self._error}")

    def stop(self) -> None:
        """Signal the network thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._connected = False

    def send_tool_action(self, msg: dict) -> None:
        """Enqueue a tool action message for the host."""
        self._outbound.put(pack(msg))

    def send_settings_change(self, msg: dict) -> None:
        """Enqueue a settings change message for the host."""
        self._outbound.put(pack(msg))

    def send(self, msg: dict) -> None:
        """Enqueue any message for the host."""
        self._outbound.put(pack(msg))

    def poll_incoming(self) -> list[dict]:
        """Non-blocking read of all pending incoming messages."""
        msgs = []
        while True:
            try:
                msgs.append(self._inbound.get_nowait())
            except queue.Empty:
                break
        return msgs

    # ── Background Thread ────────────────────────────────────

    def _run_thread(self) -> None:
        """Entry point for the network thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_run())
        except Exception as e:
            self._error = str(e)
            self._ready.set()
        finally:
            self._loop.close()

    async def _async_run(self) -> None:
        """Async main loop: connect, join room, then send/recv."""
        try:
            async with websockets.connect(self._relay_url) as ws:
                # Join room
                await ws.send(pack({"t": MsgType.JOIN, "code": self._room_code}))
                resp = unpack(await ws.recv())
                if resp.get("t") == "error":
                    self._error = resp.get("msg", "Unknown error")
                    self._ready.set()
                    return
                if resp.get("t") != MsgType.JOIN or not resp.get("ok"):
                    self._error = "Failed to join room"
                    self._ready.set()
                    return
                self._connected = True
                self._ready.set()

                # Send/recv loop
                recv_task = asyncio.create_task(self._recv_loop(ws))
                send_task = asyncio.create_task(self._send_loop(ws))
                stop_task = asyncio.create_task(self._wait_stop())

                done, pending = await asyncio.wait(
                    [recv_task, send_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            self._error = str(e)
            self._ready.set()
        finally:
            self._connected = False

    async def _recv_loop(self, ws) -> None:
        """Continuously receive messages from the relay."""
        try:
            async for raw in ws:
                msg = unpack(raw)
                self._inbound.put(msg)
        except websockets.ConnectionClosed:
            self._inbound.put({"t": MsgType.HOST_LEFT})

    async def _send_loop(self, ws) -> None:
        """Continuously drain the outbound queue and send messages."""
        while not self._stop_event.is_set():
            try:
                raw = self._outbound.get_nowait()
                await ws.send(raw)
            except queue.Empty:
                await asyncio.sleep(0.005)
            except websockets.ConnectionClosed:
                break

    async def _wait_stop(self) -> None:
        """Wait for the stop event."""
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)
