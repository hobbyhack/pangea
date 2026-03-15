"""Tests for the WebSocket relay server."""

import asyncio
import threading
import time

import pytest
import websockets

from pangea.protocol import MsgType, pack, unpack


@pytest.fixture(scope="module")
def relay_server():
    """Start a relay server in a background thread for testing."""
    from pangea.server import run_server

    loop = asyncio.new_event_loop()
    port = 18765  # Use a non-default port for testing

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve(port))

    async def _serve(p):
        from pangea.server import handler
        async with websockets.serve(handler, "127.0.0.1", p):
            await asyncio.Future()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    time.sleep(0.3)  # Give server time to start
    yield f"ws://127.0.0.1:{port}"
    loop.call_soon_threadsafe(loop.stop)


class TestRelayServer:
    def test_create_room(self, relay_server):
        async def _test():
            async with websockets.connect(relay_server) as ws:
                await ws.send(pack({"t": MsgType.ROOM_CREATE}))
                resp = unpack(await ws.recv())
                assert resp["t"] == MsgType.ROOM_CREATE
                assert "code" in resp
                assert len(resp["code"]) == 4

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_test())

    def test_join_room(self, relay_server):
        async def _test():
            async with websockets.connect(relay_server) as host_ws:
                # Host creates room
                await host_ws.send(pack({"t": MsgType.ROOM_CREATE}))
                resp = unpack(await host_ws.recv())
                code = resp["code"]

                # Client joins
                async with websockets.connect(relay_server) as client_ws:
                    await client_ws.send(pack({"t": MsgType.JOIN, "code": code}))

                    # Host should get CLIENT_JOINED
                    host_msg = unpack(await host_ws.recv())
                    assert host_msg["t"] == MsgType.CLIENT_JOINED

                    # Client should get JOIN confirmation
                    client_msg = unpack(await client_ws.recv())
                    assert client_msg["t"] == MsgType.JOIN
                    assert client_msg["ok"] is True

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_test())

    def test_join_invalid_room(self, relay_server):
        async def _test():
            async with websockets.connect(relay_server) as ws:
                await ws.send(pack({"t": MsgType.JOIN, "code": "ZZZZ"}))
                resp = unpack(await ws.recv())
                assert resp["t"] == "error"

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_test())

    def test_message_routing(self, relay_server):
        """Host messages go to clients, client messages go to host."""
        async def _test():
            async with websockets.connect(relay_server) as host_ws:
                await host_ws.send(pack({"t": MsgType.ROOM_CREATE}))
                resp = unpack(await host_ws.recv())
                code = resp["code"]

                async with websockets.connect(relay_server) as client_ws:
                    await client_ws.send(pack({"t": MsgType.JOIN, "code": code}))
                    # Drain join notifications
                    await host_ws.recv()  # CLIENT_JOINED
                    await client_ws.recv()  # JOIN ok

                    # Host sends snapshot → client receives it
                    test_snap = {"t": MsgType.SNAPSHOT, "data": [1, 2, 3]}
                    await host_ws.send(pack(test_snap))
                    client_msg = unpack(await client_ws.recv())
                    assert client_msg["t"] == MsgType.SNAPSHOT
                    assert client_msg["data"] == [1, 2, 3]

                    # Client sends tool action → host receives it
                    test_action = {"t": MsgType.TOOL_ACTION, "tool": "food", "x": 10, "y": 20}
                    await client_ws.send(pack(test_action))
                    host_msg = unpack(await host_ws.recv())
                    assert host_msg["t"] == MsgType.TOOL_ACTION
                    assert host_msg["tool"] == "food"

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_test())
