"""Tests for NetworkHost and NetworkClient classes."""

import asyncio
import threading
import time

import pytest
import websockets

from pangea.network import NetworkClient, NetworkHost
from pangea.protocol import MsgType, pack, unpack


@pytest.fixture(scope="module")
def relay_url():
    """Start a relay server for testing."""
    from pangea.server import handler

    loop = asyncio.new_event_loop()
    port = 18766

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve())

    async def _serve():
        async with websockets.serve(handler, "127.0.0.1", port):
            await asyncio.Future()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield f"ws://127.0.0.1:{port}"
    loop.call_soon_threadsafe(loop.stop)


class TestNetworkHost:
    def test_start_creates_room(self, relay_url):
        host = NetworkHost(relay_url)
        code = host.start()
        assert len(code) == 4
        assert host.connected
        assert host.player_count == 0
        host.stop()

    def test_stop_disconnects(self, relay_url):
        host = NetworkHost(relay_url)
        host.start()
        host.stop()
        assert not host.connected


class TestNetworkClient:
    def test_join_room(self, relay_url):
        host = NetworkHost(relay_url)
        code = host.start()

        client = NetworkClient(relay_url, code)
        client.start()
        assert client.connected

        time.sleep(0.2)
        assert host.player_count == 1

        client.stop()
        host.stop()

    def test_join_invalid_room(self, relay_url):
        client = NetworkClient(relay_url, "XXXX")
        with pytest.raises(RuntimeError):
            client.start()


class TestHostClientCommunication:
    def test_snapshot_broadcast(self, relay_url):
        host = NetworkHost(relay_url)
        code = host.start()
        client = NetworkClient(relay_url, code)
        client.start()
        time.sleep(0.2)

        # Host broadcasts a snapshot
        snap = {"t": MsgType.SNAPSHOT, "c": [], "f": [], "p": [],
                "et": 1.0, "st": 0.0, "dnt": 0.0, "gen": 1, "tb": 0, "td": 0}
        host.broadcast_snapshot(snap)
        time.sleep(0.3)

        msgs = client.poll_incoming()
        snapshot_msgs = [m for m in msgs if m.get("t") == MsgType.SNAPSHOT]
        assert len(snapshot_msgs) >= 1
        assert snapshot_msgs[0]["gen"] == 1

        client.stop()
        host.stop()

    def test_tool_action_to_host(self, relay_url):
        host = NetworkHost(relay_url)
        code = host.start()
        client = NetworkClient(relay_url, code)
        client.start()
        time.sleep(0.2)

        # Drain join notification
        host.poll_incoming()

        # Client sends tool action
        action = {"t": MsgType.TOOL_ACTION, "tool": "food", "x": 50.0, "y": 75.0}
        client.send_tool_action(action)
        time.sleep(0.3)

        msgs = host.poll_incoming()
        action_msgs = [m for m in msgs if m.get("t") == MsgType.TOOL_ACTION]
        assert len(action_msgs) >= 1
        assert action_msgs[0]["tool"] == "food"
        assert action_msgs[0]["x"] == 50.0

        client.stop()
        host.stop()
