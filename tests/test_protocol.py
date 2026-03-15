"""Tests for the network protocol — serialization round-trips."""

import math
import random

from pangea.creature import Creature
from pangea.dna import DNA
from pangea.protocol import (
    MsgType,
    apply_full_state,
    apply_snapshot,
    full_state_from_world,
    generation_end_msg,
    pack,
    settings_change_msg,
    snapshot_from_world,
    tool_action_msg,
    unpack,
)
from pangea.settings import SimSettings
from pangea.tools import PlayerTools, Zone, Barrier
from pangea.world import Food, World


def _make_world(n_creatures=5, n_food=10) -> tuple[World, SimSettings, PlayerTools]:
    """Create a small test world."""
    settings = SimSettings()
    settings.world_width = 400
    settings.world_height = 300
    settings.hazard_count = 0
    settings.biome_count = 0
    settings.predator_count = 0
    settings.initial_food_count = 0

    tools = PlayerTools()
    dna_list = [DNA.random() for _ in range(n_creatures)]
    creatures = [
        Creature(dna, random.uniform(10, 390), random.uniform(10, 290))
        for dna in dna_list
    ]
    world = World(creatures, settings=settings, tools=tools)

    # Add some food
    for _ in range(n_food):
        world.food.append(Food(
            x=random.uniform(10, 390), y=random.uniform(10, 290),
            energy=30.0, radius=4.0, age=1.0, lifetime=15.0, is_corpse=False,
        ))

    world.elapsed_time = 5.5
    world.season_time = 2.3
    world.day_night_time = 10.0
    world.generation = 3
    world.total_births = 7
    world.total_deaths = 4

    return world, settings, tools


class TestPackUnpack:
    def test_round_trip(self):
        msg = {"t": MsgType.SNAPSHOT, "data": [1, 2, 3], "nested": {"a": 1.5}}
        result = unpack(pack(msg))
        assert result["t"] == MsgType.SNAPSHOT
        assert result["data"] == [1, 2, 3]
        assert result["nested"]["a"] == 1.5

    def test_binary(self):
        data = pack({"t": "test"})
        assert isinstance(data, bytes)


class TestSnapshot:
    def test_snapshot_round_trip(self):
        world, settings, tools = _make_world(n_creatures=10, n_food=20)
        snap = snapshot_from_world(world)

        assert snap["t"] == MsgType.SNAPSHOT
        assert len(snap["c"]) == 10
        assert len(snap["f"]) == 20
        assert snap["et"] == world.elapsed_time
        assert snap["gen"] == 3

    def test_apply_snapshot_preserves_positions(self):
        world, settings, tools = _make_world(n_creatures=5, n_food=8)

        # Record original positions
        orig_positions = [(c.x, c.y) for c in world.creatures]

        snap = snapshot_from_world(world)

        # Create a fresh world and apply the snapshot
        world2 = World(
            [Creature(DNA.random(), 0, 0) for _ in range(5)],
            settings=settings, tools=tools,
        )
        apply_snapshot(world2, snap)

        for i in range(5):
            assert abs(world2.creatures[i].x - orig_positions[i][0]) < 0.001
            assert abs(world2.creatures[i].y - orig_positions[i][1]) < 0.001

    def test_apply_snapshot_creature_count_change(self):
        world, settings, tools = _make_world(n_creatures=5, n_food=3)
        snap = snapshot_from_world(world)

        # World with fewer creatures
        world2 = World(
            [Creature(DNA.random(), 0, 0) for _ in range(3)],
            settings=settings, tools=tools,
        )
        apply_snapshot(world2, snap)
        assert len(world2.creatures) == 5

        # World with more creatures
        world3 = World(
            [Creature(DNA.random(), 0, 0) for _ in range(8)],
            settings=settings, tools=tools,
        )
        apply_snapshot(world3, snap)
        assert len(world3.creatures) == 5

    def test_apply_snapshot_food(self):
        world, settings, tools = _make_world(n_creatures=2, n_food=6)
        snap = snapshot_from_world(world)

        world2 = World(
            [Creature(DNA.random(), 0, 0) for _ in range(2)],
            settings=settings, tools=tools,
        )
        apply_snapshot(world2, snap)
        assert len(world2.food) == 6

    def test_apply_snapshot_timers(self):
        world, settings, tools = _make_world()
        snap = snapshot_from_world(world)

        world2 = World(
            [Creature(DNA.random(), 0, 0) for _ in range(5)],
            settings=settings, tools=tools,
        )
        apply_snapshot(world2, snap)
        assert world2.elapsed_time == 5.5
        assert world2.generation == 3
        assert world2.total_births == 7
        assert world2.total_deaths == 4

    def test_snapshot_msgpack_round_trip(self):
        """Full pack/unpack cycle through msgpack."""
        world, settings, tools = _make_world(n_creatures=3, n_food=5)
        snap = snapshot_from_world(world)
        data = pack(snap)
        restored = unpack(data)

        world2 = World(
            [Creature(DNA.random(), 0, 0) for _ in range(3)],
            settings=settings, tools=tools,
        )
        apply_snapshot(world2, restored)
        assert len(world2.creatures) == 3
        assert len(world2.food) == 5


class TestFullState:
    def test_full_state_round_trip(self):
        world, settings, tools = _make_world(n_creatures=5, n_food=8)

        # Add some tools state
        tools.zones.append(Zone(x=100, y=100, radius=60, zone_type="poison",
                                strength=1.0, lifetime=15, age=3))
        tools.barriers.append(Barrier(x1=50, y1=50, x2=150, y2=150,
                                      thickness=6, lifetime=30, age=1))
        tools.drought_active = True

        gen_history = [{"gen": 1, "avg_speed": 20.0}]

        full = full_state_from_world(
            world, settings, tools, "isolation", 3, gen_history,
        )
        assert full["t"] == MsgType.FULL_STATE

        result = apply_full_state(full)
        world2, settings2, tools2, mode, gen, history = result

        assert mode == "isolation"
        assert gen == 3
        assert len(world2.creatures) == 5
        assert len(world2.food) == 8
        assert tools2.drought_active is True
        assert len(tools2.zones) == 1
        assert len(tools2.barriers) == 1
        assert tools2.zones[0].zone_type == "poison"
        assert len(history) == 1
        assert settings2.world_width == 400

    def test_full_state_msgpack_round_trip(self):
        world, settings, tools = _make_world(n_creatures=3, n_food=4)
        full = full_state_from_world(world, settings, tools, "freeplay", 1)
        data = pack(full)
        restored = unpack(data)
        result = apply_full_state(restored)
        world2, _, _, mode, _, _ = result
        assert mode == "freeplay"
        assert len(world2.creatures) == 3


class TestMessages:
    def test_tool_action_msg(self):
        msg = tool_action_msg("food", 100.5, 200.3)
        assert msg["t"] == MsgType.TOOL_ACTION
        assert msg["tool"] == "food"
        assert msg["x"] == 100.5
        assert "x2" not in msg

    def test_tool_action_with_coords(self):
        msg = tool_action_msg("barrier", 10, 20, x2=30, y2=40)
        assert msg["x2"] == 30
        assert msg["y2"] == 40

    def test_settings_change_msg(self):
        msg = settings_change_msg({"mutation_rate": 0.2, "food_spawn_rate": 1.0})
        assert msg["t"] == MsgType.SETTINGS_CHANGE
        assert msg["changes"]["mutation_rate"] == 0.2

    def test_generation_end_msg(self):
        msg = generation_end_msg(5, [{"speed": 20}], {"avg_food": 3.5})
        assert msg["t"] == MsgType.GENERATION_END
        assert msg["generation"] == 5
        assert len(msg["top_dna"]) == 1
