"""Tests for GPU compute engine — CPU-GPU equivalence.

All tests use the Taichi CPU backend so they run in CI without a GPU.
Each test compares GPU kernel output against the existing Python implementation.
"""

import copy

import numpy as np
import pytest

# Gate all tests on taichi availability
ti = pytest.importorskip("taichi")

from pangea.compute import ComputeEngine
from pangea.config import (
    CARNIVORE_ATTACK_RANGE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
)
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.species import (
    SpeciesRegistry,
    default_carnivore,
    default_herbivore,
    default_scavenger,
)
from pangea.world import Food, World


# ── Helpers ─────────────────────────────────────────────────


def _make_registry() -> SpeciesRegistry:
    reg = SpeciesRegistry()
    reg.register(default_herbivore())
    reg.register(default_carnivore())
    reg.register(default_scavenger())
    return reg


def _make_creature(
    x: float = 100.0,
    y: float = 100.0,
    species_id: str = "herbivore",
    registry: SpeciesRegistry | None = None,
) -> Creature:
    dna = DNA.random(species_id)
    sp = (registry or _make_registry()).get(species_id)
    return Creature(dna, x, y, species=sp)


def _make_food(x: float = 50.0, y: float = 50.0, energy: float = 30.0) -> Food:
    return Food(x=x, y=y, energy=energy)


def _engine():
    """Create a ComputeEngine using CPU backend."""
    e = ComputeEngine(use_gpu=False)
    e.upload_species(_make_registry())
    return e


# ── Tests ───────────────────────────────────────────────────


class TestUploadDownloadRoundtrip:
    def test_creature_state_preserved(self):
        """Uploading then downloading creature state should produce identical values."""
        engine = _engine()
        reg = _make_registry()

        c = _make_creature(x=123.4, y=567.8, registry=reg)
        c.energy = 77.5
        c.food_eaten = 3
        c.feeds_count = 5
        c.age = 5.2
        c.heading = 1.5
        c.speed = 2.3
        c.under_attack = 0.7
        c.last_turn = 0.4
        c.distance_traveled = 42.0
        c.breed_cooldown = 3.0

        engine.upload_creatures([c])

        c2 = _make_creature(x=0, y=0, registry=reg)
        engine.download_creatures([c2])

        assert c2.x == pytest.approx(123.4, rel=1e-4)
        assert c2.y == pytest.approx(567.8, rel=1e-4)
        assert c2.energy == pytest.approx(77.5, rel=1e-4)
        assert c2.food_eaten == 3
        assert c2.feeds_count == 5
        assert c2.age == pytest.approx(5.2, rel=1e-4)
        assert c2.heading == pytest.approx(1.5, rel=1e-4)
        assert c2.speed == pytest.approx(2.3, rel=1e-4)
        assert c2.under_attack == pytest.approx(0.7, rel=1e-4)
        assert c2.last_turn == pytest.approx(0.4, rel=1e-4)
        assert c2.distance_traveled == pytest.approx(42.0, rel=1e-4)
        assert c2.breed_cooldown == pytest.approx(3.0, rel=1e-4)
        assert c2.alive is True


class TestBrainForward:
    def test_matches_cpu(self):
        """GPU brain forward should match NeuralNetwork.forward()."""
        engine = _engine()
        reg = _make_registry()

        np.random.seed(42)
        creatures = [_make_creature(registry=reg) for _ in range(5)]
        engine.upload_creatures(creatures)

        import pangea.compute as comp

        expected_outputs = []
        for i, c in enumerate(creatures):
            inputs = np.random.randn(NN_INPUT_SIZE).astype(np.float32)
            cpu_out = c.brain.forward(inputs)
            expected_outputs.append(cpu_out)
            for j in range(NN_INPUT_SIZE):
                comp._cinputs[i, j] = float(inputs[j])

        comp._k_brain_forward(len(creatures))

        for i, expected in enumerate(expected_outputs):
            for o in range(NN_OUTPUT_SIZE):
                gpu_val = float(comp._coutputs[i, o])
                assert gpu_val == pytest.approx(float(expected[o]), abs=1e-4), (
                    f"Creature {i}, output {o}: GPU={gpu_val}, CPU={expected[o]}"
                )


class TestSense:
    def test_matches_cpu_basic(self):
        """GPU sense kernel should produce same 12 sensor values as CPU."""
        engine = _engine()
        reg = _make_registry()

        np.random.seed(123)
        creatures = [
            _make_creature(x=200, y=200, registry=reg),
            _make_creature(x=250, y=200, species_id="carnivore", registry=reg),
        ]
        food = [_make_food(x=180, y=190)]
        world_w, world_h = 500.0, 500.0

        engine.upload_creatures(creatures)
        engine.upload_food(food)
        engine.upload_environment([], [])

        daylight = 1.0
        cpu_inputs_list = []
        for c in creatures:
            if not c.alive:
                continue
            sp = c.species
            night_mult = sp.settings.night_vision_multiplier if sp else 0.3
            vision_multiplier = night_mult + (1 - night_mult) * daylight
            cpu_inputs = c.sense(
                food, world_w, world_h, False,
                vision_multiplier=vision_multiplier,
                creatures=creatures,
                biome_speed=1.0,
                biome_danger=0.0,
            )
            cpu_inputs_list.append(cpu_inputs)

        import pangea.compute as comp

        nc = len(creatures)
        nf = len(food)
        comp._k_sense(nc, nf, 0, world_w, world_h, 0, daylight)

        for i, cpu_inputs in enumerate(cpu_inputs_list):
            for j in range(NN_INPUT_SIZE):
                gpu_val = float(comp._cinputs[i, j])
                cpu_val = float(cpu_inputs[j])
                assert gpu_val == pytest.approx(cpu_val, abs=1e-3), (
                    f"Creature {i}, sensor {j}: GPU={gpu_val}, CPU={cpu_val}"
                )


class TestPhysics:
    def test_position_update(self):
        """GPU physics should move creatures same as CPU update()."""
        engine = _engine()
        reg = _make_registry()

        np.random.seed(7)
        c = _make_creature(x=200, y=200, registry=reg)
        c.heading = 0.5
        c.speed = 2.0
        c.energy = 80.0

        c_cpu = copy.deepcopy(c)
        c_cpu.update(1.0 / 60.0, speed_multiplier=1.0)

        engine.upload_creatures([c])
        engine.upload_environment([], [])

        import pangea.compute as comp

        comp._cbiome_speed[0] = 1.0
        comp._cbiome_danger[0] = 0.0
        comp._k_physics(1, 0, 1.0 / 60.0, 500.0, 500.0, 0)
        engine.download_creatures([c])

        assert c.x == pytest.approx(c_cpu.x, abs=1e-3)
        assert c.y == pytest.approx(c_cpu.y, abs=1e-3)
        assert c.energy == pytest.approx(c_cpu.energy, abs=1e-3)

    def test_death_by_energy(self):
        """Creature with near-zero energy should die."""
        engine = _engine()
        reg = _make_registry()

        c = _make_creature(registry=reg)
        c.energy = 0.01
        c.speed = 5.0

        engine.upload_creatures([c])
        engine.upload_environment([], [])

        import pangea.compute as comp

        comp._cbiome_speed[0] = 1.0
        comp._cbiome_danger[0] = 0.0
        comp._k_physics(1, 0, 1.0 / 60.0, 500.0, 500.0, 0)
        engine.download_creatures([c])

        assert c.alive is False
        assert c.energy == 0.0


class TestCombat:
    def test_damage_applied(self):
        """Carnivore attacking herbivore should drain victim energy."""
        engine = _engine()
        reg = _make_registry()

        carnivore = _make_creature(x=100, y=100, species_id="carnivore", registry=reg)
        herbivore = _make_creature(x=100, y=103, species_id="herbivore", registry=reg)
        carnivore.energy = 80.0
        herbivore.energy = 80.0

        engine.upload_creatures([carnivore, herbivore])

        import pangea.compute as comp

        dt = 1.0 / 60.0
        comp._k_combat(2, dt, CARNIVORE_ATTACK_RANGE)
        engine.download_creatures([carnivore, herbivore])

        assert herbivore.energy < 80.0
        assert herbivore.under_attack == pytest.approx(1.0)
        assert carnivore.energy > 80.0


class TestCollisions:
    def test_creature_eats_food(self):
        """Creature on top of food should eat it."""
        engine = _engine()
        reg = _make_registry()

        c = _make_creature(x=50, y=50, registry=reg)
        c.energy = 50.0
        food = [_make_food(x=50, y=50, energy=30.0)]

        engine.upload_creatures([c])
        engine.upload_food(food)

        import pangea.compute as comp

        comp._k_collisions(1, 1, 500.0, 500.0, 0)
        engine.download_creatures([c])

        herb_sp = reg.get("herbivore")
        expected_gain = 30.0 * herb_sp.plant_food_multiplier
        assert c.energy == pytest.approx(50.0 + expected_gain, abs=1e-3)
        assert c.food_eaten == 1
        assert c.feeds_count == 1

        result = engine.download_food_compacted(food)
        assert len(result) == 0

    def test_no_double_eat(self):
        """Two creatures on same food — only one gets the energy."""
        engine = _engine()
        reg = _make_registry()

        c1 = _make_creature(x=50, y=50, registry=reg)
        c2 = _make_creature(x=50, y=50, registry=reg)
        c1.energy = 50.0
        c2.energy = 50.0
        food = [_make_food(x=50, y=50, energy=30.0)]

        engine.upload_creatures([c1, c2])
        engine.upload_food(food)

        import pangea.compute as comp

        comp._k_collisions(2, 1, 500.0, 500.0, 0)
        engine.download_creatures([c1, c2])

        total_food_eaten = c1.food_eaten + c2.food_eaten
        assert total_food_eaten == 1
        total_feeds = c1.feeds_count + c2.feeds_count
        assert total_feeds == 1


class TestFoodCompaction:
    def test_eaten_food_removed(self):
        """download_food_compacted should exclude eaten food items."""
        engine = _engine()

        food = [
            _make_food(x=10, y=10),
            _make_food(x=50, y=50),
            _make_food(x=90, y=90),
        ]
        engine.upload_food(food)

        import pangea.compute as comp

        comp._falive[1] = 0

        result = engine.download_food_compacted(food)
        assert len(result) == 2
        assert result[0].x == pytest.approx(10.0)
        assert result[1].x == pytest.approx(90.0)


class TestFullFrame:
    def test_gpu_vs_cpu_one_frame(self):
        """Run one full frame on both CPU and GPU, verify similar outcomes."""
        reg = _make_registry()
        np.random.seed(999)

        creatures_cpu = [
            _make_creature(x=200, y=200, registry=reg),
            _make_creature(x=300, y=300, species_id="carnivore", registry=reg),
        ]
        creatures_gpu = copy.deepcopy(creatures_cpu)

        food_cpu = [_make_food(x=190, y=195)]
        food_gpu = [_make_food(x=190, y=195)]

        from pangea.settings import SimSettings

        settings = SimSettings()
        settings.species_registry = reg
        settings.initial_food_count = 0
        settings.hazard_count = 0
        settings.biomes_enabled = False
        settings.food_spawn_rate = 0.0

        world_cpu = World(creatures_cpu, width=500, height=500,
                          settings=settings.copy(), use_gpu=False)
        world_cpu.food = food_cpu
        world_cpu.freeplay = True

        world_gpu = World(creatures_gpu, width=500, height=500,
                          settings=settings.copy(), use_gpu=True)
        world_gpu.food = food_gpu
        world_gpu.freeplay = True

        dt = 1.0 / 60.0
        world_cpu.update(dt)
        world_gpu.update(dt)

        for i in range(len(creatures_cpu)):
            cc = creatures_cpu[i]
            cg = creatures_gpu[i]
            assert cg.x == pytest.approx(cc.x, abs=0.5), f"Creature {i} x"
            assert cg.y == pytest.approx(cc.y, abs=0.5), f"Creature {i} y"
            assert cg.alive == cc.alive, f"Creature {i} alive"
            assert cg.energy == pytest.approx(cc.energy, abs=1.0), f"Creature {i} energy"
