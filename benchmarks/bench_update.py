#!/usr/bin/env python3
"""
Benchmark: CPU vs GPU frame times for World.update().

Usage:
    python benchmarks/bench_update.py [--gpu]

Runs the simulation loop for N frames at various creature/food counts
and reports per-frame timing.
"""

import argparse
import random
import time

import numpy as np

from pangea.creature import Creature
from pangea.dna import DNA
from pangea.settings import SimSettings
from pangea.species import default_registry
from pangea.world import Food, World


def make_world(
    n_creatures: int,
    n_food: int,
    use_gpu: bool = False,
) -> World:
    """Create a test world with the specified population."""
    registry = default_registry()
    settings = SimSettings()
    settings.species_registry = registry
    settings.initial_food_count = 0
    settings.hazard_count = 2
    settings.biomes_enabled = True
    settings.biome_count = 3
    settings.food_spawn_rate = 0.0  # No CPU spawning during benchmark
    settings.world_width = 1024
    settings.world_height = 768

    species_ids = list(registry.ids())
    creatures = []
    for i in range(n_creatures):
        sid = species_ids[i % len(species_ids)]
        dna = DNA.random(sid)
        x = random.uniform(50, settings.world_width - 50)
        y = random.uniform(50, settings.world_height - 50)
        sp = registry.get(sid)
        creatures.append(Creature(dna, x, y, species=sp))

    world = World(creatures, settings=settings, use_gpu=use_gpu)

    # Add food
    for _ in range(n_food):
        x = random.uniform(10, settings.world_width - 10)
        y = random.uniform(10, settings.world_height - 10)
        world.food.append(Food(x=x, y=y))

    # Upload food to GPU if needed
    if use_gpu and world._compute is not None:
        world._compute.upload_food(world.food)

    return world


def benchmark(n_creatures: int, n_food: int, use_gpu: bool, n_frames: int = 100) -> dict:
    """Run benchmark and return timing results."""
    random.seed(42)
    np.random.seed(42)

    world = make_world(n_creatures, n_food, use_gpu=use_gpu)
    dt = 1.0 / 60.0

    # Warmup (1 frame for JIT compilation on GPU)
    world.update(dt)

    # Timed run
    start = time.perf_counter()
    for _ in range(n_frames):
        world.update(dt)
    elapsed = time.perf_counter() - start

    per_frame_ms = (elapsed / n_frames) * 1000
    fps = n_frames / elapsed

    return {
        "n_creatures": n_creatures,
        "n_food": n_food,
        "backend": "GPU" if use_gpu else "CPU",
        "frames": n_frames,
        "total_s": elapsed,
        "per_frame_ms": per_frame_ms,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU simulation")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (Taichi)")
    parser.add_argument("--frames", type=int, default=100, help="Frames per scenario")
    args = parser.parse_args()

    scenarios = [
        (50, 100),
        (100, 200),
        (200, 500),
        (500, 1000),
    ]

    backends = ["GPU", "CPU"] if args.gpu else ["CPU"]

    print(f"{'Backend':<8} {'Creatures':>10} {'Food':>6} {'Frames':>7} "
          f"{'Total(s)':>9} {'ms/frame':>9} {'FPS':>7}")
    print("-" * 65)

    for backend in backends:
        use_gpu = (backend == "GPU")
        for n_c, n_f in scenarios:
            try:
                result = benchmark(n_c, n_f, use_gpu, n_frames=args.frames)
                print(f"{result['backend']:<8} {result['n_creatures']:>10} "
                      f"{result['n_food']:>6} {result['frames']:>7} "
                      f"{result['total_s']:>9.3f} {result['per_frame_ms']:>9.2f} "
                      f"{result['fps']:>7.1f}")
            except Exception as e:
                print(f"{backend:<8} {n_c:>10} {n_f:>6} {'FAILED':>7}  {e}")


if __name__ == "__main__":
    main()
