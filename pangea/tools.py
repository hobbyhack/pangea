"""
Player Tools -- interactive tools for influencing evolution.
============================================================
During Isolation Mode, the player can use these tools to change
the environment and steer evolutionary pressure:

    1. Place Food    -- click to drop food clusters
    2. Food Drought  -- temporarily stop all food spawning
    3. Poison Zone   -- place a zone that drains energy
    4. Barrier Wall  -- place walls that block movement
    5. Bounty Zone   -- place a zone that boosts food spawning

Each tool is selected from the toolbar and activated with mouse clicks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# -- Tool Types ---------------------------------------------------------------

TOOL_NONE = "none"
TOOL_PLACE_FOOD = "food"
TOOL_DROUGHT = "drought"
TOOL_POISON = "poison"
TOOL_BARRIER = "barrier"
TOOL_BOUNTY = "bounty"

TOOL_LIST = [TOOL_NONE, TOOL_PLACE_FOOD, TOOL_DROUGHT, TOOL_POISON, TOOL_BARRIER, TOOL_BOUNTY]

TOOL_LABELS = {
    TOOL_NONE: "Select",
    TOOL_PLACE_FOOD: "Food",
    TOOL_DROUGHT: "Drought",
    TOOL_POISON: "Poison",
    TOOL_BARRIER: "Barrier",
    TOOL_BOUNTY: "Bounty",
}

TOOL_COLORS = {
    TOOL_NONE: (150, 150, 150),
    TOOL_PLACE_FOOD: (50, 205, 50),
    TOOL_DROUGHT: (200, 160, 50),
    TOOL_POISON: (180, 40, 180),
    TOOL_BARRIER: (140, 140, 160),
    TOOL_BOUNTY: (50, 200, 200),
}

TOOL_DESCRIPTIONS = {
    TOOL_NONE: "No tool active",
    TOOL_PLACE_FOOD: "Click to place food clusters",
    TOOL_DROUGHT: "Toggle: stop natural food spawning",
    TOOL_POISON: "Click to place energy-drain zone",
    TOOL_BARRIER: "Click & drag to place wall segments",
    TOOL_BOUNTY: "Click to place food-boosting zone",
}


# -- Zone Effects -------------------------------------------------------------

@dataclass
class Zone:
    """A circular zone placed by the player that affects creatures."""

    x: float
    y: float
    radius: float = 60.0
    zone_type: str = ""       # "poison" or "bounty"
    strength: float = 1.0     # effect intensity
    lifetime: float = 15.0    # seconds before it fades
    age: float = 0.0

    @property
    def alive(self) -> bool:
        return self.age < self.lifetime

    @property
    def opacity(self) -> float:
        """Fade out in the last 3 seconds."""
        remaining = self.lifetime - self.age
        if remaining < 3.0:
            return max(0.0, remaining / 3.0)
        return 1.0


@dataclass
class Barrier:
    """A wall segment placed by the player."""

    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 6.0
    lifetime: float = 30.0
    age: float = 0.0

    @property
    def alive(self) -> bool:
        return self.age < self.lifetime

    @property
    def opacity(self) -> float:
        remaining = self.lifetime - self.age
        if remaining < 3.0:
            return max(0.0, remaining / 3.0)
        return 1.0


# -- Player Tool State -------------------------------------------------------

class PlayerTools:
    """
    Manages the player's active tool and placed effects.

    The simulation checks these each frame to apply environmental effects.
    """

    def __init__(self) -> None:
        self.active_tool: str = TOOL_NONE
        self.drought_active: bool = False
        self.zones: list[Zone] = []
        self.barriers: list[Barrier] = []

        # For barrier drag placement
        self._barrier_start: tuple[float, float] | None = None

    def select_tool(self, tool: str) -> None:
        """Select a tool from the toolbar."""
        if tool == TOOL_DROUGHT:
            # Drought is a toggle
            self.drought_active = not self.drought_active
            self.active_tool = TOOL_NONE
        else:
            self.active_tool = tool

    def on_mouse_down(self, x: float, y: float) -> list[tuple[float, float]]:
        """
        Handle mouse press at (x, y). Returns list of food positions
        to spawn (only for TOOL_PLACE_FOOD).
        """
        food_positions: list[tuple[float, float]] = []

        if self.active_tool == TOOL_PLACE_FOOD:
            # Scatter a cluster of 5-8 food around the click point
            count = random.randint(5, 8)
            for _ in range(count):
                fx = x + random.gauss(0, 25)
                fy = y + random.gauss(0, 25)
                food_positions.append((fx, fy))

        elif self.active_tool == TOOL_POISON:
            self.zones.append(Zone(x=x, y=y, zone_type="poison", radius=60, lifetime=15))

        elif self.active_tool == TOOL_BOUNTY:
            self.zones.append(Zone(x=x, y=y, zone_type="bounty", radius=80, lifetime=20))

        elif self.active_tool == TOOL_BARRIER:
            self._barrier_start = (x, y)

        return food_positions

    def on_mouse_up(self, x: float, y: float) -> None:
        """Handle mouse release (for barrier drag)."""
        if self.active_tool == TOOL_BARRIER and self._barrier_start is not None:
            sx, sy = self._barrier_start
            dist = math.sqrt((x - sx) ** 2 + (y - sy) ** 2)
            if dist > 15:  # minimum barrier length
                self.barriers.append(Barrier(x1=sx, y1=sy, x2=x, y2=y))
            self._barrier_start = None

    def update(self, dt: float) -> None:
        """Age all zones and barriers, remove expired ones."""
        for zone in self.zones:
            zone.age += dt
        self.zones = [z for z in self.zones if z.alive]

        for barrier in self.barriers:
            barrier.age += dt
        self.barriers = [b for b in self.barriers if b.alive]

    def get_energy_modifier(self, x: float, y: float) -> float:
        """
        Get the energy drain/boost at a position from active zones.

        Returns:
            Negative = energy drain per second, Positive = energy boost.
        """
        modifier = 0.0
        for zone in self.zones:
            dx = x - zone.x
            dy = y - zone.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < zone.radius:
                # Stronger effect closer to center
                intensity = (1.0 - dist / zone.radius) * zone.strength * zone.opacity
                if zone.zone_type == "poison":
                    modifier -= intensity * 2.0  # drain energy
                elif zone.zone_type == "bounty":
                    modifier += intensity * 0.5  # slight energy boost
        return modifier

    def check_barrier_collision(self, x: float, y: float, radius: float) -> tuple[float, float] | None:
        """
        Check if a creature at (x, y) with given radius hits any barrier.

        Returns:
            Push-back vector (dx, dy) if colliding, None otherwise.
        """
        for barrier in self.barriers:
            if not barrier.alive:
                continue

            # Point-to-line-segment distance
            bx = barrier.x2 - barrier.x1
            by = barrier.y2 - barrier.y1
            length_sq = bx * bx + by * by
            if length_sq < 1:
                continue

            t = max(0, min(1, ((x - barrier.x1) * bx + (y - barrier.y1) * by) / length_sq))
            closest_x = barrier.x1 + t * bx
            closest_y = barrier.y1 + t * by

            dx = x - closest_x
            dy = y - closest_y
            dist = math.sqrt(dx * dx + dy * dy)

            hit_radius = radius + barrier.thickness / 2
            if dist < hit_radius and dist > 0:
                # Push creature away from barrier
                push = (hit_radius - dist)
                return (dx / dist * push, dy / dist * push)

        return None

    def get_food_spawn_multiplier(self) -> float:
        """
        Returns the food spawn rate multiplier.

        0.0 if drought is active, 1.0 normally, >1.0 if bounty zones
        would boost spawning (handled separately in world update).
        """
        if self.drought_active:
            return 0.0
        return 1.0

    def reset(self) -> None:
        """Clear all placed effects (for new generation or restart)."""
        self.zones.clear()
        self.barriers.clear()
        self.drought_active = False
        self._barrier_start = None
