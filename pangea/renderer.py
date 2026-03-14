"""
Renderer — all pygame drawing logic with rich visuals.
============================================================
Draws the world, creatures, food, biomes, zones, barriers, toolbar,
HUD overlay, particles, and glow effects.
"""

from __future__ import annotations

import math
import random

import pygame

from pangea.config import (
    BASE_ENERGY,
    COLOR_BACKGROUND,
    COLOR_BIOME_ROAD,
    COLOR_BIOME_WATER,
    COLOR_FOOD,
    COLOR_HAZARD_COLD,
    COLOR_HAZARD_LAVA,
    COLOR_HUD_TEXT,
    COLOR_LINEAGE_A,
    COLOR_LINEAGE_B,
    COLOR_PREDATOR,
)
import pangea.config as config
from pangea.creature import Creature
from pangea.tools import (
    TOOL_BARRIER,
    TOOL_COLORS,
    TOOL_DESCRIPTIONS,
    TOOL_LABELS,
    TOOL_LIST,
    PlayerTools,
)
from pangea.world import World


# ── Particle System ─────────────────────────────────────────

class Particle:
    """A small visual particle for effects."""

    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(
        self, x: float, y: float, vx: float, vy: float,
        life: float, color: tuple[int, int, int], size: float = 2.0,
    ) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def update(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    @property
    def alpha(self) -> float:
        return max(0.0, self.life / self.max_life)


class ParticleSystem:
    """Manages visual particles for eating, death, and zone effects."""

    def __init__(self) -> None:
        self.particles: list[Particle] = []

    def emit_eat(self, x: float, y: float) -> None:
        """Burst of green/yellow particles when a creature eats."""
        for _ in range(6):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(30, 80)
            color = random.choice([
                (80, 255, 80), (120, 255, 50), (200, 255, 50), (255, 255, 80),
            ])
            self.particles.append(Particle(
                x, y,
                math.cos(angle) * speed, math.sin(angle) * speed,
                life=random.uniform(0.3, 0.6), color=color, size=random.uniform(1.5, 3.0),
            ))

    def emit_death(self, x: float, y: float, color: tuple[int, int, int]) -> None:
        """Fading ring of particles when a creature dies."""
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(15, 50)
            # Desaturate the creature's color toward gray
            dc = tuple(min(255, c + 60) for c in color)
            self.particles.append(Particle(
                x, y,
                math.cos(angle) * speed, math.sin(angle) * speed,
                life=random.uniform(0.5, 1.0), color=dc, size=random.uniform(1.0, 2.5),
            ))

    def emit_zone(self, x: float, y: float, zone_type: str) -> None:
        """Occasional ambient particles from active zones."""
        color = (180, 40, 180) if zone_type == "poison" else (50, 200, 200)
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, 40)
        px = x + math.cos(angle) * dist
        py = y + math.sin(angle) * dist
        self.particles.append(Particle(
            px, py, random.gauss(0, 5), random.uniform(-15, -30),
            life=random.uniform(0.5, 1.2), color=color, size=random.uniform(1.0, 2.0),
        ))

    def update(self, dt: float) -> None:
        self.particles = [p for p in self.particles if p.update(dt)]

    def draw(self, surface: pygame.Surface) -> None:
        for p in self.particles:
            alpha = p.alpha
            size = max(1, int(p.size * alpha))
            color = tuple(int(c * alpha) for c in p.color)
            pygame.draw.circle(surface, color, (int(p.x), int(p.y)), size)


# ── Main Renderer ───────────────────────────────────────────

class Renderer:
    """Handles all pygame rendering for the simulation."""

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_large = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 48, bold=True)
        self.font_toolbar = pygame.font.SysFont("consolas", 12)

        self.particles = ParticleSystem()
        self.frame = 0

        # Track previous alive state for death particles
        self._prev_alive: dict[int, bool] = {}
        # Track previous food count for eat particles
        self._prev_food_eaten: dict[int, int] = {}

        # Pre-render glow surfaces for performance
        self._food_glow = self._make_glow_surface(12, (30, 140, 30, 60))
        self._poison_glow = self._make_glow_surface(80, (100, 20, 100, 30))
        self._bounty_glow = self._make_glow_surface(100, (20, 80, 80, 25))

    # ── Glow Helpers ─────────────────────────────────────────

    def _make_glow_surface(
        self, radius: int, color: tuple[int, int, int, int],
    ) -> pygame.Surface:
        """Create a soft radial glow surface."""
        size = radius * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            alpha = int(color[3] * (r / radius))
            c = (color[0], color[1], color[2], alpha)
            pygame.draw.circle(surf, c, (radius, radius), r)
        return surf

    # ── Main Draw ────────────────────────────────────────────

    def draw(
        self,
        world: World,
        mode: str = "isolation",
        paused: bool = False,
        tools: PlayerTools | None = None,
        show_toolbar: bool = False,
    ) -> None:
        """Draw the complete frame."""
        self.frame += 1

        # Background with subtle gradient
        self._draw_background()

        # Biomes (under everything else)
        self._draw_biomes(world)

        # Zones (under everything)
        if tools:
            self._draw_zones(tools)

        # Barriers
        if tools:
            self._draw_barriers(tools)

        # Hazard zones
        self._draw_hazards(world)

        # Food with glow
        self._draw_food(world)

        # Creatures
        self._draw_creatures(world, mode)

        # Predators
        self._draw_predators(world)

        # Update particle tracking
        self._track_events(world, mode)

        # Particles (on top of creatures)
        self.particles.update(1 / 60)
        self.particles.draw(self.surface)

        # Zone ambient particles
        if tools:
            for zone in tools.zones:
                if random.random() < 0.15:
                    self.particles.emit_zone(zone.x, zone.y, zone.zone_type)

        # Day/night darkness overlay (before HUD so HUD stays readable)
        self._draw_darkness_overlay(world)

        # HUD
        self._draw_hud(world, mode, tools)

        # Toolbar
        if show_toolbar and tools:
            self._draw_toolbar(tools)

        if paused:
            self._draw_pause_indicator()

    # ── Day/Night Overlay ─────────────────────────────────────

    def _draw_darkness_overlay(self, world: World) -> None:
        """Draw a semi-transparent dark overlay based on the day/night cycle."""
        darkness = 1.0 - world.daylight_factor  # 0 at full day, 1 at full night
        alpha = int(darkness * 180)  # max alpha ~180
        if alpha > 0:
            w = self.surface.get_width()
            h = self.surface.get_height()
            # Reuse a cached surface to avoid per-frame allocation
            if not hasattr(self, "_night_overlay") or self._night_overlay.get_size() != (w, h):
                self._night_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            self._night_overlay.fill((0, 0, 20, alpha))
            self.surface.blit(self._night_overlay, (0, 0))

    @staticmethod
    def _time_of_day_label(daylight_factor: float) -> str:
        """Return a label for the current time of day."""
        if daylight_factor >= 0.75:
            return "Day"
        elif daylight_factor >= 0.25:
            return "Dawn/Dusk"
        else:
            return "Night"

    # ── Background ───────────────────────────────────────────

    def _draw_background(self) -> None:
        """Draw a flat dark background."""
        self.surface.fill(COLOR_BACKGROUND)

    # ── Biomes ────────────────────────────────────────────────

    def _draw_biomes(self, world: World) -> None:
        """Draw biome regions as semi-transparent filled circles."""
        for biome in world.biomes:
            radius = int(biome.radius)
            cx, cy = int(biome.x), int(biome.y)

            if biome.biome_type == "water":
                color = (*COLOR_BIOME_WATER, 50)
            else:  # road
                color = (*COLOR_BIOME_ROAD, 40)

            biome_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(biome_surf, color, (radius, radius), radius)
            # Subtle border
            border_color = (*color[:3], min(255, color[3] + 30))
            pygame.draw.circle(biome_surf, border_color, (radius, radius), radius, 2)
            self.surface.blit(biome_surf, (cx - radius, cy - radius))

    # ── Food ─────────────────────────────────────────────────

    def _draw_food(self, world: World) -> None:
        """Draw food items with a soft glow, fading as they age."""
        glow = self._food_glow
        gw = glow.get_width() // 2
        for food in world.food:
            fx, fy = int(food.x), int(food.y)

            # Compute fade factor: 1.0 = fresh, dims toward 0.3 as food expires
            if food.lifetime > 0:
                freshness = max(0.3, 1.0 - food.age / food.lifetime * 0.7)
            else:
                freshness = 1.0

            # Glow (reduce intensity for old food)
            if freshness > 0.5:
                self.surface.blit(glow, (fx - gw, fy - gw), special_flags=pygame.BLEND_ADD)

            # Core — lerp from bright green toward dim green
            core_color = (
                int(60 * freshness),
                int(220 * freshness),
                int(60 * freshness),
            )
            pygame.draw.circle(self.surface, core_color, (fx, fy), max(2, int(food.radius)))

            # Bright center
            center_color = (
                int(140 * freshness),
                int(255 * freshness),
                int(140 * freshness),
            )
            pygame.draw.circle(self.surface, center_color, (fx, fy), max(1, int(food.radius) - 1))

    # ── Zones ────────────────────────────────────────────────

    def _draw_zones(self, tools: PlayerTools) -> None:
        """Draw player-placed zones with pulsing glow."""
        for zone in tools.zones:
            opacity = zone.opacity
            radius = int(zone.radius)
            cx, cy = int(zone.x), int(zone.y)

            if zone.zone_type == "poison":
                # Purple pulsing ring
                pulse = 0.7 + 0.3 * math.sin(self.frame * 0.1)
                alpha = int(40 * opacity * pulse)
                color = (140, 30, 140, alpha)
                ring_color = (180, 50, 180, int(80 * opacity))
            else:  # bounty
                pulse = 0.7 + 0.3 * math.sin(self.frame * 0.08)
                alpha = int(35 * opacity * pulse)
                color = (30, 130, 130, alpha)
                ring_color = (50, 200, 200, int(70 * opacity))

            # Filled circle
            zone_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(zone_surf, color, (radius, radius), radius)
            pygame.draw.circle(zone_surf, ring_color, (radius, radius), radius, 2)
            self.surface.blit(zone_surf, (cx - radius, cy - radius))

            # Label
            label = "POISON" if zone.zone_type == "poison" else "BOUNTY"
            label_color = (180, 60, 180) if zone.zone_type == "poison" else (60, 180, 180)
            text = self.font_small.render(label, True, label_color)
            text.set_alpha(int(160 * opacity))
            self.surface.blit(text, text.get_rect(center=(cx, cy)))

    # ── Barriers ─────────────────────────────────────────────

    def _draw_barriers(self, tools: PlayerTools) -> None:
        """Draw player-placed barrier walls."""
        for barrier in tools.barriers:
            opacity = barrier.opacity
            color = (
                int(160 * opacity),
                int(160 * opacity),
                int(180 * opacity),
            )
            glow_color = (
                int(80 * opacity),
                int(80 * opacity),
                int(100 * opacity),
            )
            # Glow line (thicker)
            pygame.draw.line(
                self.surface, glow_color,
                (int(barrier.x1), int(barrier.y1)),
                (int(barrier.x2), int(barrier.y2)),
                int(barrier.thickness + 4),
            )
            # Core line
            pygame.draw.line(
                self.surface, color,
                (int(barrier.x1), int(barrier.y1)),
                (int(barrier.x2), int(barrier.y2)),
                int(barrier.thickness),
            )

        # Draw barrier being placed (drag preview)
        if tools.active_tool == TOOL_BARRIER and tools._barrier_start is not None:
            sx, sy = tools._barrier_start
            mx, my = pygame.mouse.get_pos()
            pygame.draw.line(
                self.surface, (140, 140, 160, 120),
                (int(sx), int(sy)), (mx, my), 3,
            )

    # ── Hazards ───────────────────────────────────────────────

    def _draw_hazards(self, world: World) -> None:
        """Draw hazard zones as concentric circles with pulsing opacity."""
        for hazard in world.hazards:
            radius = int(hazard.radius)
            cx, cy = int(hazard.x), int(hazard.y)

            if hazard.hazard_type == "lava":
                base_color = COLOR_HAZARD_LAVA
                ring_color = (255, 120, 50)
            else:  # cold
                base_color = COLOR_HAZARD_COLD
                ring_color = (160, 200, 255)

            pulse = 0.7 + 0.3 * math.sin(self.frame * 0.08)
            alpha = int(50 * pulse)

            hazard_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

            # Concentric rings for visual depth
            for r_frac in [1.0, 0.7, 0.4]:
                r = int(radius * r_frac)
                ring_alpha = int(alpha * (1.2 - r_frac))
                color = (*base_color, min(255, ring_alpha))
                pygame.draw.circle(hazard_surf, color, (radius, radius), r)

            # Outer ring
            ring_alpha = int(80 * pulse)
            pygame.draw.circle(
                hazard_surf, (*ring_color, ring_alpha), (radius, radius), radius, 2,
            )

            self.surface.blit(hazard_surf, (cx - radius, cy - radius))

            # Label
            label = "LAVA" if hazard.hazard_type == "lava" else "COLD"
            label_color = (255, 140, 60) if hazard.hazard_type == "lava" else (140, 200, 255)
            text = self.font_small.render(label, True, label_color)
            text.set_alpha(int(160 * pulse))
            self.surface.blit(text, text.get_rect(center=(cx, cy)))

    # ── Predators ─────────────────────────────────────────────

    def _draw_predators(self, world: World) -> None:
        """Draw predators as red triangles pointing in their heading direction."""
        for predator in world.predators:
            cx, cy = int(predator.x), int(predator.y)
            r = int(predator.radius)
            h = predator.heading

            # Triangle points: tip in heading direction, two back corners
            tip_x = cx + int(math.cos(h) * (r + 4))
            tip_y = cy + int(math.sin(h) * (r + 4))
            left_x = cx + int(math.cos(h + 2.5) * r)
            left_y = cy + int(math.sin(h + 2.5) * r)
            right_x = cx + int(math.cos(h - 2.5) * r)
            right_y = cy + int(math.sin(h - 2.5) * r)

            pygame.draw.polygon(
                self.surface, COLOR_PREDATOR,
                [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)],
            )
            # Dark outline
            pygame.draw.polygon(
                self.surface, (180, 30, 30),
                [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)],
                2,
            )

    # ── Creatures ────────────────────────────────────────────

    def _draw_creatures(self, world: World, mode: str) -> None:
        """Draw all living creatures with richer visuals."""
        for creature in world.creatures:
            if not creature.alive:
                continue

            color = self._creature_color(creature, mode)
            radius = max(2, int(creature.dna.effective_radius))
            cx, cy = int(creature.x), int(creature.y)

            # Outer glow (subtle)
            glow_radius = radius + 3
            glow_color = tuple(min(255, c + 30) for c in color)
            glow_surf = pygame.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surf, (*glow_color, 35),
                (glow_radius + 2, glow_radius + 2), glow_radius,
            )
            self.surface.blit(glow_surf, (cx - glow_radius - 2, cy - glow_radius - 2))

            # Body (with slight rim lighting)
            pygame.draw.circle(self.surface, color, (cx, cy), radius)
            # Inner highlight
            highlight = tuple(min(255, c + 50) for c in color)
            if radius > 3:
                pygame.draw.circle(
                    self.surface, highlight,
                    (cx - radius // 4, cy - radius // 4),
                    max(1, radius // 3),
                )

            # Heading indicator
            hx = cx + int(math.cos(creature.heading) * (radius + 5))
            hy = cy + int(math.sin(creature.heading) * (radius + 5))
            pygame.draw.line(self.surface, (220, 220, 240), (cx, cy), (hx, hy), 2)

    def _creature_color(self, creature: Creature, mode: str) -> tuple[int, int, int]:
        """
        Determine creature color based on mode.

        Isolation: rich color based on dominant trait and energy.
        Convergence: team colors with energy brightness.
        """
        if mode == "convergence":
            # Team color modulated by energy
            base = COLOR_LINEAGE_A if creature.lineage == "A" else COLOR_LINEAGE_B
            energy_ratio = max(0.3, min(1.0, creature.energy / BASE_ENERGY))
            return tuple(int(c * energy_ratio) for c in base)

        # Isolation: color based on dominant trait for visual diversity
        dna = creature.dna
        traits = [dna.speed, dna.size, dna.vision, dna.efficiency, dna.lifespan]
        max_trait = max(traits)
        trait_idx = traits.index(max_trait)
        energy_ratio = max(0.3, min(1.0, creature.energy / BASE_ENERGY))

        # Each dominant trait gets a distinct hue
        trait_colors = [
            (80, 200, 255),   # Speed -> cyan
            (255, 140, 60),   # Size -> orange
            (180, 100, 255),  # Vision -> purple
            (100, 255, 120),  # Efficiency -> green
            (220, 200, 60),   # Lifespan -> yellow/gold
        ]
        base = trait_colors[trait_idx]

        # Blend with secondary trait for more variety
        secondary_idx = sorted(range(5), key=lambda i: traits[i], reverse=True)[1]
        secondary = trait_colors[secondary_idx]
        blend = 0.25
        blended = tuple(int(base[i] * (1 - blend) + secondary[i] * blend) for i in range(3))

        return tuple(int(c * energy_ratio) for c in blended)

    # ── Event Tracking (for particles) ───────────────────────

    def _track_events(self, world: World, mode: str) -> None:
        """Track creature state changes to emit particles."""
        for creature in world.creatures:
            cid = id(creature)

            # Death detection
            was_alive = self._prev_alive.get(cid, True)
            if was_alive and not creature.alive:
                color = self._creature_color(creature, mode)
                self.particles.emit_death(creature.x, creature.y, color)
            self._prev_alive[cid] = creature.alive

            # Eat detection
            prev_food = self._prev_food_eaten.get(cid, 0)
            if creature.food_eaten > prev_food:
                self.particles.emit_eat(creature.x, creature.y)
            self._prev_food_eaten[cid] = creature.food_eaten

    # ── HUD ──────────────────────────────────────────────────

    def _draw_hud(self, world: World, mode: str, tools: PlayerTools | None = None) -> None:
        """Draw the heads-up display with generation info."""
        # Semi-transparent HUD background
        hud_h = 150 if mode == "isolation" else 200
        hud_surf = pygame.Surface((260, hud_h), pygame.SRCALPHA)
        hud_surf.fill((10, 10, 20, 160))
        self.surface.blit(hud_surf, (5, 5))

        y = 12
        lines = [
            (f"Gen: {world.generation}", (140, 180, 255)),
            (f"Alive: {world.alive_count()} / {len(world.creatures)}", (180, 220, 180)),
            (f"Time: {world.elapsed_time:.1f}s", (180, 180, 200)),
            (f"Food: {len(world.food)}", (140, 230, 140)),
            (self._time_of_day_label(world.daylight_factor), (255, 220, 140)),
        ]

        if tools and tools.drought_active:
            lines.append(("DROUGHT ACTIVE", (255, 200, 50)))

        # Seasonal indicator
        season_mult = world.seasonal_multiplier()
        if season_mult >= 0.8:
            season_label = "Abundant"
            season_color = (100, 255, 130)
        elif season_mult >= 0.5:
            season_label = "Normal"
            season_color = (180, 200, 180)
        else:
            season_label = "Scarce"
            season_color = (255, 160, 80)
        lines.append((f"Season: {season_label} ({season_mult:.0%})", season_color))

        if mode == "convergence":
            lines.append(("", (0, 0, 0)))
            a_alive = world.alive_count_by_lineage("A")
            b_alive = world.alive_count_by_lineage("B")
            a_food = world.food_eaten_by_lineage("A")
            b_food = world.food_eaten_by_lineage("B")
            lines.append((f"Red:  {a_alive} alive  {a_food} food", (220, 90, 90)))
            lines.append((f"Blue: {b_alive} alive  {b_food} food", (90, 130, 220)))

        for text_str, color in lines:
            if text_str:
                text = self.font.render(text_str, True, color)
                self.surface.blit(text, (14, y))
            y += 20

        # Controls hint at bottom
        controls = "SPACE=Pause  F=Fast  D=Debug  1-6=Tools  ESC=Menu"
        text = self.font_small.render(controls, True, (70, 70, 95))
        self.surface.blit(text, (10, config.WINDOW_HEIGHT - 22))

    # ── Toolbar ──────────────────────────────────────────────

    def _draw_toolbar(self, tools: PlayerTools) -> None:
        """Draw the player tools toolbar at the top-right."""
        toolbar_x = config.WINDOW_WIDTH - 380
        toolbar_y = 8
        btn_w = 58
        btn_h = 38
        gap = 4

        # Background
        total_w = len(TOOL_LIST) * (btn_w + gap) + gap
        tb_surf = pygame.Surface((total_w, btn_h + 28), pygame.SRCALPHA)
        tb_surf.fill((10, 12, 25, 180))
        self.surface.blit(tb_surf, (toolbar_x - gap, toolbar_y - 4))

        for i, tool in enumerate(TOOL_LIST):
            x = toolbar_x + i * (btn_w + gap)
            is_active = tools.active_tool == tool
            is_drought = tool == "drought" and tools.drought_active

            # Button background
            if is_active or is_drought:
                bg_color = TOOL_COLORS.get(tool, (80, 80, 80))
                border_color = (255, 255, 255)
            else:
                bg_color = (35, 38, 50)
                border_color = (60, 65, 80)

            rect = pygame.Rect(x, toolbar_y, btn_w, btn_h)
            pygame.draw.rect(self.surface, bg_color, rect, border_radius=4)
            pygame.draw.rect(self.surface, border_color, rect, 1, border_radius=4)

            # Label
            label = TOOL_LABELS.get(tool, tool)
            text = self.font_toolbar.render(label, True, (200, 210, 230))
            self.surface.blit(text, text.get_rect(center=(x + btn_w // 2, toolbar_y + btn_h // 2)))

            # Hotkey number
            key_text = self.font_toolbar.render(str(i + 1), True, (100, 100, 120))
            self.surface.blit(key_text, (x + 2, toolbar_y + 2))

        # Active tool description
        desc = TOOL_DESCRIPTIONS.get(tools.active_tool, "")
        if tools.drought_active:
            desc = "Drought ON -- no natural food spawning"
        desc_text = self.font_small.render(desc, True, (120, 130, 160))
        self.surface.blit(desc_text, (toolbar_x, toolbar_y + btn_h + 6))

    # ── Tool Cursor ──────────────────────────────────────────

    def draw_tool_cursor(self, tools: PlayerTools) -> None:
        """Draw a custom cursor showing the active tool's effect radius."""
        mx, my = pygame.mouse.get_pos()
        tool = tools.active_tool

        if tool == "food":
            # Scatter preview
            pygame.draw.circle(self.surface, (60, 200, 60, 80), (mx, my), 25, 1)
            for i in range(5):
                angle = i * 2 * math.pi / 5 + self.frame * 0.05
                px = mx + int(math.cos(angle) * 15)
                py = my + int(math.sin(angle) * 15)
                pygame.draw.circle(self.surface, (80, 220, 80), (px, py), 2)

        elif tool == "poison":
            surf = pygame.Surface((120, 120), pygame.SRCALPHA)
            pygame.draw.circle(surf, (140, 30, 140, 40), (60, 60), 60)
            pygame.draw.circle(surf, (180, 50, 180, 80), (60, 60), 60, 2)
            self.surface.blit(surf, (mx - 60, my - 60))

        elif tool == "bounty":
            surf = pygame.Surface((160, 160), pygame.SRCALPHA)
            pygame.draw.circle(surf, (30, 130, 130, 35), (80, 80), 80)
            pygame.draw.circle(surf, (50, 200, 200, 70), (80, 80), 80, 2)
            self.surface.blit(surf, (mx - 80, my - 80))

        elif tool == "barrier":
            pygame.draw.circle(self.surface, (140, 140, 160), (mx, my), 4)

    def _draw_pause_indicator(self) -> None:
        """Draw a translucent pause overlay."""
        overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.surface.blit(overlay, (0, 0))

        text = self.font_large.render("PAUSED", True, (200, 210, 230))
        rect = text.get_rect(center=(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2))
        self.surface.blit(text, rect)

    # ── Debug Overlay ────────────────────────────────────────

    def draw_debug(self, world: World) -> None:
        """Draw debug info: vision ranges, energy bars, sensor lines."""
        for creature in world.creatures:
            if not creature.alive:
                continue

            cx, cy = int(creature.x), int(creature.y)
            vision = int(creature.dna.effective_vision)

            # Vision range circle
            vis_surf = pygame.Surface((vision * 2, vision * 2), pygame.SRCALPHA)
            pygame.draw.circle(vis_surf, (40, 50, 80, 25), (vision, vision), vision)
            pygame.draw.circle(vis_surf, (60, 70, 100, 60), (vision, vision), vision, 1)
            self.surface.blit(vis_surf, (cx - vision, cy - vision))

            # Energy bar above creature
            bar_width = 24
            bar_height = 4
            ratio = max(0.0, min(1.0, creature.energy / BASE_ENERGY))
            bx = cx - bar_width // 2
            by = cy - int(creature.dna.effective_radius) - 10

            # Background
            pygame.draw.rect(self.surface, (30, 30, 40), (bx - 1, by - 1, bar_width + 2, bar_height + 2), border_radius=2)
            # Fill with gradient
            if ratio > 0.5:
                bar_color = (50, 200, 80)
            elif ratio > 0.25:
                bar_color = (200, 200, 50)
            else:
                bar_color = (220, 60, 50)
            pygame.draw.rect(self.surface, bar_color, (bx, by, int(bar_width * ratio), bar_height), border_radius=2)

            # Trait info
            dna = creature.dna
            info = f"S{dna.speed} Z{dna.size} V{dna.vision} E{dna.efficiency} L{dna.lifespan}"
            info_text = self.font_small.render(info, True, (120, 130, 160))
            self.surface.blit(info_text, (cx - info_text.get_width() // 2, by - 16))

    # ── Generation Stats Screen ──────────────────────────────

    def draw_generation_stats(
        self,
        world: World,
        best_fitness: float,
        avg_fitness: float,
        mode: str = "isolation",
    ) -> None:
        """Draw a generation summary overlay."""
        overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        self.surface.blit(overlay, (0, 0))

        cx, cy = config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2
        lines = [
            (f"Generation {world.generation} Complete", (140, 180, 255), True),
            ("", (0, 0, 0), False),
            (f"Best Fitness: {best_fitness:.1f}", (100, 255, 130), False),
            (f"Avg Fitness:  {avg_fitness:.1f}", (180, 200, 220), False),
        ]

        if mode == "convergence":
            lines.append(("", (0, 0, 0), False))
            lines.append((f"Red food: {world.food_eaten_by_lineage('A')}", (220, 90, 90), False))
            lines.append((f"Blue food: {world.food_eaten_by_lineage('B')}", (90, 130, 220), False))

        lines.append(("", (0, 0, 0), False))
        lines.append(("Starting next generation...", (120, 120, 150), False))

        y = cy - len(lines) * 14
        for text_str, color, is_title in lines:
            if text_str:
                font = self.font_large if is_title else self.font
                text = font.render(text_str, True, color)
                rect = text.get_rect(center=(cx, y))
                self.surface.blit(text, rect)
            y += 26

    def reset_tracking(self) -> None:
        """Reset particle/trail tracking for a new generation."""
        self._prev_alive.clear()
        self._prev_food_eaten.clear()
