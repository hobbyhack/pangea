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
    COLOR_BIOME_DESERT,
    COLOR_BIOME_FOREST,
    COLOR_BIOME_MOUNTAIN,
    COLOR_BIOME_ROAD,
    COLOR_BIOME_SWAMP,
    COLOR_BIOME_WATER,
    COLOR_FOOD,
    COLOR_HAZARD_COLD,
    COLOR_HAZARD_LAVA,
    COLOR_HERBIVORE,
    COLOR_HUD_TEXT,
)
import pangea.config as config
from pangea.config import EVOLUTION_POINTS
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
        self.screen = surface  # the actual display surface
        self.surface = surface  # drawing target (may be world_surface during draw)
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_large = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 48, bold=True)
        self.font_toolbar = pygame.font.SysFont("consolas", 12)

        self.particles = ParticleSystem()
        self.frame = 0
        # Selected creature for inspection
        self.selected_creature: Creature | None = None

        # Track previous alive state for death particles
        self._prev_alive: dict[int, bool] = {}
        # Track previous feeds count for eat particles
        self._prev_feeds_count: dict[int, int] = {}

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
        paused: bool = False,
        tools: PlayerTools | None = None,
        show_toolbar: bool = False,
        fast_forward: int = 0,
        debug: bool = False,
    ) -> None:
        """Draw the complete frame."""
        self.frame += 1

        # Set up world-sized drawing surface
        self._begin_world_draw(world)

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
        self._draw_creatures(world)

        # Update particle tracking
        self._track_events(world)

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

        # Debug overlay (world-space, before scaling)
        if debug:
            self.draw_debug(world)

        # Scale world onto screen, switch to screen-space drawing
        self._end_world_draw()

        # HUD (drawn directly on screen, not world surface)
        self._draw_hud(world, tools)

        # Toolbar
        if show_toolbar and tools:
            self._draw_toolbar(tools)

        if fast_forward:
            label = self.font.render(f">> {fast_forward}x", True, (255, 200, 80))
            self.surface.blit(label, (config.WINDOW_WIDTH // 2 - label.get_width() // 2, 6))

        if paused:
            self._draw_pause_indicator()

    def _begin_world_draw(self, world: World) -> None:
        """Prepare for world drawing (1:1 mapping, no scaling)."""
        self.surface = self.screen

    def _end_world_draw(self) -> None:
        """Finish world drawing and switch to screen-space drawing."""
        self.surface = self.screen

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

    _BIOME_COLORS = {
        "water": (COLOR_BIOME_WATER, 50),
        "road": (COLOR_BIOME_ROAD, 40),
        "forest": (COLOR_BIOME_FOREST, 55),
        "desert": (COLOR_BIOME_DESERT, 45),
        "swamp": (COLOR_BIOME_SWAMP, 50),
        "mountain": (COLOR_BIOME_MOUNTAIN, 60),
    }

    _BIOME_LABELS = {
        "water": "WATER",
        "road": "ROAD",
        "forest": "FOREST",
        "desert": "DESERT",
        "swamp": "SWAMP",
        "mountain": "MTN",
    }

    def _draw_biomes(self, world: World) -> None:
        """Draw biome regions as semi-transparent filled circles with labels."""
        # Use cached surfaces keyed by (biome_type, radius)
        if not hasattr(self, "_biome_surf_cache"):
            self._biome_surf_cache: dict[tuple[str, int], tuple[pygame.Surface, pygame.Surface]] = {}

        for biome in world.biomes:
            radius = int(biome.radius)
            cx, cy = int(biome.x), int(biome.y)

            cache_key = (biome.biome_type, radius)
            if cache_key not in self._biome_surf_cache:
                base_color, alpha = self._BIOME_COLORS.get(
                    biome.biome_type, (COLOR_BIOME_ROAD, 40)
                )
                color = (*base_color, alpha)
                biome_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(biome_surf, color, (radius, radius), radius)
                border_color = (*color[:3], min(255, color[3] + 30))
                pygame.draw.circle(biome_surf, border_color, (radius, radius), radius, 2)
                # Cache label surface too
                label = self._BIOME_LABELS.get(biome.biome_type, biome.biome_type.upper())
                label_color = tuple(min(255, c + 60) for c in base_color)
                text = self.font_small.render(label, True, label_color)
                text.set_alpha(100)
                self._biome_surf_cache[cache_key] = (biome_surf, text)

            biome_surf, text = self._biome_surf_cache[cache_key]
            self.surface.blit(biome_surf, (cx - radius, cy - radius))
            self.surface.blit(text, text.get_rect(center=(cx, cy)))

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

            if food.is_corpse:
                # Corpse: brown/tan color, no glow
                core_color = (
                    int(160 * freshness),
                    int(110 * freshness),
                    int(40 * freshness),
                )
                center_color = (
                    int(200 * freshness),
                    int(150 * freshness),
                    int(60 * freshness),
                )
            else:
                # Normal food: green with glow
                if freshness > 0.5:
                    self.surface.blit(glow, (fx - gw, fy - gw), special_flags=pygame.BLEND_ADD)

                core_color = (
                    int(60 * freshness),
                    int(220 * freshness),
                    int(60 * freshness),
                )
                center_color = (
                    int(140 * freshness),
                    int(255 * freshness),
                    int(140 * freshness),
                )

            pygame.draw.circle(self.surface, core_color, (fx, fy), max(2, int(food.radius)))
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

    # ── Creatures ────────────────────────────────────────────

    def _draw_creatures(self, world: World) -> None:
        """Draw all living creatures with richer visuals."""
        for creature in world.creatures:
            if not creature.alive:
                continue

            color = self._creature_color(creature)
            radius = max(2, int(creature.dna.effective_radius))
            cx, cy = int(creature.x), int(creature.y)

            # Outer glow (subtle) — cached by (radius, color)
            glow_radius = radius + 3
            glow_color = tuple(min(255, c + 30) for c in color)
            glow_key = (glow_radius, glow_color)
            if not hasattr(self, "_creature_glow_cache"):
                self._creature_glow_cache: dict[tuple, pygame.Surface] = {}
            if glow_key not in self._creature_glow_cache:
                glow_surf = pygame.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(
                    glow_surf, (*glow_color, 35),
                    (glow_radius + 2, glow_radius + 2), glow_radius,
                )
                self._creature_glow_cache[glow_key] = glow_surf
            self.surface.blit(self._creature_glow_cache[glow_key], (cx - glow_radius - 2, cy - glow_radius - 2))

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

    def _creature_color(self, creature: Creature) -> tuple[int, int, int]:
        """
        Determine creature color based on species and dominant trait.

        Blends species color with dominant trait color, modulated by energy.
        """
        _base_e = creature.species.settings.base_energy if creature.species else BASE_ENERGY
        energy_ratio = max(0.3, min(1.0, creature.energy / _base_e))

        # Species base tint
        species_color = creature.species.color if creature.species else COLOR_HERBIVORE

        # Blend species color with dominant-trait color
        dna = creature.dna
        traits = [dna.speed, dna.size, dna.vision, dna.efficiency, dna.lifespan]
        trait_idx = traits.index(max(traits))

        trait_colors = [
            (80, 200, 255),   # Speed -> cyan
            (255, 140, 60),   # Size -> orange
            (180, 100, 255),  # Vision -> purple
            (100, 255, 120),  # Efficiency -> green
            (220, 200, 60),   # Lifespan -> yellow/gold
        ]
        trait_color = trait_colors[trait_idx]

        # 50% species, 50% trait
        blended = tuple(
            int((species_color[i] * 0.5 + trait_color[i] * 0.5) * energy_ratio)
            for i in range(3)
        )
        return blended

    # ── Event Tracking (for particles) ───────────────────────

    def _track_events(self, world: World) -> None:
        """Track creature state changes to emit particles."""
        for creature in world.creatures:
            cid = id(creature)

            # Death detection
            was_alive = self._prev_alive.get(cid, True)
            if was_alive and not creature.alive:
                color = self._creature_color(creature)
                self.particles.emit_death(creature.x, creature.y, color)
            self._prev_alive[cid] = creature.alive

            # Eat detection (triggers on all energy gains including attacks/scavenge)
            prev_feeds = self._prev_feeds_count.get(cid, 0)
            if creature.feeds_count > prev_feeds:
                self.particles.emit_eat(creature.x, creature.y)
            self._prev_feeds_count[cid] = creature.feeds_count

    # ── HUD ──────────────────────────────────────────────────

    def _draw_hud(self, world: World, tools: PlayerTools | None = None) -> None:
        """Draw the heads-up display with generation info."""
        # Semi-transparent HUD background
        hud_h = 230
        hud_surf = pygame.Surface((280, hud_h), pygame.SRCALPHA)
        hud_surf.fill((10, 10, 20, 160))
        self.surface.blit(hud_surf, (5, 5))

        y = 12

        alive = world.alive_count()
        cap = world.settings.total_freeplay_carrying_capacity()
        elapsed = world.elapsed_time
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        alive_creatures = [c for c in world.creatures if c.alive]
        avg_gen = (
            sum(c.generation for c in alive_creatures) / len(alive_creatures)
            if alive_creatures else 0
        )
        births_rate = getattr(world, '_freeplay_births_per_min', 0.0)
        deaths_rate = getattr(world, '_freeplay_deaths_per_min', 0.0)
        lines = [
            (f"Freeplay  {mins}m {secs:02d}s", (220, 180, 100)),
            (f"Pop: {alive} / {cap} (cap)", (180, 220, 180)),
            (f"Births: {world.total_births}  Deaths: {world.total_deaths}", (180, 180, 200)),
            (f"B/min: {births_rate:.1f}  D/min: {deaths_rate:.1f}", (160, 170, 190)),
            (f"Avg Gen: {avg_gen:.1f}", (140, 180, 255)),
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

        for text_str, color in lines:
            if text_str:
                text = self.font.render(text_str, True, color)
                self.surface.blit(text, (14, y))
            y += 20

        # Controls hint at bottom
        controls = "SPACE=Pause  F=Fast  +/-=Speed  D=Debug  E=Evo  1-6=Tools  RClick=Inspect  ESC=Menu"
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
            _base_e = creature.species.settings.base_energy if creature.species else BASE_ENERGY
            ratio = max(0.0, min(1.0, creature.energy / _base_e))
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
        self._prev_feeds_count.clear()

    # ── Evolution Panel ──────────────────────────────────────

    def draw_evolution_panel(
        self,
        world: World,
        generation_history: list[dict],
    ) -> None:
        """Draw the evolution/species tracker side panel."""
        panel_w = 340
        panel_h = config.WINDOW_HEIGHT - 20
        panel_x = config.WINDOW_WIDTH - panel_w - 10
        panel_y = 10

        # Semi-transparent panel background
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((10, 12, 25, 210))
        pygame.draw.rect(panel, (50, 55, 80), (0, 0, panel_w, panel_h), 1, border_radius=6)
        self.surface.blit(panel, (panel_x, panel_y))

        self._draw_freeplay_evolution_panel(
            world, panel_x, panel_y, panel_w, panel_h, generation_history,
        )

    def _draw_freeplay_evolution_panel(
        self,
        world: World,
        panel_x: int,
        panel_y: int,
        panel_w: int,
        panel_h: int,
        history: list[dict],
    ) -> None:
        """Draw the freeplay species tracker panel with per-diet stats and graphs."""
        y = panel_y + 8

        # ── Title ──
        title = self.font_large.render("Species Tracker", True, (140, 180, 255))
        self.surface.blit(title, (panel_x + 12, y))
        y += 28

        alive = [c for c in world.creatures if c.alive]

        # ── Minimap (compact) ──
        map_x = panel_x + 12
        map_w = panel_w - 24
        map_h = 100
        pygame.draw.rect(self.surface, (20, 22, 35), (map_x, y, map_w, map_h), border_radius=4)
        pygame.draw.rect(self.surface, (40, 45, 60), (map_x, y, map_w, map_h), 1, border_radius=4)

        sx = map_w / max(world.width, 1)
        sy = map_h / max(world.height, 1)
        for food in world.food:
            fx = map_x + int(food.x * sx)
            fy = y + int(food.y * sy)
            pygame.draw.circle(self.surface, (40, 140, 40), (fx, fy), 1)
        for creature in world.creatures:
            if not creature.alive:
                continue
            cx = map_x + int(creature.x * sx)
            cy = y + int(creature.y * sy)
            color = self._creature_color(creature)
            r = max(2, int(creature.dna.effective_radius * sx * 2))
            pygame.draw.circle(self.surface, color, (cx, cy), r)

        label = self.font_small.render(
            f"{world.alive_count()} alive", True, (120, 130, 160),
        )
        self.surface.blit(label, (map_x + map_w - label.get_width(), y + map_h + 2))
        y += map_h + 18

        # ── Per-Species Cards ──
        registry = world.settings.species_registry
        by_species: dict[str, list] = {sp.id: [] for sp in registry.all()}
        for c in alive:
            by_species.setdefault(c.dna.species_id, []).append(c)

        species_info = [
            (sp.id, sp.name, sp.color, sp.settings)
            for sp in registry.all()
        ]

        trait_colors = [
            (80, 200, 255), (255, 140, 60), (180, 100, 255),
            (100, 255, 120), (220, 200, 60),
        ]
        max_pts = EVOLUTION_POINTS * 0.6

        # Hover tooltip definitions for stat labels
        # Each entry: (label_text, hitbox_rect, tooltip_text)
        hover_items: list[tuple[str, pygame.Rect, str]] = []
        mx, my = pygame.mouse.get_pos()

        # Get latest snapshot stats
        latest_snapshot = history[-1] if history else None

        for sp_id, name, color, ss in species_info:
            creatures = by_species.get(sp_id, [])
            n = len(creatures)

            # Get per-species stats from snapshot
            sp_stats = (
                latest_snapshot.get(f"{sp_id}_stats", {})
                if latest_snapshot else {}
            )

            # Card background
            card_h = 146
            card_bg = pygame.Surface((panel_w - 24, card_h), pygame.SRCALPHA)
            card_bg.fill((20, 22, 35, 180))
            self.surface.blit(card_bg, (panel_x + 12, y))
            pygame.draw.rect(
                self.surface, color,
                (panel_x + 12, y, 3, card_h),  # left color accent
            )

            # Header: Name + count/cap
            header = self.font.render(
                f"{name}  {n}/{ss.freeplay_hard_cap}", True, color,
            )
            self.surface.blit(header, (panel_x + 20, y + 3))
            hover_items.append((
                "Pop", pygame.Rect(panel_x + 20, y + 3, header.get_width(), 16),
                "Alive count / hard population cap",
            ))

            if n > 0:
                avg_gen = sum(c.generation for c in creatures) / n
                avg_food = sum(c.feeds_count for c in creatures) / n
                avg_energy = sum(c.energy for c in creatures) / n
                avg_age = sum(c.age for c in creatures) / n
                avg_offspring = sum(c.offspring_count for c in creatures) / n

                # Brain utilization: % of weights with |w| > 0.1
                import numpy as np
                total_weights = 0
                active_weights = 0
                for c in creatures:
                    for w in c.dna.weights:
                        flat = np.abs(w).ravel()
                        total_weights += len(flat)
                        active_weights += int(np.sum(flat > 0.1))
                brain_pct = (active_weights / total_weights * 100) if total_weights else 0

                # Stats row 1: Gen, Food, Energy, Age
                lx = panel_x + 20
                row1_parts = [
                    (f"Gen:{avg_gen:.1f}", "Avg evolutionary generation"),
                    (f"  Feeds:{avg_food:.1f}", "Avg feeds per creature (P=plant A=attack C=corpse)"),
                    (f"  Energy:{avg_energy:.0f}", "Avg current energy level"),
                    (f"  Age:{avg_age:.0f}s", "Avg age in seconds"),
                ]
                rx = lx
                for text, tip in row1_parts:
                    txt = self.font_small.render(text, True, (160, 170, 190))
                    self.surface.blit(txt, (rx, y + 21))
                    hover_items.append((
                        text, pygame.Rect(rx, y + 21, txt.get_width(), 13), tip,
                    ))
                    rx += txt.get_width()

                # Stats row 2: Feed breakdown (P=plant, A=attack, C=corpse)
                avg_p = sum(c.plant_feeds for c in creatures) / n
                avg_a = sum(c.attack_feeds for c in creatures) / n
                avg_c = sum(c.corpse_feeds for c in creatures) / n
                feed_row = f"P:{avg_p:.1f}  A:{avg_a:.1f}  C:{avg_c:.1f}"
                feed_txt = self.font_small.render(
                    feed_row, True, (130, 150, 170),
                )
                self.surface.blit(feed_txt, (panel_x + 20, y + 37))
                hover_items.append((
                    "PAC",
                    pygame.Rect(panel_x + 20, y + 37, feed_txt.get_width(), 13),
                    "Avg feeds breakdown: Plant / Attack / Corpse+Scavenge",
                ))

                # Stats row 3: Extinction metrics
                ext_count = sp_stats.get("extinction_count", 0)
                time_since = sp_stats.get("time_since_extinction", 0)
                if ext_count > 0:
                    mins = int(time_since) // 60
                    secs = int(time_since) % 60
                    row2 = f"Extinctions:{ext_count}  Last:{mins}:{secs:02d} ago"
                else:
                    row2 = "No extinctions"
                txt2 = self.font_small.render(row2, True, (140, 220, 180))
                self.surface.blit(txt2, (panel_x + 20, y + 53))
                hover_items.append((
                    "Ext", pygame.Rect(panel_x + 20, y + 53, txt2.get_width(), 13),
                    "Times this species went to 0 alive and was respawned",
                ))

                # Stats row 4: Offspring + Brain %
                row3 = f"Offspring:{avg_offspring:.1f}"
                txt3 = self.font_small.render(row3, True, (140, 150, 170))
                self.surface.blit(txt3, (panel_x + 20, y + 69))
                hover_items.append((
                    "Offspring", pygame.Rect(panel_x + 20, y + 69, txt3.get_width(), 13),
                    "Avg children produced per creature",
                ))

                brain_txt = self.font_small.render(
                    f"Brain:{brain_pct:.0f}%", True, (180, 160, 255),
                )
                brain_x = panel_x + 20 + txt3.get_width() + 12
                self.surface.blit(brain_txt, (brain_x, y + 69))
                hover_items.append((
                    "Brain", pygame.Rect(brain_x, y + 69, brain_txt.get_width(), 13),
                    "% of neural network weights actively used (|w| > 0.1)",
                ))

                # Stats row 5: MinPop, 1st Food, DeathE
                min_pop = sp_stats.get("min_pop", 0)
                avg_first = sp_stats.get("avg_first_food_time", -1)
                avg_death_e = sp_stats.get("avg_energy_at_death", -1)
                row4_parts = [
                    (f"MinPop:{min_pop}",
                     "Lowest population count since last extinction"),
                    (f"  1stFood:{avg_first:.1f}s" if avg_first >= 0 else "  1stFood:--",
                     "Avg seconds until a newborn eats its first food \u2014 measures NN foraging competence"),
                    (f"  DeathE:{avg_death_e:.0f}" if avg_death_e >= 0 else "  DeathE:--",
                     "Avg energy when creatures die \u2014 0 means starvation, higher means dying of old age or predation"),
                ]
                rx4 = panel_x + 20
                for text, tip in row4_parts:
                    txt4 = self.font_small.render(text, True, (160, 170, 190))
                    self.surface.blit(txt4, (rx4, y + 85))
                    hover_items.append((
                        text, pygame.Rect(rx4, y + 85, txt4.get_width(), 13), tip,
                    ))
                    rx4 += txt4.get_width()

                # Stats row 6: Genetic Diversity
                diversity = sp_stats.get("genetic_diversity", 0)
                div_txt = self.font_small.render(
                    f"Diversity:{diversity:.3f}", True, (160, 170, 190),
                )
                self.surface.blit(div_txt, (panel_x + 20, y + 101))
                hover_items.append((
                    "Diversity",
                    pygame.Rect(panel_x + 20, y + 101, div_txt.get_width(), 13),
                    "NN weight variance across population \u2014 too low means inbreeding, too high means no convergence",
                ))

                # Trait bars (bigger)
                bar_x = panel_x + 20
                bar_y_start = y + 114
                bar_w = (panel_w - 44) // 5 - 2
                bar_h = 10
                trait_vals = [
                    sum(c.dna.speed for c in creatures) / n,
                    sum(c.dna.size for c in creatures) / n,
                    sum(c.dna.vision for c in creatures) / n,
                    sum(c.dna.efficiency for c in creatures) / n,
                    sum(c.dna.lifespan for c in creatures) / n,
                ]
                trait_labels = ["Spd", "Siz", "Vis", "Eff", "Lif"]
                trait_tips = [
                    "Speed - movement rate (from 100pt genetic budget)",
                    "Size - body radius, affects collision (from 100pt genetic budget)",
                    "Vision - sensor range for detecting food/threats (from 100pt genetic budget)",
                    "Efficiency - energy consumption rate (from 100pt genetic budget)",
                    "Lifespan - max age before natural death (from 100pt genetic budget)",
                ]
                for i, (val, tc, tl, tip) in enumerate(
                    zip(trait_vals, trait_colors, trait_labels, trait_tips)
                ):
                    bx = bar_x + i * (bar_w + 2)
                    # Label
                    lb = self.font_small.render(tl, True, tc)
                    self.surface.blit(lb, (bx, bar_y_start - 1))
                    # Bar bg
                    bbx = bx + lb.get_width() + 3
                    actual_bar_w = bar_w - lb.get_width() - 3
                    pygame.draw.rect(
                        self.surface, (30, 32, 45),
                        (bbx, bar_y_start + 2, actual_bar_w, bar_h),
                        border_radius=2,
                    )
                    fill_w = max(1, int(actual_bar_w * min(val / max_pts, 1.0)))
                    pygame.draw.rect(
                        self.surface, tc,
                        (bbx, bar_y_start + 2, fill_w, bar_h),
                        border_radius=2,
                    )
                    # Value label inside/beside bar
                    val_lbl = self.font_small.render(f"{val:.0f}", True, (200, 210, 220))
                    self.surface.blit(val_lbl, (bbx + 2, bar_y_start + 1))
                    hover_items.append((
                        tl, pygame.Rect(bx, bar_y_start - 1, bar_w, bar_h + 4), tip,
                    ))

                # Budget allocation bar (how 100 pts are distributed)
                budget_y = bar_y_start + bar_h + 8
                budget_total = sum(trait_vals)
                if budget_total > 0:
                    bx = panel_x + 20
                    budget_bar_w = panel_w - 44
                    for i, (val, tc) in enumerate(zip(trait_vals, trait_colors)):
                        seg_w = max(1, int(budget_bar_w * val / budget_total))
                        pygame.draw.rect(
                            self.surface, tc,
                            (bx, budget_y, seg_w, 6), border_radius=1,
                        )
                        bx += seg_w
                    hover_items.append((
                        "Budget",
                        pygame.Rect(panel_x + 20, budget_y, budget_bar_w, 6),
                        "Genetic budget distribution - 100 pts split across all traits",
                    ))

            else:
                ext_count = sp_stats.get("extinction_count", 0)
                ext_text = f"EXTINCT  ({ext_count} total)" if ext_count else "EXTINCT"
                extinct_lbl = self.font_small.render(
                    ext_text, True, (120, 60, 60),
                )
                self.surface.blit(extinct_lbl, (panel_x + 20, y + 28))

            y += card_h + 6

        # ── Hover Tooltip ──
        for _, rect, tip in hover_items:
            if rect.collidepoint(mx, my):
                tip_surf = self.font_small.render(tip, True, (230, 235, 245))
                tip_w = tip_surf.get_width() + 10
                tip_h = tip_surf.get_height() + 6
                # Position tooltip to left of panel to avoid clipping
                tx = min(mx + 12, panel_x - tip_w - 4)
                ty = my - 8
                bg = pygame.Surface((tip_w, tip_h), pygame.SRCALPHA)
                bg.fill((15, 18, 30, 230))
                pygame.draw.rect(bg, (80, 90, 120), (0, 0, tip_w, tip_h), 1, border_radius=3)
                self.surface.blit(bg, (tx, ty))
                self.surface.blit(tip_surf, (tx + 5, ty + 3))
                break  # only show one tooltip at a time

    # ── Creature Inspection ──────────────────────────────────

    def try_select_creature(self, world: World, mx: int, my: int) -> bool:
        """
        Try to select a creature at the given screen coordinates.

        Returns True if a creature was selected, False otherwise.
        """
        best_creature = None
        best_dist_sq = float("inf")

        for creature in world.creatures:
            if not creature.alive:
                continue
            dx = creature.x - mx
            dy = creature.y - my
            dist_sq = dx * dx + dy * dy
            # Click within creature radius + a generous margin
            click_radius = max(10, creature.dna.effective_radius + 5)
            if dist_sq < click_radius * click_radius and dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_creature = creature

        self.selected_creature = best_creature
        return best_creature is not None

    def deselect_creature(self) -> None:
        """Deselect the currently selected creature."""
        self.selected_creature = None

    def draw_creature_stats(self, world: World) -> None:
        """Draw a stats panel for the selected creature."""
        creature = self.selected_creature
        if creature is None or not creature.alive:
            self.selected_creature = None
            return

        dna = creature.dna
        cx, cy = int(creature.x), int(creature.y)

        # Selection ring around the creature
        radius = max(2, int(dna.effective_radius))
        ring_radius = radius + 8
        pulse = 0.7 + 0.3 * math.sin(self.frame * 0.15)
        ring_alpha = int(200 * pulse)
        ring_surf = pygame.Surface((ring_radius * 2 + 4, ring_radius * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(
            ring_surf, (255, 255, 255, ring_alpha),
            (ring_radius + 2, ring_radius + 2), ring_radius, 2,
        )
        self.surface.blit(ring_surf, (cx - ring_radius - 2, cy - ring_radius - 2))

        # Stats panel — position near creature but keep on-screen
        panel_w = 200
        panel_h = 210
        px = cx + ring_radius + 10
        py = cy - panel_h // 2

        # Keep panel on screen
        sw = self.surface.get_width()
        sh = self.surface.get_height()
        if px + panel_w > sw - 5:
            px = cx - ring_radius - panel_w - 10
        if py < 5:
            py = 5
        if py + panel_h > sh - 5:
            py = sh - panel_h - 5

        # Panel background
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((10, 12, 25, 220))
        pygame.draw.rect(panel, (80, 90, 120), (0, 0, panel_w, panel_h), 1, border_radius=6)
        self.surface.blit(panel, (px, py))

        # Title
        title = self.font.render("Creature Stats", True, (140, 180, 255))
        self.surface.blit(title, (px + 8, py + 6))

        # Stats lines
        y = py + 28
        _base_e = creature.species.settings.base_energy if creature.species else BASE_ENERGY
        energy_pct = creature.energy / _base_e * 100
        age_max = dna.effective_lifespan
        stats_lines = [
            (f"Energy: {creature.energy:.0f} ({energy_pct:.0f}%)",
             (100, 255, 130) if energy_pct > 50 else (255, 200, 50) if energy_pct > 25 else (255, 80, 60)),
            (f"Feeds:  {creature.feeds_count}  (P:{creature.plant_feeds} A:{creature.attack_feeds} C:{creature.corpse_feeds})",
             (140, 230, 140)),
            (f"Age:    {creature.age:.1f}s / {age_max:.0f}s", (180, 180, 200)),
            ("", (0, 0, 0)),
            (f"Speed:      {dna.speed:3d}  ({dna.max_speed:.1f} px/f)", (80, 200, 255)),
            (f"Size:       {dna.size:3d}  (r={dna.effective_radius:.1f})", (255, 140, 60)),
            (f"Vision:     {dna.vision:3d}  ({dna.effective_vision:.0f} px)", (180, 100, 255)),
            (f"Efficiency: {dna.efficiency:3d}  (x{dna.effective_efficiency:.2f})", (100, 255, 120)),
            (f"Lifespan:   {dna.lifespan:3d}  ({age_max:.0f}s)", (220, 200, 60)),
            (f"Species:    {creature.species.name if creature.species else dna.species_id}",
             creature.species.color if creature.species else (200, 200, 200)),
        ]

        for text_str, color in stats_lines:
            if text_str:
                txt = self.font_small.render(text_str, True, color)
                self.surface.blit(txt, (px + 8, y))
            y += 16
