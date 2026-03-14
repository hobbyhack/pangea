"""
Menu -- pygame menu screens for mode selection, settings, and file picking.
============================================================
Provides the main menu, in-app settings panel, file selection for
convergence mode, and the pause overlay.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from pangea.config import (
    COLOR_BUTTON,
    COLOR_BUTTON_HOVER,
    COLOR_BUTTON_TEXT,
    COLOR_HUD_TEXT,
    COLOR_MENU_BG,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from pangea.settings import SETTING_DEFS, SimSettings


class Button:
    """A simple rectangular button with hover effect and optional color override."""

    def __init__(
        self,
        x: int, y: int, width: int, height: int, text: str,
        color: tuple[int, int, int] | None = None,
        hover_color: tuple[int, int, int] | None = None,
        text_color: tuple[int, int, int] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False
        self._color = color or COLOR_BUTTON
        self._hover_color = hover_color or COLOR_BUTTON_HOVER
        self._text_color = text_color or COLOR_BUTTON_TEXT

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = self._hover_color if self.hovered else self._color
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (80, 90, 120), self.rect, 2, border_radius=6)

        text_surf = font.render(self.text, True, self._text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos: tuple[int, int]) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)


class Menu:
    """Main menu and sub-menus for the simulator."""

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        self.font = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_title = pygame.font.SysFont("consolas", 52, bold=True)
        self.font_subtitle = pygame.font.SysFont("consolas", 16)
        self.font_heading = pygame.font.SysFont("consolas", 18, bold=True)

    # -- Main Menu ------------------------------------------------------------

    def show_main_menu(self, settings: SimSettings | None = None) -> tuple[str, SimSettings]:
        """
        Display the main menu and return the user's choice + settings.

        Returns:
            Tuple of (choice_string, settings_object).
        """
        if settings is None:
            settings = SimSettings()

        cx = WINDOW_WIDTH // 2
        btn_w, btn_h = 280, 50

        buttons = {
            "isolation": Button(cx - btn_w // 2, 330, btn_w, btn_h, "Isolation Mode",
                                color=(40, 70, 50), hover_color=(55, 100, 65)),
            "convergence": Button(cx - btn_w // 2, 400, btn_w, btn_h, "Convergence Mode",
                                  color=(50, 45, 75), hover_color=(70, 60, 110)),
            "settings": Button(cx - btn_w // 2, 470, btn_w, btn_h, "Settings",
                               color=(55, 55, 65), hover_color=(75, 75, 90)),
            "quit": Button(cx - btn_w // 2, 540, btn_w, btn_h, "Quit",
                           color=(65, 40, 40), hover_color=(90, 50, 50)),
        }

        clock = pygame.time.Clock()
        frame = 0

        while True:
            mouse_pos = pygame.mouse.get_pos()
            frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ("quit", settings)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return ("quit", settings)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, btn in buttons.items():
                        if btn.is_clicked(mouse_pos):
                            if name == "settings":
                                settings = self.show_settings(settings)
                            else:
                                return (name, settings)

            # Draw
            self._draw_menu_bg(frame)

            # Title with glow effect
            title_color = self._pulse_color((80, 160, 255), (120, 200, 255), frame, 120)
            title = self.font_title.render("PANGEA", True, title_color)
            title_rect = title.get_rect(center=(cx, 180))
            self.surface.blit(title, title_rect)

            # Subtitle
            sub = self.font_subtitle.render(
                "Distributed Evolution Simulator", True, (120, 130, 160)
            )
            sub_rect = sub.get_rect(center=(cx, 240))
            self.surface.blit(sub, sub_rect)

            # Version
            ver = self.font_small.render("v0.1.0", True, (70, 70, 90))
            self.surface.blit(ver, ver.get_rect(center=(cx, 270)))

            for btn in buttons.values():
                btn.update(mouse_pos)
                btn.draw(self.surface, self.font)

            pygame.display.flip()
            clock.tick(30)

    # -- Settings Panel -------------------------------------------------------

    def show_settings(self, settings: SimSettings) -> SimSettings:
        """
        Show the in-app settings panel with sliders for all tunable parameters.

        Returns:
            Updated SimSettings object.
        """
        settings = settings.copy()
        clock = pygame.time.Clock()

        # Layout constants
        panel_x = 200
        panel_w = WINDOW_WIDTH - 400
        slider_w = 250
        slider_h = 8
        row_h = 36

        # Build slider data
        sliders: list[dict] = []
        y = 140
        last_category = ""
        for sdef in SETTING_DEFS:
            if sdef.category != last_category:
                last_category = sdef.category
                y += 10  # category gap
            sliders.append({
                "def": sdef,
                "y": y,
                "dragging": False,
            })
            y += row_h

        back_btn = Button(WINDOW_WIDTH // 2 - 80, y + 30, 160, 45, "Back",
                          color=(50, 60, 80), hover_color=(70, 80, 110))
        reset_btn = Button(WINDOW_WIDTH // 2 + 100, y + 30, 140, 45, "Reset",
                           color=(80, 45, 45), hover_color=(110, 60, 60))

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return settings
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return settings

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if back_btn.is_clicked(mouse_pos):
                        return settings
                    if reset_btn.is_clicked(mouse_pos):
                        settings = SimSettings()
                        continue

                    # Check sliders and toggles
                    for s in sliders:
                        sdef = s["def"]
                        if sdef.widget_type == "toggle":
                            # Toggle switch click area
                            toggle_x = panel_x + panel_w - slider_w - 40
                            toggle_rect = pygame.Rect(toggle_x, s["y"], 60, 26)
                            if toggle_rect.collidepoint(mouse_pos):
                                current = bool(getattr(settings, sdef.key))
                                setattr(settings, sdef.key, not current)
                        else:
                            slider_rect = pygame.Rect(
                                panel_x + panel_w - slider_w - 40, s["y"], slider_w, slider_h + 16
                            )
                            if slider_rect.collidepoint(mouse_pos):
                                s["dragging"] = True

                if event.type == pygame.MOUSEBUTTONUP:
                    for s in sliders:
                        s["dragging"] = False

            # Update dragging sliders
            for s in sliders:
                if s["dragging"]:
                    sdef = s["def"]
                    if sdef.widget_type == "toggle":
                        continue
                    sx = panel_x + panel_w - slider_w - 40
                    t = max(0.0, min(1.0, (mouse_pos[0] - sx) / slider_w))
                    raw = sdef.min_val + t * (sdef.max_val - sdef.min_val)
                    # Snap to step
                    snapped = round(raw / sdef.step) * sdef.step
                    snapped = max(sdef.min_val, min(sdef.max_val, snapped))
                    # Convert to int if step >= 1
                    if sdef.step >= 1:
                        snapped = int(snapped)
                    setattr(settings, sdef.key, snapped)

            # Draw
            self.surface.fill(COLOR_MENU_BG)

            # Title
            title = self.font_heading.render("SIMULATION SETTINGS", True, (140, 160, 200))
            self.surface.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 80)))

            hint = self.font_small.render(
                "Drag sliders to adjust. Changes apply to next generation.",
                True, (100, 100, 130),
            )
            self.surface.blit(hint, hint.get_rect(center=(WINDOW_WIDTH // 2, 108)))

            # Draw sliders and toggles
            last_cat = ""
            for s in sliders:
                sdef = s["def"]

                # Category header
                if sdef.category != last_cat:
                    last_cat = sdef.category
                    cat_text = self.font_heading.render(sdef.category, True, (100, 140, 200))
                    self.surface.blit(cat_text, (panel_x, s["y"] - 6))

                # Label
                label = self.font_small.render(sdef.label, True, (180, 180, 200))
                self.surface.blit(label, (panel_x + 140, s["y"] + 2))

                if sdef.widget_type == "toggle":
                    # Draw toggle switch
                    self._draw_toggle(
                        s["y"],
                        panel_x + panel_w - slider_w - 40,
                        bool(getattr(settings, sdef.key)),
                    )
                else:
                    # Slider track
                    sx = panel_x + panel_w - slider_w - 40
                    sy = s["y"] + 8
                    val = getattr(settings, sdef.key)
                    t = (val - sdef.min_val) / (sdef.max_val - sdef.min_val) if sdef.max_val > sdef.min_val else 0

                    # Track background
                    pygame.draw.rect(self.surface, (40, 40, 55), (sx, sy, slider_w, slider_h), border_radius=4)
                    # Filled portion with color gradient
                    fill_color = self._lerp_color((60, 130, 200), (100, 220, 160), t)
                    fill_w = int(slider_w * t)
                    if fill_w > 0:
                        pygame.draw.rect(self.surface, fill_color, (sx, sy, fill_w, slider_h), border_radius=4)

                    # Thumb
                    thumb_x = sx + int(slider_w * t)
                    thumb_color = (200, 210, 230) if s["dragging"] else (150, 160, 180)
                    pygame.draw.circle(self.surface, thumb_color, (thumb_x, sy + slider_h // 2), 7)

                    # Value text
                    val_str = f"{val:{sdef.fmt}}"
                    val_text = self.font_small.render(val_str, True, (160, 170, 190))
                    self.surface.blit(val_text, (sx + slider_w + 10, s["y"] + 2))

            back_btn.update(mouse_pos)
            back_btn.draw(self.surface, self.font)
            reset_btn.update(mouse_pos)
            reset_btn.draw(self.surface, self.font)

            pygame.display.flip()
            clock.tick(30)

    def _draw_toggle(self, y: int, x: int, enabled: bool) -> None:
        """Draw a toggle switch at the given position."""
        width = 50
        height = 22
        border_radius = height // 2
        ty = y + 4

        if enabled:
            # Active background (green-ish)
            bg_color = (50, 160, 80)
            knob_x = x + width - height // 2 - 2
        else:
            # Inactive background (dark gray)
            bg_color = (50, 50, 60)
            knob_x = x + height // 2 + 2

        # Track
        pygame.draw.rect(self.surface, bg_color, (x, ty, width, height), border_radius=border_radius)
        pygame.draw.rect(self.surface, (80, 90, 110), (x, ty, width, height), 1, border_radius=border_radius)

        # Knob
        knob_color = (220, 225, 235) if enabled else (140, 145, 155)
        pygame.draw.circle(self.surface, knob_color, (knob_x, ty + height // 2), height // 2 - 2)

        # Label text
        label = "ON" if enabled else "OFF"
        label_color = (100, 220, 130) if enabled else (130, 130, 150)
        label_text = self.font_small.render(label, True, label_color)
        self.surface.blit(label_text, (x + width + 10, y + 2))

    # -- File Selection -------------------------------------------------------

    def show_file_select(self, species_dir: str = "species") -> tuple[str, str] | None:
        """
        Show file selection screen for convergence mode.

        Returns:
            Tuple of (file_a_path, file_b_path) or None if cancelled.
        """
        species_path = Path(species_dir)
        if not species_path.exists():
            species_path.mkdir(parents=True, exist_ok=True)

        files = sorted(str(f.name) for f in species_path.glob("*.json"))

        if len(files) < 2:
            return self._show_message(
                "Need at least 2 species files in the 'species/' folder.",
                "Save species in Isolation Mode first!",
                "Press ESC to go back.",
            )

        selected: list[int] = []
        scroll_offset = 0
        max_visible = 12

        cx = WINDOW_WIDTH // 2
        clock = pygame.time.Clock()

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i in range(min(max_visible, len(files) - scroll_offset)):
                        idx = i + scroll_offset
                        item_rect = pygame.Rect(cx - 200, 200 + i * 35, 400, 30)
                        if item_rect.collidepoint(mouse_pos):
                            if idx in selected:
                                selected.remove(idx)
                            elif len(selected) < 2:
                                selected.append(idx)

                    if len(selected) == 2:
                        start_rect = pygame.Rect(cx - 100, 650, 200, 45)
                        if start_rect.collidepoint(mouse_pos):
                            file_a = str(species_path / files[selected[0]])
                            file_b = str(species_path / files[selected[1]])
                            return (file_a, file_b)

                if event.type == pygame.MOUSEWHEEL:
                    scroll_offset = max(0, min(len(files) - max_visible, scroll_offset - event.y))

            self.surface.fill(COLOR_MENU_BG)

            title = self.font.render("Select 2 Species Files", True, COLOR_HUD_TEXT)
            self.surface.blit(title, title.get_rect(center=(cx, 130)))

            hint = self.font_subtitle.render(
                "Click to select (Red = Species A, Blue = Species B)",
                True, (120, 120, 150),
            )
            self.surface.blit(hint, hint.get_rect(center=(cx, 165)))

            for i in range(min(max_visible, len(files) - scroll_offset)):
                idx = i + scroll_offset
                item_rect = pygame.Rect(cx - 200, 200 + i * 35, 400, 30)

                if idx in selected:
                    sel_idx = selected.index(idx)
                    color = (100, 40, 40) if sel_idx == 0 else (40, 50, 100)
                    label = " [A - Red]" if sel_idx == 0 else " [B - Blue]"
                else:
                    color = (35, 38, 52)
                    label = ""

                hovered = item_rect.collidepoint(mouse_pos)
                if hovered:
                    color = tuple(min(255, c + 20) for c in color)

                pygame.draw.rect(self.surface, color, item_rect, border_radius=4)
                pygame.draw.rect(self.surface, (60, 65, 80), item_rect, 1, border_radius=4)
                text = self.font_subtitle.render(files[idx] + label, True, COLOR_HUD_TEXT)
                self.surface.blit(text, (item_rect.x + 10, item_rect.y + 7))

            if len(selected) == 2:
                start_rect = pygame.Rect(cx - 100, 650, 200, 45)
                start_hovered = start_rect.collidepoint(mouse_pos)
                start_color = (60, 90, 140) if start_hovered else (45, 65, 105)
                pygame.draw.rect(self.surface, start_color, start_rect, border_radius=6)
                start_text = self.font.render("Start Battle!", True, COLOR_BUTTON_TEXT)
                self.surface.blit(start_text, start_text.get_rect(center=start_rect.center))

            pygame.display.flip()
            clock.tick(30)

    # -- Pause Menu -----------------------------------------------------------

    def show_pause_menu(self, mode: str = "isolation", settings: SimSettings | None = None) -> tuple[str, SimSettings | None]:
        """
        Show the pause overlay with options.

        Returns:
            Tuple of (action_string, updated_settings_or_None).
        """
        cx = WINDOW_WIDTH // 2
        cy = WINDOW_HEIGHT // 2
        btn_w, btn_h = 220, 45

        options = ["resume", "restart", "main_menu"]
        labels = ["Resume", "Restart", "Main Menu"]
        colors = [
            ((40, 70, 50), (55, 100, 65)),
            ((55, 55, 55), (75, 75, 75)),
            ((65, 40, 40), (90, 50, 50)),
        ]
        if mode == "isolation":
            options.insert(1, "save_quit")
            labels.insert(1, "Save & Quit")
            colors.insert(1, ((50, 50, 70), (65, 65, 95)))
            options.insert(2, "settings")
            labels.insert(2, "Settings")
            colors.insert(2, ((55, 55, 65), (75, 75, 90)))

        buttons = {}
        for i, (name, label) in enumerate(zip(options, labels)):
            c, hc = colors[i]
            buttons[name] = Button(
                cx - btn_w // 2, cy - 80 + i * 55, btn_w, btn_h, label,
                color=c, hover_color=hc,
            )

        clock = pygame.time.Clock()

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ("main_menu", settings)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        return ("resume", settings)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, btn in buttons.items():
                        if btn.is_clicked(mouse_pos):
                            if name == "settings" and settings is not None:
                                settings = self.show_settings(settings)
                                continue
                            return (name, settings)

            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.surface.blit(overlay, (0, 0))

            title = self.font_heading.render("PAUSED", True, (200, 210, 230))
            self.surface.blit(title, title.get_rect(center=(cx, cy - 140)))

            for btn in buttons.values():
                btn.update(mouse_pos)
                btn.draw(self.surface, self.font)

            pygame.display.flip()
            clock.tick(30)

    # -- Results Screen -------------------------------------------------------

    def show_convergence_results(
        self,
        winner: str,
        a_food: int,
        b_food: int,
        a_gens: int,
        b_gens: int,
    ) -> None:
        """Show convergence mode results."""
        cx = WINDOW_WIDTH // 2
        clock = pygame.time.Clock()
        frame = 0

        while True:
            frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return

            self._draw_menu_bg(frame)

            if winner == "tie":
                title_text = "IT'S A TIE!"
                title_color = (220, 220, 100)
            elif winner == "A":
                title_text = "RED SPECIES WINS!"
                title_color = (255, 90, 90)
            else:
                title_text = "BLUE SPECIES WINS!"
                title_color = (90, 140, 255)

            title = self.font_title.render(title_text, True, title_color)
            self.surface.blit(title, title.get_rect(center=(cx, 200)))

            lines = [
                f"Red (A):  {a_food} food eaten, survived {a_gens} generations",
                f"Blue (B): {b_food} food eaten, survived {b_gens} generations",
                "",
                "Press any key to return to menu",
            ]
            y = 350
            for line in lines:
                text = self.font.render(line, True, COLOR_HUD_TEXT)
                self.surface.blit(text, text.get_rect(center=(cx, y)))
                y += 35

            pygame.display.flip()
            clock.tick(30)

    # -- Utility --------------------------------------------------------------

    def _draw_menu_bg(self, frame: int) -> None:
        """Draw an animated gradient background for menus."""
        self.surface.fill(COLOR_MENU_BG)

        # Subtle animated gradient overlay
        import math
        for y in range(0, WINDOW_HEIGHT, 4):
            t = y / WINDOW_HEIGHT
            wave = math.sin(frame * 0.02 + t * 3) * 0.5 + 0.5
            r = int(15 + wave * 8)
            g = int(15 + t * 10 + wave * 5)
            b = int(30 + t * 15 + wave * 10)
            pygame.draw.line(self.surface, (r, g, b), (0, y), (WINDOW_WIDTH, y))

    def _pulse_color(
        self,
        c1: tuple[int, int, int],
        c2: tuple[int, int, int],
        frame: int,
        period: int,
    ) -> tuple[int, int, int]:
        """Smoothly pulse between two colors."""
        import math
        t = (math.sin(frame * 2 * math.pi / period) + 1) / 2
        return self._lerp_color(c1, c2, t)

    def _lerp_color(
        self,
        c1: tuple[int, int, int],
        c2: tuple[int, int, int],
        t: float,
    ) -> tuple[int, int, int]:
        """Linearly interpolate between two colors."""
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _show_message(self, *lines: str) -> None:
        """Show a simple message screen until ESC is pressed."""
        cx = WINDOW_WIDTH // 2
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            self.surface.fill(COLOR_MENU_BG)
            y = WINDOW_HEIGHT // 2 - len(lines) * 15
            for line in lines:
                text = self.font.render(line, True, COLOR_HUD_TEXT)
                self.surface.blit(text, text.get_rect(center=(cx, y)))
                y += 30

            pygame.display.flip()
            clock.tick(30)
