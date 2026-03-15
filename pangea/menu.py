"""
Menu — pygame menu screens for settings and game management.
============================================================
Provides the main menu, in-app settings panel, and the pause overlay.
"""

from __future__ import annotations

import time as _time
from datetime import datetime
from pathlib import Path

import pygame

from pangea.config import (
    COLOR_BUTTON,
    COLOR_BUTTON_HOVER,
    COLOR_BUTTON_TEXT,
    COLOR_HUD_TEXT,
    COLOR_MENU_BG,
)
import pangea.config as config
from pangea.save_load import delete_save, list_saves, list_species_files
from pangea.settings import SETTING_DEFS, SimSettings
from pangea.species import (
    Species,
    SpeciesSettings,
    SpeciesRegistry,
    default_herbivore,
    default_carnivore,
    default_scavenger,
)


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

    def __init__(
        self,
        surface: pygame.Surface,
        on_toggle_fullscreen: object = None,
        on_toggle_maximized: object = None,
        on_resize: object = None,
    ) -> None:
        self.surface = surface
        self.font = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_title = pygame.font.SysFont("consolas", 52, bold=True)
        self.font_subtitle = pygame.font.SysFont("consolas", 16)
        self.font_heading = pygame.font.SysFont("consolas", 18, bold=True)
        self._on_toggle_fullscreen = on_toggle_fullscreen
        self._on_toggle_maximized = on_toggle_maximized
        self._on_resize = on_resize

    def _handle_window_event(self, event: pygame.event.Event) -> bool:
        """Check for F10/F11/VIDEORESIZE and invoke callbacks. Returns True if handled."""
        if event.type == pygame.VIDEORESIZE:
            if self._on_resize is not None:
                self._on_resize(event.w, event.h)
                self.surface = pygame.display.get_surface()
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11 and self._on_toggle_fullscreen is not None:
                self._on_toggle_fullscreen()
                self.surface = pygame.display.get_surface()
                return True
            if event.key == pygame.K_F10 and self._on_toggle_maximized is not None:
                self._on_toggle_maximized()
                self.surface = pygame.display.get_surface()
                return True
        return False

    # ── Main Menu ────────────────────────────────────────────

    def _build_main_buttons(self) -> dict[str, Button]:
        """Build main menu buttons centered on the current window size."""
        cx = config.WINDOW_WIDTH // 2
        cy = config.WINDOW_HEIGHT // 2
        btn_w, btn_h = 280, 50
        start_y = cy - 130
        return {
            "freeplay": Button(cx - btn_w // 2, start_y, btn_w, btn_h, "Play",
                               color=(40, 70, 50), hover_color=(55, 100, 65)),
            "host": Button(cx - btn_w // 2, start_y + 70, btn_w, btn_h, "Host Game",
                           color=(45, 65, 75), hover_color=(60, 90, 105)),
            "join": Button(cx - btn_w // 2, start_y + 140, btn_w, btn_h, "Join Game",
                           color=(55, 50, 70), hover_color=(75, 70, 100)),
            "settings": Button(cx - btn_w // 2, start_y + 210, btn_w, btn_h, "Settings",
                               color=(55, 55, 65), hover_color=(75, 75, 90)),
            "quit": Button(cx - btn_w // 2, start_y + 280, btn_w, btn_h, "Quit",
                           color=(65, 40, 40), hover_color=(90, 50, 50)),
        }

    def show_main_menu(self, settings: SimSettings | None = None) -> tuple[str, SimSettings]:
        """
        Display the main menu and return the user's choice + settings.

        Returns:
            Tuple of (choice_string, settings_object).
        """
        if settings is None:
            settings = SimSettings()

        buttons = self._build_main_buttons()

        clock = pygame.time.Clock()
        frame = 0

        while True:
            mouse_pos = pygame.mouse.get_pos()
            frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ("quit", settings)
                if self._handle_window_event(event):
                    buttons = self._build_main_buttons()
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return ("quit", settings)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, btn in buttons.items():
                        if btn.is_clicked(mouse_pos):
                            if name == "settings":
                                settings = self.show_settings(settings)
                                buttons = self._build_main_buttons()
                            else:
                                return (name, settings)

            # Draw
            self._draw_menu_bg(frame)

            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2

            # Title with glow effect
            title_color = self._pulse_color((80, 160, 255), (120, 200, 255), frame, 120)
            title = self.font_title.render("PANGEA", True, title_color)
            title_rect = title.get_rect(center=(cx, cy - 210))
            self.surface.blit(title, title_rect)

            # Subtitle
            sub = self.font_subtitle.render(
                "Distributed Evolution Simulator", True, (120, 130, 160)
            )
            sub_rect = sub.get_rect(center=(cx, cy - 150))
            self.surface.blit(sub, sub_rect)

            # Version
            ver = self.font_small.render("v0.1.0", True, (70, 70, 90))
            self.surface.blit(ver, ver.get_rect(center=(cx, cy - 120)))

            for btn in buttons.values():
                btn.update(mouse_pos)
                btn.draw(self.surface, self.font)

            pygame.display.flip()
            clock.tick(30)

    # ── New / Load Save ─────────────────────────────────────

    def show_mode_select(self) -> str | dict | None:
        """
        Show New Game / Load Save screen.

        Returns:
            "new"  — start a new game
            dict   — a loaded save (from load_game)
            None   — user pressed back / ESC
        """
        clock = pygame.time.Clock()
        frame = 0

        while True:
            # Refresh save list each loop iteration (in case of deletion)
            saves = list_saves()

            # Build buttons
            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2
            btn_w, btn_h = 420, 45
            list_top = cy - 80
            max_visible = min(8, (config.WINDOW_HEIGHT - list_top - 140) // 50)

            new_btn = Button(
                cx - btn_w // 2, list_top - 80, btn_w, btn_h, "New Game",
                color=(40, 70, 50), hover_color=(55, 100, 65),
            )
            back_btn = Button(
                cx - 80, config.WINDOW_HEIGHT - 70, 160, 40, "Back",
                color=(65, 40, 40), hover_color=(90, 50, 50),
            )

            # Inner event/draw loop (breaks on save list change)
            scroll_offset = 0
            redraw = True

            while redraw:
                mouse_pos = pygame.mouse.get_pos()
                frame += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if self._handle_window_event(event):
                        break  # rebuild buttons
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return None

                    if event.type == pygame.MOUSEWHEEL and saves:
                        max_scroll = max(0, len(saves) - max_visible)
                        scroll_offset = max(0, min(max_scroll, scroll_offset - event.y))

                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if new_btn.is_clicked(mouse_pos):
                            return "new"
                        if back_btn.is_clicked(mouse_pos):
                            return None

                        # Check save item clicks
                        for i in range(min(max_visible, len(saves) - scroll_offset)):
                            idx = i + scroll_offset
                            item_rect = pygame.Rect(cx - btn_w // 2, list_top + i * 50, btn_w, 42)
                            if item_rect.collidepoint(mouse_pos):
                                if saves[idx].get("snapshot"):
                                    from pangea.save_load import load_snapshot
                                    return load_snapshot(saves[idx]["filepath"])
                                else:
                                    from pangea.save_load import load_game
                                    return load_game(saves[idx]["filepath"])

                    # Right-click to delete a save
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and saves:
                        for i in range(min(max_visible, len(saves) - scroll_offset)):
                            idx = i + scroll_offset
                            item_rect = pygame.Rect(cx - btn_w // 2, list_top + i * 50, btn_w, 42)
                            if item_rect.collidepoint(mouse_pos):
                                name = saves[idx]["save_name"]
                                if self._show_confirm(f"Delete '{name}'?", "This cannot be undone."):
                                    delete_save(saves[idx]["filepath"])
                                    redraw = False  # break to refresh saves list
                                break

                if not redraw:
                    break

                # ── Draw ──
                self._draw_menu_bg(frame)

                # Title
                title_text = "Freeplay"
                title = self.font_heading.render(title_text, True, (140, 170, 220))
                self.surface.blit(title, title.get_rect(center=(cx, list_top - 140)))

                # New Game button
                new_btn.update(mouse_pos)
                new_btn.draw(self.surface, self.font)

                # Save list
                if saves:
                    hint = self.font_small.render(
                        "Left-click: load  |  Right-click: delete  |  Scroll to see more",
                        True, (100, 105, 130),
                    )
                    self.surface.blit(hint, hint.get_rect(center=(cx, list_top - 18)))

                    for i in range(min(max_visible, len(saves) - scroll_offset)):
                        idx = i + scroll_offset
                        save = saves[idx]
                        item_rect = pygame.Rect(cx - btn_w // 2, list_top + i * 50, btn_w, 42)

                        hovered = item_rect.collidepoint(mouse_pos)
                        bg = (45, 48, 65) if hovered else (30, 33, 48)
                        pygame.draw.rect(self.surface, bg, item_rect, border_radius=5)
                        pygame.draw.rect(self.surface, (60, 65, 85), item_rect, 1, border_radius=5)

                        # Save name + generation info
                        label = save["save_name"]
                        gen_info = f"Gen {save['generation']}  |  {save['creature_count']} creatures"
                        name_surf = self.font_subtitle.render(label, True, (200, 210, 235))
                        info_surf = self.font_small.render(gen_info, True, (120, 130, 160))
                        self.surface.blit(name_surf, (item_rect.x + 12, item_rect.y + 5))
                        self.surface.blit(info_surf, (item_rect.x + 12, item_rect.y + 23))

                        # Timestamp on the right
                        ts = save.get("timestamp", "")
                        if ts:
                            # Format: YYYYMMDD_HHMMSS → YYYY-MM-DD HH:MM
                            try:
                                display_ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"
                            except (IndexError, ValueError):
                                display_ts = ts
                            ts_surf = self.font_small.render(display_ts, True, (90, 95, 115))
                            self.surface.blit(ts_surf, (item_rect.right - ts_surf.get_width() - 12, item_rect.y + 14))

                    # Scroll indicator
                    if len(saves) > max_visible:
                        total = len(saves)
                        bar_area_h = max_visible * 50
                        bar_h = max(20, int(bar_area_h * max_visible / total))
                        max_scroll = max(1, total - max_visible)
                        bar_y = list_top + int((bar_area_h - bar_h) * scroll_offset / max_scroll)
                        bar_x = cx + btn_w // 2 + 6
                        pygame.draw.rect(self.surface, (40, 43, 58), (bar_x, list_top, 5, bar_area_h), border_radius=2)
                        pygame.draw.rect(self.surface, (90, 100, 130), (bar_x, bar_y, 5, bar_h), border_radius=2)
                else:
                    no_saves = self.font_subtitle.render("No saved games yet", True, (90, 95, 115))
                    self.surface.blit(no_saves, no_saves.get_rect(center=(cx, list_top + 30)))

                # Back button
                back_btn.update(mouse_pos)
                back_btn.draw(self.surface, self.font)

                pygame.display.flip()
                clock.tick(30)

    # ── Settings Panel ──────────────────────────────────────

    SETTINGS_FILE_DIR = "settings"

    def _settings_layout(self) -> dict:
        """Compute settings panel layout based on current window size."""
        w, h = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
        panel_x = max(40, w // 6)
        panel_w = w - panel_x * 2
        slider_w = min(250, panel_w // 2)
        header_h = 120
        footer_h = 80
        scroll_area_h = h - header_h - footer_h
        btn_y = h - footer_h + 15
        btn_w = 110
        gap = 10
        total_w = btn_w * 4 + gap * 3
        start_x = w // 2 - total_w // 2
        return {
            "panel_x": panel_x, "panel_w": panel_w, "slider_w": slider_w,
            "header_h": header_h, "footer_h": footer_h,
            "scroll_area_h": scroll_area_h, "btn_y": btn_y,
            "back_btn": Button(start_x, btn_y, btn_w, 45, "Back",
                               color=(50, 60, 80), hover_color=(70, 80, 110)),
            "save_btn": Button(start_x + btn_w + gap, btn_y, btn_w, 45, "Save",
                               color=(40, 65, 50), hover_color=(55, 90, 65)),
            "load_btn": Button(start_x + (btn_w + gap) * 2, btn_y, btn_w, 45, "Load",
                               color=(45, 50, 70), hover_color=(60, 68, 95)),
            "reset_btn": Button(start_x + (btn_w + gap) * 3, btn_y, btn_w, 45, "Reset",
                                color=(80, 45, 45), hover_color=(110, 60, 60)),
        }

    def show_settings(self, settings: SimSettings) -> SimSettings:
        """
        Show the in-app settings panel with sliders for all tunable parameters.
        Supports mouse-wheel scrolling when content exceeds the window.

        Returns:
            Updated SimSettings object.
        """
        settings = settings.copy()
        clock = pygame.time.Clock()

        slider_h = 8
        row_h = 36

        # Build slider data with positions relative to content top (0)
        sliders: list[dict] = []
        y = 0
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

        content_height = y
        lay = self._settings_layout()
        max_scroll = max(0, content_height - lay["scroll_area_h"])
        scroll_y = 0

        hovered_tooltip = ""

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return settings
                if self._handle_window_event(event):
                    lay = self._settings_layout()
                    max_scroll = max(0, content_height - lay["scroll_area_h"])
                    scroll_y = min(scroll_y, max_scroll)
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return settings

                if event.type == pygame.MOUSEWHEEL:
                    scroll_y = max(0, min(max_scroll, scroll_y - event.y * 30))

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if lay["back_btn"].is_clicked(mouse_pos):
                        return settings
                    if lay["save_btn"].is_clicked(mouse_pos):
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        name = self._show_text_input("Save Name:", ts)
                        if name:
                            safe = "".join(
                                c if c.isalnum() or c in "-_ " else "_" for c in name
                            )
                            Path(self.SETTINGS_FILE_DIR).mkdir(parents=True, exist_ok=True)
                            filepath = f"{self.SETTINGS_FILE_DIR}/{safe}.json"
                            settings.save_to_file(filepath)
                        lay = self._settings_layout()
                        continue
                    if lay["load_btn"].is_clicked(mouse_pos):
                        def _settings_name(f: Path) -> str:
                            return f.stem.replace("settings_", "")
                        filepath = self._show_file_manager(
                            self.SETTINGS_FILE_DIR,
                            title="SETTINGS FILES",
                            name_transform=_settings_name,
                        )
                        if filepath:
                            try:
                                settings = SimSettings.load_from_file(filepath)
                            except Exception:
                                self._show_message("Failed to load settings file.")
                        lay = self._settings_layout()
                        continue
                    if lay["reset_btn"].is_clicked(mouse_pos):
                        settings = SimSettings()
                        continue

                    panel_x = lay["panel_x"]
                    panel_w = lay["panel_w"]
                    slider_w = lay["slider_w"]
                    header_h = lay["header_h"]
                    footer_h = lay["footer_h"]

                    # Check sliders / toggles
                    for s in sliders:
                        sdef = s["def"]
                        draw_y = s["y"] - scroll_y + header_h
                        if draw_y < header_h - 10 or draw_y > config.WINDOW_HEIGHT - footer_h:
                            continue
                        if sdef.widget_type == "toggle":
                            toggle_rect = pygame.Rect(
                                panel_x + panel_w - slider_w - 40, draw_y, 50, 24
                            )
                            if toggle_rect.collidepoint(mouse_pos):
                                cur = getattr(settings, sdef.key)
                                setattr(settings, sdef.key, not cur)
                        else:
                            slider_rect = pygame.Rect(
                                panel_x + panel_w - slider_w - 40, draw_y, slider_w, slider_h + 16
                            )
                            if slider_rect.collidepoint(mouse_pos):
                                s["dragging"] = True

                if event.type == pygame.MOUSEBUTTONUP:
                    for s in sliders:
                        s["dragging"] = False

            # Read current layout values for drawing
            panel_x = lay["panel_x"]
            panel_w = lay["panel_w"]
            slider_w = lay["slider_w"]
            header_h = lay["header_h"]
            footer_h = lay["footer_h"]
            scroll_area_h = lay["scroll_area_h"]

            # Update dragging sliders (skip toggles)
            for s in sliders:
                if s["dragging"] and s["def"].widget_type != "toggle":
                    sdef = s["def"]
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

            # Title (fixed header)
            title = self.font_heading.render("SIMULATION SETTINGS", True, (140, 160, 200))
            self.surface.blit(title, title.get_rect(center=(config.WINDOW_WIDTH // 2, 80)))

            hint = self.font_small.render(
                "Drag sliders to adjust. Scroll to see more. Changes apply next gen.",
                True, (100, 100, 130),
            )
            self.surface.blit(hint, hint.get_rect(center=(config.WINDOW_WIDTH // 2, 108)))

            # Clip drawing to scroll area
            scroll_clip = pygame.Rect(0, header_h, config.WINDOW_WIDTH, scroll_area_h)
            self.surface.set_clip(scroll_clip)

            # Draw sliders and detect hover for tooltips
            hovered_tooltip = ""
            last_cat = ""
            for s in sliders:
                sdef = s["def"]
                draw_y = s["y"] - scroll_y + header_h

                # Skip off-screen items
                if draw_y < header_h - row_h or draw_y > config.WINDOW_HEIGHT - footer_h + row_h:
                    continue

                # Category header
                if sdef.category != last_cat:
                    last_cat = sdef.category
                    cat_text = self.font_heading.render(sdef.category, True, (100, 140, 200))
                    self.surface.blit(cat_text, (panel_x, draw_y - 6))

                # Label
                label = self.font_small.render(sdef.label, True, (180, 180, 200))
                label_rect = label.get_rect(topleft=(panel_x + 140, draw_y + 2))
                self.surface.blit(label, label_rect)

                # Tooltip hover detection
                if sdef.tooltip and label_rect.collidepoint(mouse_pos):
                    hovered_tooltip = sdef.tooltip

                sx = panel_x + panel_w - slider_w - 40
                val = getattr(settings, sdef.key)

                if sdef.widget_type == "toggle":
                    # Draw toggle switch
                    self._draw_toggle(sx, draw_y + 4, bool(val))
                else:
                    # Slider track
                    sy = draw_y + 8
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
                    self.surface.blit(val_text, (sx + slider_w + 10, draw_y + 2))

            # Remove clip
            self.surface.set_clip(None)

            # Scroll indicator
            if max_scroll > 0:
                bar_h = max(20, int(scroll_area_h * scroll_area_h / content_height))
                bar_y = header_h + int((scroll_area_h - bar_h) * scroll_y / max_scroll)
                bar_x = config.WINDOW_WIDTH - 14
                pygame.draw.rect(self.surface, (50, 55, 70), (bar_x, header_h, 6, scroll_area_h), border_radius=3)
                pygame.draw.rect(self.surface, (100, 110, 140), (bar_x, bar_y, 6, bar_h), border_radius=3)

            # Footer buttons (fixed)
            footer_bg = pygame.Rect(0, config.WINDOW_HEIGHT - footer_h, config.WINDOW_WIDTH, footer_h)
            pygame.draw.rect(self.surface, COLOR_MENU_BG, footer_bg)
            for btn_key in ("back_btn", "save_btn", "load_btn", "reset_btn"):
                lay[btn_key].update(mouse_pos)
                lay[btn_key].draw(self.surface, self.font)

            # Tooltip overlay (drawn last, on top of everything)
            if hovered_tooltip:
                self._draw_tooltip(mouse_pos, hovered_tooltip)

            pygame.display.flip()
            clock.tick(30)

    # ── File Selection ───────────────────────────────────────

    def _load_species_names(self, species_path: Path, files: list[str]) -> dict[str, str]:
        """Load species_name from each JSON file. Returns {filename: display_name}."""
        import json
        names: dict[str, str] = {}
        for fname in files:
            try:
                with open(species_path / fname, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("species_name", "")
                gen = data.get("generation", 0)
                count = len(data.get("creatures", []))
                names[fname] = f"{name}  (gen {gen}, {count} creatures)"
            except Exception:
                names[fname] = fname
        return names

    def _rename_species(self, filepath: Path, new_name: str) -> None:
        """Update the species_name field in a JSON file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["species_name"] = new_name
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _show_text_input(self, prompt: str, initial: str = "") -> str | None:
        """Show a text input dialog. Returns the entered text or None if cancelled."""
        clock = pygame.time.Clock()
        text = initial
        cursor_blink = 0

        while True:
            cursor_blink += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_RETURN:
                        return text.strip()
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        if event.unicode and event.unicode.isprintable() and len(text) < 40:
                            text += event.unicode

            # Dark overlay
            overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.surface.blit(overlay, (0, 0))

            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2
            dialog_w, dialog_h = 460, 140
            dialog_rect = pygame.Rect(cx - dialog_w // 2, cy - dialog_h // 2, dialog_w, dialog_h)
            pygame.draw.rect(self.surface, (25, 28, 40), dialog_rect, border_radius=8)
            pygame.draw.rect(self.surface, (80, 90, 120), dialog_rect, 2, border_radius=8)

            # Prompt
            prompt_surf = self.font_heading.render(prompt, True, (180, 190, 220))
            self.surface.blit(prompt_surf, prompt_surf.get_rect(center=(cx, cy - 35)))

            # Text field
            field_rect = pygame.Rect(cx - 180, cy - 5, 360, 30)
            pygame.draw.rect(self.surface, (15, 17, 28), field_rect, border_radius=4)
            pygame.draw.rect(self.surface, (70, 80, 110), field_rect, 1, border_radius=4)

            cursor_char = "|" if (cursor_blink // 15) % 2 == 0 else ""
            display = text + cursor_char
            text_surf = self.font_subtitle.render(display, True, (220, 225, 240))
            self.surface.blit(text_surf, (field_rect.x + 8, field_rect.y + 7))

            # Hint
            hint = self.font_small.render("Enter to confirm, Esc to cancel", True, (90, 95, 110))
            self.surface.blit(hint, hint.get_rect(center=(cx, cy + 42)))

            pygame.display.flip()
            clock.tick(30)

    # ── Color Picker ──────────────────────────────────────────

    def _show_color_picker(self, prompt: str, initial: tuple[int, int, int]) -> tuple[int, int, int] | None:
        """Show an RGB color picker dialog. Returns (r,g,b) or None if cancelled."""
        clock = pygame.time.Clock()
        r, g, b = initial
        dragging: str | None = None  # "r", "g", or "b"

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_RETURN:
                        return (r, g, b)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = mouse_pos
                    cx = config.WINDOW_WIDTH // 2
                    cy = config.WINDOW_HEIGHT // 2
                    bar_x = cx - 120
                    bar_w = 240
                    for idx, ch in enumerate(("r", "g", "b")):
                        bar_y = cy - 20 + idx * 36
                        if pygame.Rect(bar_x, bar_y, bar_w, 20).collidepoint(mx, my):
                            dragging = ch
                            break
                    # OK button
                    ok_rect = pygame.Rect(cx - 75, cy + 100, 70, 32)
                    if ok_rect.collidepoint(mx, my):
                        return (r, g, b)
                    # Cancel button
                    cancel_rect = pygame.Rect(cx + 5, cy + 100, 70, 32)
                    if cancel_rect.collidepoint(mx, my):
                        return None

            # Handle dragging
            if dragging is not None:
                cx = config.WINDOW_WIDTH // 2
                bar_x = cx - 120
                bar_w = 240
                t = max(0.0, min(1.0, (mouse_pos[0] - bar_x) / bar_w))
                val = int(t * 255)
                if dragging == "r":
                    r = val
                elif dragging == "g":
                    g = val
                else:
                    b = val

            # Draw
            overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.surface.blit(overlay, (0, 0))

            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2
            dialog_w, dialog_h = 360, 260
            dialog_rect = pygame.Rect(cx - dialog_w // 2, cy - dialog_h // 2, dialog_w, dialog_h)
            pygame.draw.rect(self.surface, (25, 28, 40), dialog_rect, border_radius=8)
            pygame.draw.rect(self.surface, (80, 90, 120), dialog_rect, 2, border_radius=8)

            # Title
            title_surf = self.font_heading.render(prompt, True, (180, 190, 220))
            self.surface.blit(title_surf, title_surf.get_rect(center=(cx, cy - 95)))

            # Color preview
            preview_rect = pygame.Rect(cx - 30, cy - 75, 60, 30)
            pygame.draw.rect(self.surface, (r, g, b), preview_rect, border_radius=4)
            pygame.draw.rect(self.surface, (100, 105, 120), preview_rect, 1, border_radius=4)

            # RGB sliders
            bar_x = cx - 120
            bar_w = 240
            channels = [("R", r, (200, 60, 60)), ("G", g, (60, 180, 60)), ("B", b, (60, 80, 200))]
            for idx, (label, val, bar_color) in enumerate(channels):
                bar_y = cy - 20 + idx * 36
                # Label
                lbl = self.font_small.render(label, True, (150, 155, 175))
                self.surface.blit(lbl, (bar_x - 24, bar_y + 2))
                # Track
                pygame.draw.rect(self.surface, (35, 35, 50), (bar_x, bar_y, bar_w, 20), border_radius=6)
                fill = int(bar_w * val / 255)
                if fill > 0:
                    pygame.draw.rect(self.surface, bar_color, (bar_x, bar_y, fill, 20), border_radius=6)
                # Thumb
                pygame.draw.circle(self.surface, (220, 225, 235), (bar_x + fill, bar_y + 10), 8)
                # Value
                val_surf = self.font_small.render(str(val), True, (160, 170, 190))
                self.surface.blit(val_surf, (bar_x + bar_w + 10, bar_y + 2))

            # OK / Cancel buttons
            ok_rect = pygame.Rect(cx - 75, cy + 100, 70, 32)
            ok_h = ok_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.surface, (40, 80, 55) if ok_h else (30, 60, 42), ok_rect, border_radius=5)
            self.surface.blit(self.font_small.render("OK", True, (180, 220, 190)),
                              self.font_small.render("OK", True, (180, 220, 190)).get_rect(center=ok_rect.center))

            cancel_rect = pygame.Rect(cx + 5, cy + 100, 70, 32)
            ch = cancel_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.surface, (70, 45, 45) if ch else (55, 35, 35), cancel_rect, border_radius=5)
            self.surface.blit(self.font_small.render("Cancel", True, (200, 160, 160)),
                              self.font_small.render("Cancel", True, (200, 160, 160)).get_rect(center=cancel_rect.center))

            # Hint
            hint = self.font_small.render("Enter to confirm, Esc to cancel", True, (90, 95, 110))
            self.surface.blit(hint, hint.get_rect(center=(cx, cy + 145)))

            pygame.display.flip()
            clock.tick(30)

    # ── Species Editor ─────────────────────────────────────────

    # Species slider definitions: (attr_name, label, min, max, step, fmt, target, tooltip)
    # target: "sp" = Species attribute, "ss" = SpeciesSettings attribute
    _SP_SLIDERS = [
        # Diet tuning
        ("plant_food_multiplier", "Plant Multiplier", 0.1, 3.0, 0.1, ".1f", "sp",
         "Energy multiplier when eating plants. Higher = more energy per plant."),
        ("attack_damage", "Attack Damage", 0.5, 10.0, 0.5, ".1f", "sp",
         "Damage dealt per attack hit against other creatures."),
        ("energy_steal_fraction", "Energy Steal %", 0.0, 1.0, 0.05, ".2f", "sp",
         "Fraction of damage that is converted into energy for the attacker."),
        ("scavenge_death_radius", "Scavenge Radius", 10, 200, 10, ".0f", "sp",
         "How close a creature must be to a corpse to scavenge energy from it."),
        ("scavenge_death_energy", "Scavenge Energy", 0, 30, 1, ".0f", "sp",
         "Energy gained when a nearby creature dies within scavenge radius."),
        # Population
        ("freeplay_initial_population", "Initial Pop", 1, 100, 1, ".0f", "ss",
         "Number of creatures spawned at the start of a new simulation."),
        ("freeplay_carrying_capacity", "Carry Capacity", 5, 200, 5, ".0f", "ss",
         "Target population. Breeding slows as population approaches this limit."),
        ("freeplay_hard_cap", "Hard Cap", 10, 300, 5, ".0f", "ss",
         "Absolute maximum population. No new births allowed above this number."),
        # Mutation
        ("mutation_rate", "Mutation Rate", 0.01, 1.0, 0.05, ".2f", "ss",
         "Probability that each neural network weight mutates during reproduction."),
        ("mutation_strength", "Mutation Str.", 0.05, 2.0, 0.05, ".2f", "ss",
         "How much each mutated weight can change. Higher = more dramatic mutations."),
        ("crossover_rate", "Crossover Rate", 0.0, 1.0, 0.05, ".2f", "ss",
         "Chance of mixing genes from two parents instead of cloning one parent."),
        ("trait_mutation_range", "Trait Mut Range", 0, 20, 1, ".0f", "ss",
         "Max points a genetic trait (size, speed, sense) can shift per generation."),
        # Breeding
        ("freeplay_breed_min_age", "Breed Min Age", 1, 30, 1, ".0f", "ss",
         "Minimum age (in seconds) before a creature is old enough to breed."),
        ("freeplay_breed_min_food", "Breed Min Food", 1, 20, 1, ".0f", "ss",
         "Number of food items a creature must have eaten before it can breed."),
        ("freeplay_breed_energy_threshold", "Breed Energy %", 0.1, 1.0, 0.05, ".2f", "ss",
         "Minimum energy level (as fraction of max) required to breed."),
        ("freeplay_breed_cooldown", "Breed Cooldown", 1, 60, 1, ".0f", "ss",
         "Seconds a creature must wait after breeding before it can breed again."),
        ("freeplay_breed_energy_cost", "Breed Cost", 5, 100, 5, ".0f", "ss",
         "Energy spent by the parent when producing an offspring."),
        ("freeplay_child_energy", "Child Energy", 10, 200, 10, ".0f", "ss",
         "Starting energy given to a newborn creature."),
        # Creature
        ("base_energy", "Start Energy", 20, 500, 10, ".0f", "ss",
         "Energy each creature starts with."),
        ("energy_cost_per_thrust", "Move Cost", 0.01, 0.5, 0.01, ".2f", "ss",
         "Energy drained per unit of thrust each frame."),
        ("turn_cost", "Turn Cost", 0.0, 0.5, 0.01, ".2f", "ss",
         "Extra energy cost for turning (0 = free turning)."),
        ("food_heal", "Food Heal (sec)", 0.0, 10.0, 0.5, ".1f", "ss",
         "Lifespan seconds restored per food eaten."),
        # Night Vision
        ("night_vision_multiplier", "Night Vision", 0.0, 1.0, 0.05, ".2f", "ss",
         "Vision multiplier at night (1.0 = no reduction)."),
        # Fitness
        ("fitness_food_weight", "Food Weight", 0.0, 50.0, 1.0, ".1f", "ss",
         "Food eaten contribution to fitness score."),
        ("fitness_time_weight", "Survival Weight", 0.0, 5.0, 0.05, ".2f", "ss",
         "Time alive contribution to fitness score."),
        ("fitness_energy_weight", "Energy Weight", 0.0, 5.0, 0.05, ".2f", "ss",
         "Remaining energy contribution to fitness score."),
        ("territory_fitness_weight", "Territory Weight", 0.0, 5.0, 0.1, ".1f", "ss",
         "Area explored contribution to fitness score."),
        ("fitness_offspring_weight", "Offspring Weight", 0.0, 20.0, 0.5, ".1f", "ss",
         "Breeding success contribution to fitness score."),
        ("fitness_distance_weight", "Distance Weight", 0.0, 2.0, 0.05, ".2f", "ss",
         "Distance traveled contribution to fitness (rewards movement over spinning)."),
    ]

    # Tooltip descriptions for diet flag toggles
    _DIET_FLAG_TOOLTIPS = {
        "can_eat_plants": "Whether this species can eat plant food items for energy.",
        "can_attack_other_species": "Whether this species can attack creatures of different species.",
        "can_attack_own_species": "Whether this species can attack creatures of the same species.",
        "can_eat_other_corpse": "Whether this species can eat corpses of other species for energy.",
        "can_eat_own_corpse": "Whether this species can eat corpses of its own species for energy.",
    }

    _EXTINCTION_TOOLTIP = "What happens when this species goes extinct. Respawn Best: revive from top DNA. Respawn Random: revive with random DNA. Permanent: species is gone forever."

    SPECIES_DIR = "species_settings"

    def _species_card_height(self) -> int:
        """Height of a single species card in the editor."""
        # Header(30) + action row(28) + diet flags(3 rows * 28) + sliders(len * 22) + padding
        return 30 + 28 + 84 + len(self._SP_SLIDERS) * 22 + 16

    def show_species_editor(self, settings: SimSettings) -> SimSettings:
        """
        Full-screen species editor for creating, editing, and removing species.

        Shows all species in a scrollable list with diet flags, combat/scavenge
        tuning, population, mutation, breeding, and extinction settings.
        """
        clock = pygame.time.Clock()
        registry = settings.species_registry
        scroll_y = 0
        dragging_slider: tuple | None = None  # (species_id, attr, target, min, max, step)
        hovered_tooltip = ""

        palette = [
            (80, 200, 80), (200, 60, 60), (180, 140, 50),
            (60, 140, 220), (200, 100, 200), (100, 200, 200),
            (220, 160, 60), (140, 200, 100), (200, 80, 140),
        ]

        card_h = self._species_card_height()

        while True:
            mouse_pos = pygame.mouse.get_pos()
            species_list = registry.all()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return settings
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return settings
                if event.type == pygame.MOUSEWHEEL:
                    scroll_y = max(0, scroll_y - event.y * 30)

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging_slider = None

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = mouse_pos

                    # Back button
                    back_rect = pygame.Rect(20, config.WINDOW_HEIGHT - 50, 100, 36)
                    if back_rect.collidepoint(mx, my):
                        return settings

                    # Add Species button
                    add_rect = pygame.Rect(config.WINDOW_WIDTH - 180, config.WINDOW_HEIGHT - 50, 160, 36)
                    if add_rect.collidepoint(mx, my):
                        name = self._show_text_input("Species Name:", "New Species")
                        if name:
                            slug = name.lower().replace(" ", "_")
                            uid = registry.generate_unique_id(slug)
                            color = palette[len(species_list) % len(palette)]
                            new_sp = Species(
                                id=uid, name=name, color=color,
                                settings=SpeciesSettings(),
                            )
                            registry.register(new_sp)
                        continue

                    # Check per-species card interactions
                    card_x = 30
                    card_w = config.WINDOW_WIDTH - 60
                    y_pos = 70 - scroll_y
                    for i, sp in enumerate(species_list):
                        if y_pos + card_h < 0 or y_pos > config.WINDOW_HEIGHT:
                            y_pos += card_h + 12
                            continue

                        # Delete button
                        del_rect = pygame.Rect(card_x + card_w - 70, y_pos + 6, 60, 22)
                        if del_rect.collidepoint(mx, my) and len(species_list) > 1:
                            if self._show_confirm(f"Delete {sp.name}?", "Creatures of this species will be removed."):
                                registry.remove(sp.id)
                            break

                        # Rename button
                        rename_rect = pygame.Rect(card_x + card_w - 140, y_pos + 6, 60, 22)
                        if rename_rect.collidepoint(mx, my):
                            new_name = self._show_text_input("Rename Species:", sp.name)
                            if new_name:
                                sp.name = new_name
                            break

                        # Color swatch button
                        color_rect = pygame.Rect(card_x + card_w - 210, y_pos + 6, 60, 22)
                        if color_rect.collidepoint(mx, my):
                            new_color = self._show_color_picker("Species Color", sp.color)
                            if new_color is not None:
                                sp.color = new_color
                            break

                        # Enabled toggle
                        enabled_rect = pygame.Rect(card_x + 16, y_pos + 30, 50, 20)
                        if enabled_rect.collidepoint(mx, my):
                            sp.enabled = not sp.enabled
                            break

                        # Save species settings button
                        save_sp_rect = pygame.Rect(card_x + card_w - 220, y_pos + 30, 60, 22)
                        if save_sp_rect.collidepoint(mx, my):
                            self._save_species_settings(sp)
                            break

                        # Load species settings button
                        load_sp_rect = pygame.Rect(card_x + card_w - 150, y_pos + 30, 60, 22)
                        if load_sp_rect.collidepoint(mx, my):
                            self._load_species_settings(sp)
                            break

                        # Diet flag toggles
                        toggle_y = y_pos + 58
                        toggle_x_l = card_x + 20
                        toggle_x_r = card_x + card_w // 2 + 10
                        tw = 50
                        diet_flags = [
                            ("can_eat_plants", toggle_x_l, toggle_y),
                            ("can_attack_other_species", toggle_x_r, toggle_y),
                            ("can_attack_own_species", toggle_x_l, toggle_y + 28),
                            ("can_eat_other_corpse", toggle_x_r, toggle_y + 28),
                            ("can_eat_own_corpse", toggle_x_l, toggle_y + 56),
                        ]
                        toggled = False
                        for flag_name, fx, fy in diet_flags:
                            tr = pygame.Rect(fx + 160, fy, tw, 20)
                            if tr.collidepoint(mx, my):
                                setattr(sp, flag_name, not getattr(sp, flag_name))
                                toggled = True
                                break
                        if toggled:
                            break

                        # Extinction mode toggle (cycle on click)
                        ext_y = toggle_y + 56
                        ext_rect = pygame.Rect(toggle_x_r + 160, ext_y, 120, 20)
                        if ext_rect.collidepoint(mx, my):
                            from pangea.species import EXTINCTION_MODES
                            try:
                                idx = EXTINCTION_MODES.index(sp.settings.extinction_mode)
                            except ValueError:
                                idx = 0
                            sp.settings.extinction_mode = EXTINCTION_MODES[(idx + 1) % len(EXTINCTION_MODES)]
                            break

                        # Sliders
                        slider_base_y = toggle_y + 84
                        slider_x = card_x + 170
                        slider_w = card_w - 250
                        clicked_slider = False
                        for si, (attr, label, smin, smax, step, fmt, target, _tip) in enumerate(self._SP_SLIDERS):
                            sy = slider_base_y + si * 22
                            sr = pygame.Rect(slider_x, sy, slider_w, 14)
                            if sr.collidepoint(mx, my):
                                dragging_slider = (sp.id, attr, target, smin, smax, step)
                                clicked_slider = True
                                break
                        if clicked_slider:
                            break

                        y_pos += card_h + 12

            # Handle slider dragging
            if dragging_slider is not None:
                sp_id, attr, target, smin, smax, step = dragging_slider
                sp = registry.get(sp_id)
                if sp:
                    card_x = 30
                    card_w = config.WINDOW_WIDTH - 60
                    slider_x = card_x + 170
                    slider_w = card_w - 250
                    t = max(0.0, min(1.0, (mouse_pos[0] - slider_x) / slider_w))
                    raw = smin + t * (smax - smin)
                    snapped = round(raw / step) * step
                    snapped = max(smin, min(smax, snapped))
                    if step >= 1:
                        snapped = int(snapped)
                    obj = sp if target == "sp" else sp.settings
                    setattr(obj, attr, snapped)

            # ── Draw ──
            self.surface.fill(COLOR_MENU_BG)

            title = self.font_heading.render("SPECIES EDITOR", True, (180, 190, 220))
            self.surface.blit(title, (30, 20))
            hint = self.font_small.render(
                f"{len(species_list)} species  |  ESC to go back", True, (100, 105, 130),
            )
            self.surface.blit(hint, (30, 44))

            card_x = 30
            card_w = config.WINDOW_WIDTH - 60
            y_pos = 70 - scroll_y
            clip_rect = pygame.Rect(0, 60, config.WINDOW_WIDTH, config.WINDOW_HEIGHT - 120)
            self.surface.set_clip(clip_rect)

            hovered_tooltip = ""

            for i, sp in enumerate(species_list):
                if y_pos + card_h < 60 or y_pos > config.WINDOW_HEIGHT - 60:
                    y_pos += card_h + 12
                    continue

                # Card background
                cr = pygame.Rect(card_x, y_pos, card_w, card_h)
                pygame.draw.rect(self.surface, (25, 28, 40), cr, border_radius=6)
                pygame.draw.rect(self.surface, (50, 55, 70), cr, 1, border_radius=6)
                pygame.draw.rect(self.surface, sp.color, (card_x, y_pos, 6, card_h), border_radius=3)

                # Header
                ns = self.font_heading.render(sp.name, True, sp.color)
                self.surface.blit(ns, (card_x + 16, y_pos + 6))
                ids = self.font_small.render(f"({sp.id})", True, (90, 95, 115))
                self.surface.blit(ids, (card_x + 18 + ns.get_width() + 6, y_pos + 10))

                # Color/Rename/Delete buttons
                color_rect = pygame.Rect(card_x + card_w - 210, y_pos + 6, 60, 22)
                clh = color_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.surface, (50, 55, 75) if clh else (35, 38, 52), color_rect, border_radius=4)
                # Color swatch preview inside the button
                swatch_rect = pygame.Rect(color_rect.x + 4, color_rect.y + 4, 14, 14)
                pygame.draw.rect(self.surface, sp.color, swatch_rect, border_radius=3)
                clabel = self.font_small.render("Color", True, (160, 165, 185))
                self.surface.blit(clabel, (color_rect.x + 22, color_rect.y + 3))
                if clh:
                    hovered_tooltip = "Change the display color of this species."

                rename_rect = pygame.Rect(card_x + card_w - 140, y_pos + 6, 60, 22)
                rh = rename_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.surface, (50, 55, 75) if rh else (35, 38, 52), rename_rect, border_radius=4)
                self.surface.blit(self.font_small.render("Rename", True, (160, 165, 185)),
                                  self.font_small.render("Rename", True, (160, 165, 185)).get_rect(center=rename_rect.center))
                if rh:
                    hovered_tooltip = "Change the display name of this species."
                if len(species_list) > 1:
                    del_rect = pygame.Rect(card_x + card_w - 70, y_pos + 6, 60, 22)
                    dh = del_rect.collidepoint(mouse_pos)
                    pygame.draw.rect(self.surface, (90, 40, 40) if dh else (60, 35, 35), del_rect, border_radius=4)
                    self.surface.blit(self.font_small.render("Delete", True, (200, 140, 140)),
                                      self.font_small.render("Delete", True, (200, 140, 140)).get_rect(center=del_rect.center))
                    if dh:
                        hovered_tooltip = "Remove this species and all its creatures from the simulation."

                # Enabled toggle + Save/Load buttons row
                action_y = y_pos + 30
                self.surface.blit(self.font_small.render("Enabled", True, (150, 155, 175)), (card_x + 76, action_y + 2))
                self._draw_toggle(card_x + 16, action_y, sp.enabled)
                enabled_area = pygame.Rect(card_x + 16, action_y, 130, 20)
                if enabled_area.collidepoint(mouse_pos):
                    hovered_tooltip = "Pause this species: no new births and no extinction respawn while disabled."

                # Dimmed overlay hint when disabled
                if not sp.enabled:
                    dim_label = self.font_small.render("(paused)", True, (200, 130, 130))
                    self.surface.blit(dim_label, (card_x + 160, action_y + 2))

                # Save/Load species settings buttons
                save_sp_rect = pygame.Rect(card_x + card_w - 220, action_y, 60, 22)
                sh = save_sp_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.surface, (40, 65, 50) if sh else (30, 50, 38), save_sp_rect, border_radius=4)
                self.surface.blit(self.font_small.render("Save", True, (140, 200, 160)),
                                  self.font_small.render("Save", True, (140, 200, 160)).get_rect(center=save_sp_rect.center))
                if sh:
                    hovered_tooltip = "Save this species' settings to a file."

                load_sp_rect = pygame.Rect(card_x + card_w - 150, action_y, 60, 22)
                lh = load_sp_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.surface, (45, 50, 70) if lh else (35, 40, 55), load_sp_rect, border_radius=4)
                self.surface.blit(self.font_small.render("Load", True, (160, 175, 210)),
                                  self.font_small.render("Load", True, (160, 175, 210)).get_rect(center=load_sp_rect.center))
                if lh:
                    hovered_tooltip = "Load species settings from a file."

                # Diet flag toggles
                toggle_y = y_pos + 58
                tx_l = card_x + 20
                tx_r = card_x + card_w // 2 + 10
                diet_flags = [
                    ("can_eat_plants", "Eat Plants", tx_l, toggle_y),
                    ("can_attack_other_species", "Attack Others", tx_r, toggle_y),
                    ("can_attack_own_species", "Attack Own", tx_l, toggle_y + 28),
                    ("can_eat_other_corpse", "Eat Other Corpse", tx_r, toggle_y + 28),
                    ("can_eat_own_corpse", "Eat Own Corpse", tx_l, toggle_y + 56),
                ]
                for flag_name, flag_label, fx, fy in diet_flags:
                    val = getattr(sp, flag_name)
                    label_surf = self.font_small.render(flag_label, True, (150, 155, 175))
                    self.surface.blit(label_surf, (fx, fy + 2))
                    self._draw_toggle(fx + 160, fy, val)
                    # Hover detection on label + toggle area
                    flag_area = pygame.Rect(fx, fy, 160 + 50, 20)
                    if flag_area.collidepoint(mouse_pos):
                        hovered_tooltip = self._DIET_FLAG_TOOLTIPS.get(flag_name, "")

                # Extinction mode (clickable cycle)
                ext_y = toggle_y + 56
                self.surface.blit(self.font_small.render("Extinction", True, (150, 155, 175)), (tx_r, ext_y + 2))
                ext_display = sp.settings.extinction_mode.replace("_", " ").title()
                ext_rect = pygame.Rect(tx_r + 160, ext_y, 120, 20)
                eh = ext_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.surface, (40, 45, 60) if eh else (30, 33, 45), ext_rect, border_radius=4)
                et = self.font_small.render(f"< {ext_display} >", True, (140, 200, 170))
                self.surface.blit(et, et.get_rect(center=ext_rect.center))
                # Hover on extinction label or selector
                ext_full = pygame.Rect(tx_r, ext_y, 160 + 120, 20)
                if ext_full.collidepoint(mouse_pos):
                    hovered_tooltip = self._EXTINCTION_TOOLTIP

                # Sliders
                slider_base_y = toggle_y + 84
                slider_x = card_x + 170
                slider_w = card_w - 250

                for si, (attr, label, smin, smax, step, fmt, target, tip) in enumerate(self._SP_SLIDERS):
                    sy = slider_base_y + si * 22
                    obj = sp if target == "sp" else sp.settings
                    val = getattr(obj, attr)
                    t = max(0, min(1, (val - smin) / (smax - smin))) if smax > smin else 0

                    label_surf = self.font_small.render(label, True, (130, 135, 155))
                    label_pos = (card_x + 16, sy + 1)
                    self.surface.blit(label_surf, label_pos)
                    pygame.draw.rect(self.surface, (35, 35, 50), (slider_x, sy + 4, slider_w, 8), border_radius=4)
                    fill = int(slider_w * t)
                    if fill > 0:
                        pygame.draw.rect(self.surface, (50, 100, 130), (slider_x, sy + 4, fill, 8), border_radius=4)
                    pygame.draw.circle(self.surface, (180, 190, 210), (slider_x + fill, sy + 8), 5)
                    val_str = f"{val:{fmt}}"
                    self.surface.blit(self.font_small.render(val_str, True, (160, 170, 190)), (slider_x + slider_w + 8, sy + 1))

                    # Hover detection on slider row
                    row_rect = pygame.Rect(card_x + 16, sy, slider_x + slider_w + 60 - card_x - 16, 18)
                    if tip and row_rect.collidepoint(mouse_pos):
                        hovered_tooltip = tip

                y_pos += card_h + 12

            self.surface.set_clip(None)

            # Footer
            footer_y = config.WINDOW_HEIGHT - 58
            pygame.draw.rect(self.surface, COLOR_MENU_BG, (0, footer_y, config.WINDOW_WIDTH, 58))
            pygame.draw.line(self.surface, (50, 55, 70), (0, footer_y), (config.WINDOW_WIDTH, footer_y))

            back_rect = pygame.Rect(20, config.WINDOW_HEIGHT - 50, 100, 36)
            bh = back_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.surface, (55, 60, 80) if bh else (40, 44, 58), back_rect, border_radius=5)
            self.surface.blit(self.font_small.render("Back", True, (180, 185, 200)),
                              self.font_small.render("Back", True, (180, 185, 200)).get_rect(center=back_rect.center))
            if bh:
                hovered_tooltip = "Return to the previous menu. Changes are kept."

            add_rect = pygame.Rect(config.WINDOW_WIDTH - 180, config.WINDOW_HEIGHT - 50, 160, 36)
            ah = add_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.surface, (40, 80, 55) if ah else (30, 60, 42), add_rect, border_radius=5)
            self.surface.blit(self.font_small.render("+ Add Species", True, (140, 220, 160)),
                              self.font_small.render("+ Add Species", True, (140, 220, 160)).get_rect(center=add_rect.center))
            if ah:
                hovered_tooltip = "Create a new custom species with default settings."

            # Draw tooltip on top of everything
            if hovered_tooltip:
                self._draw_tooltip(mouse_pos, hovered_tooltip)

            pygame.display.flip()
            clock.tick(30)

    def show_import_species(self, settings: SimSettings) -> tuple[SimSettings, str | None]:
        """
        Show a file picker for importing a species into the world.

        Returns:
            Tuple of (updated settings, species_id of imported species or None).
        """
        files = list_species_files("species")
        if not files:
            self._show_message("No species files found.", "Save species to the species/ folder first.")
            return settings, None

        clock = pygame.time.Clock()
        scroll = 0

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return settings, None
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return settings, None
                if event.type == pygame.MOUSEWHEEL:
                    scroll = max(0, scroll - event.y)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = mouse_pos

                    # Back button
                    back_rect = pygame.Rect(20, config.WINDOW_HEIGHT - 50, 100, 36)
                    if back_rect.collidepoint(mx, my):
                        return settings, None

                    # File items
                    max_visible = (config.WINDOW_HEIGHT - 120) // 34
                    for idx in range(min(max_visible, len(files) - scroll)):
                        fi = idx + scroll
                        item_rect = pygame.Rect(30, 70 + idx * 34, config.WINDOW_WIDTH - 60, 30)
                        if item_rect.collidepoint(mx, my):
                            filepath = files[fi]
                            return self._do_import_species(settings, filepath)

            # Draw
            self.surface.fill(COLOR_MENU_BG)
            title = self.font_heading.render("IMPORT SPECIES", True, (180, 190, 220))
            self.surface.blit(title, (30, 20))
            hint = self.font_small.render("Select a species file to import into the world", True, (100, 105, 130))
            self.surface.blit(hint, (30, 44))

            max_visible = (config.WINDOW_HEIGHT - 120) // 34
            for idx in range(min(max_visible, len(files) - scroll)):
                fi = idx + scroll
                item_rect = pygame.Rect(30, 70 + idx * 34, config.WINDOW_WIDTH - 60, 30)
                hovered = item_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.surface, (40, 44, 58) if hovered else (28, 31, 42), item_rect, border_radius=4)
                pygame.draw.rect(self.surface, (50, 55, 70), item_rect, 1, border_radius=4)
                display = Path(files[fi]).stem
                ft = self.font_small.render(display, True, (180, 185, 200))
                self.surface.blit(ft, (item_rect.x + 10, item_rect.y + 7))

            # Back button
            back_rect = pygame.Rect(20, config.WINDOW_HEIGHT - 50, 100, 36)
            back_hover = back_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.surface, (55, 60, 80) if back_hover else (40, 44, 58), back_rect, border_radius=5)
            bt = self.font_small.render("Back", True, (180, 185, 200))
            self.surface.blit(bt, bt.get_rect(center=back_rect.center))

            pygame.display.flip()
            clock.tick(30)

    def _do_import_species(
        self, settings: SimSettings, filepath: str,
    ) -> tuple[SimSettings, str | None]:
        """Import a species file, register it, and return the new species_id."""
        import json
        from pangea.dna import DNA

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._show_message("Failed to read file.")
            return settings, None

        species_name = data.get("species_name", Path(filepath).stem)
        dna_list = [DNA.from_dict(c) for c in data.get("creatures", [])]
        if not dna_list:
            self._show_message("No creatures found in file.")
            return settings, None

        # Ask for species name
        name = self._show_text_input("Species Name:", species_name)
        if not name:
            return settings, None

        registry = settings.species_registry
        slug = name.lower().replace(" ", "_")
        uid = registry.generate_unique_id(slug)

        # Pick a color not already in use
        used_colors = {sp.color for sp in registry.all()}
        palette = [
            (60, 140, 220), (200, 100, 200), (100, 200, 200),
            (220, 160, 60), (140, 200, 100), (200, 80, 140),
            (80, 200, 80), (200, 60, 60), (180, 140, 50),
        ]
        color = (120, 120, 180)
        for c in palette:
            if c not in used_colors:
                color = c
                break

        new_sp = Species(
            id=uid, name=name, color=color,
            settings=SpeciesSettings(
                freeplay_initial_population=len(dna_list),
                freeplay_carrying_capacity=len(dna_list) * 2,
                freeplay_hard_cap=len(dna_list) * 3,
            ),
        )
        registry.register(new_sp)

        # Update DNA to point to new species
        for dna in dna_list:
            dna.species_id = uid

        # Store imported DNA on the species for the simulation to spawn
        new_sp._imported_dna = dna_list

        return settings, uid

    # ── Pause Menu ───────────────────────────────────────────

    def _build_pause_buttons(self) -> tuple[list[str], dict[str, Button]]:
        """Build pause menu buttons centered on the current window size."""
        cx = config.WINDOW_WIDTH // 2
        cy = config.WINDOW_HEIGHT // 2
        btn_w, btn_h = 220, 45

        options = ["resume", "save_quit", "settings", "species_editor", "import_species", "restart", "main_menu"]
        labels = ["Resume", "Save & Quit", "Settings", "Species Editor", "Import Species", "Restart", "Main Menu"]
        colors = [
            ((40, 70, 50), (55, 100, 65)),
            ((50, 50, 70), (65, 65, 95)),
            ((55, 55, 65), (75, 75, 90)),
            ((50, 55, 70), (70, 78, 100)),
            ((45, 60, 55), (60, 85, 70)),
            ((55, 55, 55), (75, 75, 75)),
            ((65, 40, 40), (90, 50, 50)),
        ]

        buttons = {}
        for i, (name, label) in enumerate(zip(options, labels)):
            c, hc = colors[i]
            buttons[name] = Button(
                cx - btn_w // 2, cy - 80 + i * 55, btn_w, btn_h, label,
                color=c, hover_color=hc,
            )
        return options, buttons

    def show_pause_menu(self, settings: SimSettings | None = None) -> tuple[str, SimSettings | None]:
        """
        Show the pause overlay with options.

        Returns:
            Tuple of (action_string, updated_settings_or_None).
        """
        _, buttons = self._build_pause_buttons()
        clock = pygame.time.Clock()

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ("main_menu", settings)
                if self._handle_window_event(event):
                    _, buttons = self._build_pause_buttons()
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        return ("resume", settings)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, btn in buttons.items():
                        if btn.is_clicked(mouse_pos):
                            if name == "settings" and settings is not None:
                                settings = self.show_settings(settings)
                                _, buttons = self._build_pause_buttons()
                                continue
                            if name == "species_editor" and settings is not None:
                                settings = self.show_species_editor(settings)
                                _, buttons = self._build_pause_buttons()
                                continue
                            if name == "import_species" and settings is not None:
                                settings, imported_id = self.show_import_species(settings)
                                if imported_id:
                                    return ("import_species:" + imported_id, settings)
                                _, buttons = self._build_pause_buttons()
                                continue
                            if name == "restart":
                                if not self._show_confirm("Restart simulation?", "All progress will be lost."):
                                    continue
                            return (name, settings)

            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2

            overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.surface.blit(overlay, (0, 0))

            title = self.font_heading.render("PAUSED", True, (200, 210, 230))
            self.surface.blit(title, title.get_rect(center=(cx, cy - 140)))

            for btn in buttons.values():
                btn.update(mouse_pos)
                btn.draw(self.surface, self.font)

            pygame.display.flip()
            clock.tick(30)

    # ── Error Dialog ────────────────────────────────────────

    def show_error(self, message: str) -> None:
        """Show an error message overlay and wait for keypress/click to dismiss."""
        clock = pygame.time.Clock()
        frame = 0

        while True:
            frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return

            self._draw_menu_bg(frame)

            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2

            title = self.font_title.render("Error", True, (255, 90, 90))
            self.surface.blit(title, title.get_rect(center=(cx, cy - 60)))

            msg = self.font.render(message, True, COLOR_HUD_TEXT)
            self.surface.blit(msg, msg.get_rect(center=(cx, cy + 10)))

            hint = self.font.render("Press any key to return", True, (140, 140, 160))
            self.surface.blit(hint, hint.get_rect(center=(cx, cy + 60)))

            pygame.display.flip()
            clock.tick(30)

    # ── Multiplayer Menus ──────────────────────────────────────

    def show_host_setup(self, settings: SimSettings) -> tuple[str, str, SimSettings] | None:
        """
        Show host game setup screen — pick relay URL.

        Returns:
            Tuple of (game_type, relay_url, settings) or None if cancelled.
        """
        from pangea.config import NET_DEFAULT_RELAY

        clock = pygame.time.Clock()
        frame = 0
        relay_url = NET_DEFAULT_RELAY

        cx = config.WINDOW_WIDTH // 2
        cy = config.WINDOW_HEIGHT // 2
        btn_w, btn_h = 240, 50

        def _build_buttons():
            nonlocal cx, cy
            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2
            return {
                "freeplay": Button(cx - btn_w // 2, cy - 20, btn_w, btn_h, "Host Freeplay",
                                   color=(40, 70, 50), hover_color=(55, 100, 65)),
                "relay": Button(cx - btn_w // 2, cy + 60, btn_w, btn_h, "Change Relay URL",
                                color=(55, 55, 65), hover_color=(75, 75, 90)),
                "back": Button(cx - btn_w // 2, cy + 130, btn_w, btn_h, "Back",
                               color=(65, 40, 40), hover_color=(90, 50, 50)),
            }

        buttons = _build_buttons()

        while True:
            mouse_pos = pygame.mouse.get_pos()
            frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if self._handle_window_event(event):
                    buttons = _build_buttons()
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, btn in buttons.items():
                        if btn.is_clicked(mouse_pos):
                            if name == "back":
                                return None
                            if name == "relay":
                                new_url = self._show_text_input("Relay Server URL:", relay_url)
                                if new_url:
                                    relay_url = new_url
                                buttons = _build_buttons()
                            elif name == "freeplay":
                                return (name, relay_url, settings)

            self._draw_menu_bg(frame)

            title = self.font_heading.render("HOST GAME", True, (100, 180, 220))
            self.surface.blit(title, title.get_rect(center=(cx, cy - 120)))

            sub = self.font.render("Host a freeplay game:", True, (160, 170, 190))
            self.surface.blit(sub, sub.get_rect(center=(cx, cy - 70)))

            # Show current relay URL
            url_text = self.font_small.render(f"Relay: {relay_url}", True, (120, 130, 150))
            self.surface.blit(url_text, url_text.get_rect(center=(cx, cy + 110)))

            for btn in buttons.values():
                btn.update(mouse_pos)
                btn.draw(self.surface, self.font)

            pygame.display.flip()
            clock.tick(30)

    def show_join_dialog(self) -> tuple[str, str] | None:
        """
        Show join game dialog — enter room code and host IP.

        Returns:
            Tuple of (room_code, relay_url) or None if cancelled.
        """
        # Get room code
        room_code = self._show_text_input("Enter Room Code:")
        if not room_code:
            return None

        room_code = room_code.strip().upper()

        # Get host IP address (default to localhost for same-machine testing)
        host_ip = self._show_text_input("Host IP Address:", "localhost")
        if not host_ip:
            return None

        host_ip = host_ip.strip()

        # Build the websocket URL from the IP
        # If user entered a full ws:// URL, use it as-is
        if host_ip.startswith("ws://") or host_ip.startswith("wss://"):
            relay_url = host_ip
        else:
            relay_url = f"ws://{host_ip}:8765"

        return (room_code, relay_url)

    def show_waiting_room(self, room_code: str, player_count: int, host_ip: str = "") -> str | None:
        """
        Show the waiting room screen while host waits for clients.

        Called in a loop by simulation.py — returns immediately each frame.

        Args:
            room_code:    The room code to display.
            player_count: Number of connected clients.
            host_ip:      The host's LAN IP to display.

        Returns:
            "start" if Start pressed, "cancel" if cancelled, None to keep waiting.
        """
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "cancel"
            if self._handle_window_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "cancel"
                if event.key == pygame.K_RETURN and player_count > 0:
                    return "start"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cx = config.WINDOW_WIDTH // 2
                cy = config.WINDOW_HEIGHT // 2
                # Start button area
                start_rect = pygame.Rect(cx - 120, cy + 80, 110, 45)
                cancel_rect = pygame.Rect(cx + 10, cy + 80, 110, 45)
                if start_rect.collidepoint(mouse_pos) and player_count > 0:
                    return "start"
                if cancel_rect.collidepoint(mouse_pos):
                    return "cancel"

        # Draw
        cx = config.WINDOW_WIDTH // 2
        cy = config.WINDOW_HEIGHT // 2

        self.surface.fill((20, 20, 35))

        title = self.font_heading.render("WAITING FOR PLAYERS", True, (100, 180, 220))
        self.surface.blit(title, title.get_rect(center=(cx, cy - 100)))

        # Room code display (large)
        code_text = self.font_title.render(room_code, True, (220, 230, 255))
        self.surface.blit(code_text, code_text.get_rect(center=(cx, cy - 30)))

        label = self.font_small.render("Room Code", True, (120, 130, 150))
        self.surface.blit(label, label.get_rect(center=(cx, cy - 70)))

        # Player count
        count_text = self.font.render(
            f"{player_count} player{'s' if player_count != 1 else ''} connected",
            True, (160, 200, 160) if player_count > 0 else (160, 140, 140),
        )
        self.surface.blit(count_text, count_text.get_rect(center=(cx, cy + 30)))

        # Start button
        start_color = (40, 70, 50) if player_count > 0 else (40, 40, 40)
        start_text_color = (220, 230, 240) if player_count > 0 else (100, 100, 100)
        start_rect = pygame.Rect(cx - 120, cy + 80, 110, 45)
        pygame.draw.rect(self.surface, start_color, start_rect, border_radius=6)
        pygame.draw.rect(self.surface, (80, 90, 120), start_rect, 2, border_radius=6)
        st = self.font.render("Start", True, start_text_color)
        self.surface.blit(st, st.get_rect(center=start_rect.center))

        # Cancel button
        cancel_rect = pygame.Rect(cx + 10, cy + 80, 110, 45)
        pygame.draw.rect(self.surface, (65, 40, 40), cancel_rect, border_radius=6)
        pygame.draw.rect(self.surface, (80, 90, 120), cancel_rect, 2, border_radius=6)
        ct = self.font.render("Cancel", True, (220, 230, 240))
        self.surface.blit(ct, ct.get_rect(center=cancel_rect.center))

        # Host IP display
        if host_ip:
            ip_label = self.font_small.render("Your IP Address:", True, (120, 130, 150))
            self.surface.blit(ip_label, ip_label.get_rect(center=(cx, cy + 140)))
            ip_text = self.font.render(host_ip, True, (200, 220, 180))
            self.surface.blit(ip_text, ip_text.get_rect(center=(cx, cy + 165)))

        hint = self.font_small.render(
            "Share the room code and IP with other players", True, (100, 110, 130)
        )
        self.surface.blit(hint, hint.get_rect(center=(cx, cy + 195)))

        pygame.display.flip()
        return None

    def show_connecting(self, message: str = "Connecting...") -> None:
        """Show a simple 'connecting' screen."""
        self.surface.fill((20, 20, 35))
        cx = config.WINDOW_WIDTH // 2
        cy = config.WINDOW_HEIGHT // 2
        text = self.font_heading.render(message, True, (100, 180, 220))
        self.surface.blit(text, text.get_rect(center=(cx, cy)))
        pygame.display.flip()

    # ── Utility ──────────────────────────────────────────────

    def _draw_menu_bg(self, frame: int) -> None:
        """Draw an animated gradient background for menus."""
        self.surface.fill(COLOR_MENU_BG)

        # Subtle animated gradient overlay
        import math
        for y in range(0, config.WINDOW_HEIGHT, 4):
            t = y / config.WINDOW_HEIGHT
            wave = math.sin(frame * 0.02 + t * 3) * 0.5 + 0.5
            r = int(15 + wave * 8)
            g = int(15 + t * 10 + wave * 5)
            b = int(30 + t * 15 + wave * 10)
            pygame.draw.line(self.surface, (r, g, b), (0, y), (config.WINDOW_WIDTH, y))

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

    def _draw_tooltip(self, pos: tuple[int, int], text: str) -> None:
        """Draw a tooltip box near the mouse cursor."""
        pad_x, pad_y = 10, 6
        tip_font = self.font_small
        text_surf = tip_font.render(text, True, (230, 235, 245))
        tw, th = text_surf.get_size()

        box_w = tw + pad_x * 2
        box_h = th + pad_y * 2

        # Position: prefer below-right of cursor, clamp to screen
        tx = pos[0] + 14
        ty = pos[1] + 18
        if tx + box_w > config.WINDOW_WIDTH - 4:
            tx = pos[0] - box_w - 4
        if ty + box_h > config.WINDOW_HEIGHT - 4:
            ty = pos[1] - box_h - 4

        bg_rect = pygame.Rect(tx, ty, box_w, box_h)
        pygame.draw.rect(self.surface, (20, 22, 35), bg_rect, border_radius=5)
        pygame.draw.rect(self.surface, (90, 100, 140), bg_rect, 1, border_radius=5)
        self.surface.blit(text_surf, (tx + pad_x, ty + pad_y))

    # ── File Management ──────────────────────────────────────────

    def _pick_file_dialog(self, title: str = "Select File") -> str | None:
        """Open a native OS file dialog to pick a JSON file."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            filepath = filedialog.askopenfilename(
                title=title,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            root.destroy()
            pygame.event.clear()  # Clear stale events from losing focus
            return filepath if filepath else None
        except Exception:
            return None

    def _show_file_manager(
        self,
        directory: str,
        title: str = "SELECT FILE",
        hint: str = "Click: load  |  Right-click: delete  |  Scroll for more",
        allow_import: bool = True,
        name_transform=None,
    ) -> str | None:
        """Show a file manager dialog with load/delete/import support.

        Args:
            directory: Directory to list files from.
            title: Title text displayed at top.
            hint: Hint text below title.
            allow_import: Show Import button for OS file dialog.
            name_transform: Optional callable(Path) -> str for display names.

        Returns:
            Filepath string of selected file, or None if cancelled.
        """
        clock = pygame.time.Clock()
        scroll = 0

        if name_transform is None:
            name_transform = lambda f: f.stem

        while True:
            # Refresh file list each iteration (reflects deletions)
            dir_path = Path(directory)
            if dir_path.exists():
                files = sorted(
                    [f for f in dir_path.iterdir() if f.suffix == ".json"],
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
            else:
                files = []

            mouse_pos = pygame.mouse.get_pos()
            max_visible = max(1, (config.WINDOW_HEIGHT - 150) // 40)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None
                if event.type == pygame.MOUSEWHEEL:
                    max_scroll = max(0, len(files) - max_visible)
                    scroll = max(0, min(max_scroll, scroll - event.y))

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = mouse_pos

                    # Back button
                    back_rect = pygame.Rect(20, config.WINDOW_HEIGHT - 55, 100, 36)
                    if back_rect.collidepoint(mx, my) and event.button == 1:
                        return None

                    # Import button
                    if allow_import and event.button == 1:
                        import_rect = pygame.Rect(
                            config.WINDOW_WIDTH - 180, config.WINDOW_HEIGHT - 55, 160, 36,
                        )
                        if import_rect.collidepoint(mx, my):
                            filepath = self._pick_file_dialog(f"Import {title}")
                            if filepath:
                                return filepath
                            continue

                    # File items
                    for idx in range(min(max_visible, len(files) - scroll)):
                        fi = idx + scroll
                        item_rect = pygame.Rect(30, 80 + idx * 40, config.WINDOW_WIDTH - 60, 36)
                        if item_rect.collidepoint(mx, my):
                            # Rename button (right side of row)
                            rename_rect = pygame.Rect(
                                item_rect.right - 70, item_rect.y + 6, 60, 24,
                            )
                            if rename_rect.collidepoint(mx, my) and event.button == 1:
                                old_name = files[fi].stem
                                new_name = self._show_text_input("Rename:", old_name)
                                if new_name and new_name != old_name:
                                    safe = "".join(
                                        c if c.isalnum() or c in "-_ " else "_"
                                        for c in new_name
                                    )
                                    new_path = files[fi].parent / f"{safe}.json"
                                    files[fi].rename(new_path)
                                break
                            if event.button == 1:  # Left click - select
                                return str(files[fi])
                            elif event.button == 3:  # Right click - delete
                                name = files[fi].stem
                                if self._show_confirm(
                                    f"Delete '{name}'?", "This cannot be undone.",
                                ):
                                    files[fi].unlink()
                                break

            # ── Draw ──
            self.surface.fill(COLOR_MENU_BG)
            title_surf = self.font_heading.render(title, True, (180, 190, 220))
            self.surface.blit(title_surf, (30, 20))
            hint_surf = self.font_small.render(hint, True, (100, 105, 130))
            self.surface.blit(hint_surf, (30, 44))

            if files:
                for idx in range(min(max_visible, len(files) - scroll)):
                    fi = idx + scroll
                    f = files[fi]
                    item_rect = pygame.Rect(30, 80 + idx * 40, config.WINDOW_WIDTH - 60, 36)
                    hovered = item_rect.collidepoint(mouse_pos)
                    bg = (45, 50, 65) if hovered else (30, 33, 45)
                    pygame.draw.rect(self.surface, bg, item_rect, border_radius=4)
                    pygame.draw.rect(self.surface, (55, 60, 75), item_rect, 1, border_radius=4)

                    display = name_transform(f)
                    ft = self.font_small.render(display, True, (180, 185, 200))
                    self.surface.blit(ft, (item_rect.x + 10, item_rect.y + 4))

                    # Modification timestamp (shifted left to make room for rename)
                    mtime = f.stat().st_mtime
                    ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mtime))
                    ts_surf = self.font_small.render(ts, True, (90, 95, 115))
                    self.surface.blit(
                        ts_surf,
                        (item_rect.right - ts_surf.get_width() - 80, item_rect.y + 4),
                    )

                    # Rename button
                    rename_rect = pygame.Rect(
                        item_rect.right - 70, item_rect.y + 6, 60, 24,
                    )
                    rh = rename_rect.collidepoint(mouse_pos)
                    pygame.draw.rect(
                        self.surface, (50, 55, 75) if rh else (35, 38, 52),
                        rename_rect, border_radius=4,
                    )
                    rt = self.font_small.render("Rename", True, (160, 165, 185))
                    self.surface.blit(rt, rt.get_rect(center=rename_rect.center))

                # Scroll indicator
                if len(files) > max_visible:
                    total = len(files)
                    bar_area_h = max_visible * 40
                    bar_h = max(20, int(bar_area_h * max_visible / total))
                    max_scroll_val = max(1, total - max_visible)
                    bar_y = 80 + int((bar_area_h - bar_h) * scroll / max_scroll_val)
                    bar_x = config.WINDOW_WIDTH - 14
                    pygame.draw.rect(
                        self.surface, (40, 43, 58), (bar_x, 80, 5, bar_area_h),
                        border_radius=2,
                    )
                    pygame.draw.rect(
                        self.surface, (90, 100, 130), (bar_x, bar_y, 5, bar_h),
                        border_radius=2,
                    )
            else:
                no_files = self.font_subtitle.render("No files found", True, (90, 95, 115))
                self.surface.blit(
                    no_files, no_files.get_rect(center=(config.WINDOW_WIDTH // 2, 120)),
                )

            # Back button
            back_rect = pygame.Rect(20, config.WINDOW_HEIGHT - 55, 100, 36)
            bh = back_rect.collidepoint(mouse_pos)
            pygame.draw.rect(
                self.surface, (55, 60, 80) if bh else (40, 44, 58),
                back_rect, border_radius=5,
            )
            bt = self.font_small.render("Back", True, (180, 185, 200))
            self.surface.blit(bt, bt.get_rect(center=back_rect.center))

            # Import button
            if allow_import:
                import_rect = pygame.Rect(
                    config.WINDOW_WIDTH - 180, config.WINDOW_HEIGHT - 55, 160, 36,
                )
                ih = import_rect.collidepoint(mouse_pos)
                pygame.draw.rect(
                    self.surface, (50, 55, 70) if ih else (35, 40, 52),
                    import_rect, border_radius=5,
                )
                it = self.font_small.render("Import File...", True, (170, 180, 210))
                self.surface.blit(it, it.get_rect(center=import_rect.center))

            pygame.display.flip()
            clock.tick(30)

    # ── Per-Species Save / Load ─────────────────────────────────

    def _save_species_settings(self, sp: Species) -> None:
        """Save a species definition (diet flags + settings) to a named JSON file."""
        import json
        name = self._show_text_input("Save Species As:", sp.id)
        if not name:
            return
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        Path(self.SPECIES_DIR).mkdir(parents=True, exist_ok=True)
        filepath = Path(self.SPECIES_DIR) / f"{safe}.json"
        with open(filepath, "w") as f:
            json.dump(sp.to_dict(), f, indent=2)

    def _apply_species_file(self, sp: Species, filepath: str) -> bool:
        """Load a species JSON file and apply its values onto an existing species."""
        import json
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = Species.from_dict(data)
            for attr in (
                "can_eat_plants", "plant_food_multiplier",
                "can_attack_other_species", "can_attack_own_species",
                "can_eat_other_corpse", "can_eat_own_corpse",
                "attack_damage", "energy_steal_fraction",
                "scavenge_death_radius", "scavenge_death_energy",
            ):
                setattr(sp, attr, getattr(loaded, attr))
            sp.settings = loaded.settings.copy()
            return True
        except Exception:
            return False

    def _load_species_settings(self, sp: Species) -> None:
        """Show a file manager and load species settings from a JSON file."""
        filepath = self._show_file_manager(
            self.SPECIES_DIR,
            title=f"LOAD SETTINGS FOR {sp.name.upper()}",
            hint="Left-click: load  |  Right-click: delete  |  Import from file",
        )
        if filepath:
            if not self._apply_species_file(sp, filepath):
                self._show_message("Failed to load species file.")

    def _draw_toggle(self, x: int, y: int, on: bool) -> None:
        """Draw a toggle switch widget."""
        w, h = 50, 20
        bg = (60, 140, 80) if on else (60, 60, 70)
        pygame.draw.rect(self.surface, bg, (x, y, w, h), border_radius=10)
        knob_x = x + w - 14 if on else x + 6
        pygame.draw.circle(self.surface, (220, 225, 235), (knob_x, y + h // 2), 8)
        label = self.font_small.render("ON" if on else "OFF", True, (180, 190, 210))
        self.surface.blit(label, (x + w + 8, y + 2))

    def _build_confirm_buttons(self) -> tuple:
        """Build confirmation dialog buttons centered on the current window size."""
        cx = config.WINDOW_WIDTH // 2
        cy = config.WINDOW_HEIGHT // 2
        btn_w, btn_h = 140, 45
        yes_btn = Button(
            cx - btn_w - 15, cy + 40, btn_w, btn_h, "Yes, Restart",
            color=(120, 50, 50), hover_color=(160, 65, 65),
        )
        no_btn = Button(
            cx + 15, cy + 40, btn_w, btn_h, "Cancel",
            color=(50, 60, 80), hover_color=(70, 80, 110),
        )
        return yes_btn, no_btn

    def _show_confirm(self, title: str, subtitle: str = "") -> bool:
        """Show a confirmation dialog. Returns True if confirmed, False if cancelled."""
        yes_btn, no_btn = self._build_confirm_buttons()
        clock = pygame.time.Clock()

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if self._handle_window_event(event):
                    yes_btn, no_btn = self._build_confirm_buttons()
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_RETURN:
                        return True
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if yes_btn.is_clicked(mouse_pos):
                        return True
                    if no_btn.is_clicked(mouse_pos):
                        return False

            # Dark overlay
            overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.surface.blit(overlay, (0, 0))

            # Dialog box
            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2
            dialog_w, dialog_h = 400, 160
            dialog_rect = pygame.Rect(cx - dialog_w // 2, cy - dialog_h // 2, dialog_w, dialog_h)
            pygame.draw.rect(self.surface, (25, 28, 40), dialog_rect, border_radius=8)
            pygame.draw.rect(self.surface, (80, 90, 120), dialog_rect, 2, border_radius=8)

            # Title
            title_surf = self.font_heading.render(title, True, (255, 200, 100))
            self.surface.blit(title_surf, title_surf.get_rect(center=(cx, cy - 30)))

            # Subtitle
            if subtitle:
                sub_surf = self.font_small.render(subtitle, True, (160, 160, 180))
                self.surface.blit(sub_surf, sub_surf.get_rect(center=(cx, cy + 5)))

            # Buttons
            yes_btn.update(mouse_pos)
            yes_btn.draw(self.surface, self.font)
            no_btn.update(mouse_pos)
            no_btn.draw(self.surface, self.font)

            pygame.display.flip()
            clock.tick(30)

    def _show_message(self, *lines: str) -> None:
        """Show a simple message screen until ESC is pressed."""
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            self.surface.fill(COLOR_MENU_BG)
            cx = config.WINDOW_WIDTH // 2
            y = config.WINDOW_HEIGHT // 2 - len(lines) * 15
            for line in lines:
                text = self.font.render(line, True, COLOR_HUD_TEXT)
                self.surface.blit(text, text.get_rect(center=(cx, y)))
                y += 30

            pygame.display.flip()
            clock.tick(30)
