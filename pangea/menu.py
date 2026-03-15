"""
Menu — pygame menu screens for mode selection, settings, and file picking.
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
)
import pangea.config as config
from pangea.save_load import delete_save, list_saves
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
        start_y = cy - 60
        return {
            "isolation": Button(cx - btn_w // 2, start_y, btn_w, btn_h, "Isolation Mode",
                                color=(40, 70, 50), hover_color=(55, 100, 65)),
            "convergence": Button(cx - btn_w // 2, start_y + 70, btn_w, btn_h, "Convergence Mode",
                                  color=(50, 45, 75), hover_color=(70, 60, 110)),
            "freeplay": Button(cx - btn_w // 2, start_y + 140, btn_w, btn_h, "Freeplay Mode",
                               color=(60, 55, 40), hover_color=(90, 80, 55)),
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

    # ── Mode Select (New / Load Save) ───────────────────────

    def _mode_display_name(self, mode: str) -> str:
        """Return a human-readable title for a game mode."""
        return {"isolation": "Isolation Mode", "convergence": "Convergence Mode", "freeplay": "Freeplay Mode"}.get(mode, mode.title())

    def show_mode_select(self, mode: str) -> str | dict | None:
        """
        Show New Game / Load Save screen for a game mode.

        Returns:
            "new"  — start a new game
            dict   — a loaded save (from load_game)
            None   — user pressed back / ESC
        """
        clock = pygame.time.Clock()
        frame = 0

        while True:
            # Refresh save list each loop iteration (in case of deletion)
            saves = list_saves(mode)

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
                title_text = self._mode_display_name(mode)
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
        return {
            "panel_x": panel_x, "panel_w": panel_w, "slider_w": slider_w,
            "header_h": header_h, "footer_h": footer_h,
            "scroll_area_h": scroll_area_h, "btn_y": btn_y,
            "back_btn": Button(w // 2 - 80, btn_y, 160, 45, "Back",
                               color=(50, 60, 80), hover_color=(70, 80, 110)),
            "reset_btn": Button(w // 2 + 100, btn_y, 140, 45, "Reset",
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
            lay["back_btn"].update(mouse_pos)
            lay["back_btn"].draw(self.surface, self.font)
            lay["reset_btn"].update(mouse_pos)
            lay["reset_btn"].draw(self.surface, self.font)

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

    def show_file_select(self, species_dir: str = "species") -> tuple[str, str] | None:
        """
        Show file selection screen for convergence mode.
        Left-click to select, right-click to delete, F2 or double-click name to rename.

        Returns:
            Tuple of (file_a_path, file_b_path) or None if cancelled.
        """
        import os

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

        names = self._load_species_names(species_path, files)
        selected: list[int] = []
        scroll_offset = 0
        max_visible = 12
        hovered_idx = -1

        clock = pygame.time.Clock()

        while True:
            mouse_pos = pygame.mouse.get_pos()
            cx = config.WINDOW_WIDTH // 2
            cy = config.WINDOW_HEIGHT // 2
            list_top = cy - 170
            start_y = cy + 260

            # Track which item is hovered
            hovered_idx = -1
            for i in range(min(max_visible, len(files) - scroll_offset)):
                idx = i + scroll_offset
                item_rect = pygame.Rect(cx - 200, list_top + i * 35, 400, 30)
                if item_rect.collidepoint(mouse_pos):
                    hovered_idx = idx

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if self._handle_window_event(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    # F2 to rename hovered item
                    if event.key == pygame.K_F2 and hovered_idx >= 0:
                        fname = files[hovered_idx]
                        old_name = names[fname].split("  (")[0]
                        new_name = self._show_text_input("Rename Species", old_name)
                        if new_name:
                            self._rename_species(species_path / fname, new_name)
                            names = self._load_species_names(species_path, files)

                # Right-click to delete
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    for i in range(min(max_visible, len(files) - scroll_offset)):
                        idx = i + scroll_offset
                        item_rect = pygame.Rect(cx - 200, list_top + i * 35, 400, 30)
                        if item_rect.collidepoint(mouse_pos):
                            fname = files[idx]
                            display = names[fname].split("  (")[0]
                            if self._show_confirm(
                                f"Delete '{display}'?",
                                "This cannot be undone.",
                            ):
                                os.remove(species_path / fname)
                                # Remove from selection
                                selected = [s for s in selected if s != idx]
                                selected = [s - 1 if s > idx else s for s in selected]
                                # Refresh file list
                                files = sorted(str(f.name) for f in species_path.glob("*.json"))
                                names = self._load_species_names(species_path, files)
                                scroll_offset = max(0, min(scroll_offset, len(files) - max_visible))
                                if len(files) < 2:
                                    return self._show_message(
                                        "Need at least 2 species files.",
                                        "Save species in Isolation Mode first!",
                                        "Press ESC to go back.",
                                    )
                            break

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i in range(min(max_visible, len(files) - scroll_offset)):
                        idx = i + scroll_offset
                        item_rect = pygame.Rect(cx - 200, list_top + i * 35, 400, 30)
                        if item_rect.collidepoint(mouse_pos):
                            if idx in selected:
                                selected.remove(idx)
                            elif len(selected) < 2:
                                selected.append(idx)

                    if len(selected) == 2:
                        start_rect = pygame.Rect(cx - 100, start_y, 200, 45)
                        if start_rect.collidepoint(mouse_pos):
                            file_a = str(species_path / files[selected[0]])
                            file_b = str(species_path / files[selected[1]])
                            return (file_a, file_b)

                if event.type == pygame.MOUSEWHEEL:
                    scroll_offset = max(0, min(len(files) - max_visible, scroll_offset - event.y))

            self.surface.fill(COLOR_MENU_BG)

            title = self.font.render("Select 2 Species Files", True, COLOR_HUD_TEXT)
            self.surface.blit(title, title.get_rect(center=(cx, list_top - 70)))

            hint = self.font_subtitle.render(
                "Left-click: select  |  Right-click: delete  |  F2: rename",
                True, (120, 120, 150),
            )
            self.surface.blit(hint, hint.get_rect(center=(cx, list_top - 35)))

            for i in range(min(max_visible, len(files) - scroll_offset)):
                idx = i + scroll_offset
                item_rect = pygame.Rect(cx - 200, list_top + i * 35, 400, 30)

                if idx in selected:
                    sel_idx = selected.index(idx)
                    color = (100, 40, 40) if sel_idx == 0 else (40, 50, 100)
                    label = " [A]" if sel_idx == 0 else " [B]"
                else:
                    color = (35, 38, 52)
                    label = ""

                hovered = item_rect.collidepoint(mouse_pos)
                if hovered:
                    color = tuple(min(255, c + 20) for c in color)

                pygame.draw.rect(self.surface, color, item_rect, border_radius=4)
                pygame.draw.rect(self.surface, (60, 65, 80), item_rect, 1, border_radius=4)

                display_name = names.get(files[idx], files[idx])
                text = self.font_subtitle.render(display_name + label, True, COLOR_HUD_TEXT)
                self.surface.blit(text, (item_rect.x + 10, item_rect.y + 7))

            if len(selected) == 2:
                start_rect = pygame.Rect(cx - 100, start_y, 200, 45)
                start_hovered = start_rect.collidepoint(mouse_pos)
                start_color = (60, 90, 140) if start_hovered else (45, 65, 105)
                pygame.draw.rect(self.surface, start_color, start_rect, border_radius=6)
                start_text = self.font.render("Start Battle!", True, COLOR_BUTTON_TEXT)
                self.surface.blit(start_text, start_text.get_rect(center=start_rect.center))

            pygame.display.flip()
            clock.tick(30)

    # ── Pause Menu ───────────────────────────────────────────

    def _build_pause_buttons(self, mode: str) -> tuple[list[str], dict[str, Button]]:
        """Build pause menu buttons centered on the current window size."""
        cx = config.WINDOW_WIDTH // 2
        cy = config.WINDOW_HEIGHT // 2
        btn_w, btn_h = 220, 45

        options = ["resume", "restart", "main_menu"]
        labels = ["Resume", "Restart", "Main Menu"]
        colors = [
            ((40, 70, 50), (55, 100, 65)),
            ((55, 55, 55), (75, 75, 75)),
            ((65, 40, 40), (90, 50, 50)),
        ]
        if mode in ("isolation", "freeplay", "convergence"):
            options.insert(1, "save_quit")
            labels.insert(1, "Save & Quit")
            colors.insert(1, ((50, 50, 70), (65, 65, 95)))
        if mode in ("isolation", "freeplay"):
            idx = options.index("save_quit") + 1
            options.insert(idx, "settings")
            labels.insert(idx, "Settings")
            colors.insert(idx, ((55, 55, 65), (75, 75, 90)))

        buttons = {}
        for i, (name, label) in enumerate(zip(options, labels)):
            c, hc = colors[i]
            buttons[name] = Button(
                cx - btn_w // 2, cy - 80 + i * 55, btn_w, btn_h, label,
                color=c, hover_color=hc,
            )
        return options, buttons

    def show_pause_menu(self, mode: str = "isolation", settings: SimSettings | None = None) -> tuple[str, SimSettings | None]:
        """
        Show the pause overlay with options.

        Returns:
            Tuple of (action_string, updated_settings_or_None).
        """
        _, buttons = self._build_pause_buttons(mode)
        clock = pygame.time.Clock()

        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ("main_menu", settings)
                if self._handle_window_event(event):
                    _, buttons = self._build_pause_buttons(mode)
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        return ("resume", settings)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, btn in buttons.items():
                        if btn.is_clicked(mouse_pos):
                            if name == "settings" and settings is not None:
                                settings = self.show_settings(settings)
                                _, buttons = self._build_pause_buttons(mode)
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

    # ── Results Screen ───────────────────────────────────────

    def show_convergence_results(
        self,
        winner: str,
        a_food: int,
        b_food: int,
        a_gens: int,
        b_gens: int,
    ) -> None:
        """Show convergence mode results."""
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
            self.surface.blit(title, title.get_rect(center=(cx, cy - 100)))

            lines = [
                f"Red (A):  {a_food} food eaten, survived {a_gens} generations",
                f"Blue (B): {b_food} food eaten, survived {b_gens} generations",
                "",
                "Press any key to return to menu",
            ]
            y = cy + 50
            for line in lines:
                text = self.font.render(line, True, COLOR_HUD_TEXT)
                self.surface.blit(text, text.get_rect(center=(cx, y)))
                y += 35

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
