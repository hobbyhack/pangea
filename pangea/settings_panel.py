"""
SettingsPanel — in-game left-side overlay for live settings tuning.
===================================================================
Toggled with the S key during simulation. Renders over the game world
on the left side. Supports scrolling, slider dragging, toggles, and
save/load of settings presets to JSON files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pygame

import pangea.config as config
from pangea.settings import SETTING_DEFS, SimSettings


# ── Constants ────────────────────────────────────────────────
PANEL_WIDTH = 360
PANEL_BG = (18, 20, 30, 220)
PANEL_BORDER = (60, 70, 100)
HEADER_HEIGHT = 50
FOOTER_HEIGHT = 90
ROW_HEIGHT = 34
SLIDER_W = 180
SLIDER_H = 8
SETTINGS_DIR = "settings"


class SettingsPanel:
    """
    In-game settings overlay drawn on the left side of the screen.

    The panel consumes mouse/keyboard events when visible so they
    don't leak to the simulation underneath.
    """

    def __init__(self) -> None:
        self.visible = False
        self.scroll_y = 0
        self._dragging: dict | None = None  # which slider is being dragged
        self._sliders: list[dict] = []
        self._rebuild_layout()

        # Fonts (initialized lazily after pygame.init)
        self._font: pygame.font.Font | None = None
        self._font_small: pygame.font.Font | None = None
        self._font_heading: pygame.font.Font | None = None

        # File picker state
        self._file_picker_mode: str | None = None  # "save" or "load" or None
        self._file_list: list[str] = []
        self._file_scroll: int = 0
        self._save_flash_timer: float = 0.0

        # Tooltip
        self._hovered_tooltip: str = ""

    # ── Fonts ─────────────────────────────────────────────────

    def _ensure_fonts(self) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", 18)
            self._font_small = pygame.font.SysFont("consolas", 13)
            self._font_heading = pygame.font.SysFont("consolas", 15, bold=True)

    # ── Layout ────────────────────────────────────────────────

    def _rebuild_layout(self) -> None:
        """Build slider metadata with relative y positions."""
        self._sliders = []
        y = 0
        last_cat = ""
        for sdef in SETTING_DEFS:
            if sdef.category != last_cat:
                last_cat = sdef.category
                y += 8  # category gap
            self._sliders.append({
                "def": sdef,
                "y": y,
                "dragging": False,
            })
            y += ROW_HEIGHT
        self._content_height = y

    # ── Public API ────────────────────────────────────────────

    def toggle(self) -> None:
        self.visible = not self.visible
        if self.visible:
            self._file_picker_mode = None

    def panel_rect(self) -> pygame.Rect:
        """Return the screen rect occupied by the panel."""
        return pygame.Rect(0, 0, PANEL_WIDTH, config.WINDOW_HEIGHT)

    def handle_event(self, event: pygame.event.Event, settings: SimSettings) -> SimSettings:
        """
        Process a pygame event if the panel is visible.

        Returns the (possibly modified) settings. The caller should check
        ``panel.visible`` and skip its own handling when the event lands
        inside the panel rect.
        """
        if not self.visible:
            return settings

        prect = self.panel_rect()

        # File picker mode has its own event handling
        if self._file_picker_mode is not None:
            return self._handle_file_picker_event(event, settings)

        # Scroll wheel
        if event.type == pygame.MOUSEWHEEL:
            if prect.collidepoint(pygame.mouse.get_pos()):
                scroll_area = config.WINDOW_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
                max_scroll = max(0, self._content_height - scroll_area)
                self.scroll_y = max(0, min(max_scroll, self.scroll_y - event.y * 28))
                return settings

        # Mouse button down
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not prect.collidepoint(mx, my):
                return settings

            px = prect.x

            # Check footer buttons
            btn_y = config.WINDOW_HEIGHT - FOOTER_HEIGHT + 10
            btn_w = (PANEL_WIDTH - 30) // 3
            for i, label in enumerate(["Save", "Load", "Reset"]):
                bx = px + 10 + i * (btn_w + 5)
                btn_rect = pygame.Rect(bx, btn_y, btn_w, 32)
                if btn_rect.collidepoint(mx, my):
                    if label == "Save":
                        self._do_save(settings)
                    elif label == "Load":
                        self._open_file_picker("load")
                    elif label == "Reset":
                        settings = SimSettings()
                    return settings

            # Check close button
            close_rect = pygame.Rect(px + PANEL_WIDTH - 30, 8, 22, 22)
            if close_rect.collidepoint(mx, my):
                self.visible = False
                return settings

            # Check sliders / toggles
            for s in self._sliders:
                sdef = s["def"]
                draw_y = s["y"] - self.scroll_y + HEADER_HEIGHT
                if draw_y < HEADER_HEIGHT - 10 or draw_y > config.WINDOW_HEIGHT - FOOTER_HEIGHT:
                    continue
                if sdef.widget_type == "toggle":
                    toggle_rect = pygame.Rect(px + PANEL_WIDTH - 80, draw_y, 50, 24)
                    if toggle_rect.collidepoint(mx, my):
                        cur = getattr(settings, sdef.key)
                        setattr(settings, sdef.key, not cur)
                        return settings
                else:
                    slider_rect = pygame.Rect(
                        px + PANEL_WIDTH - SLIDER_W - 30, draw_y,
                        SLIDER_W, SLIDER_H + 16,
                    )
                    if slider_rect.collidepoint(mx, my):
                        s["dragging"] = True
                        self._dragging = s
                        return settings

        # Mouse button up — stop dragging
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging:
                self._dragging["dragging"] = False
                self._dragging = None

        return settings

    def update_dragging(self, settings: SimSettings) -> SimSettings:
        """Update the value of the currently dragged slider."""
        if not self.visible or self._dragging is None:
            return settings
        s = self._dragging
        sdef = s["def"]
        if sdef.widget_type == "toggle":
            return settings

        mx = pygame.mouse.get_pos()[0]
        px = self.panel_rect().x
        sx = px + PANEL_WIDTH - SLIDER_W - 30
        t = max(0.0, min(1.0, (mx - sx) / SLIDER_W))
        raw = sdef.min_val + t * (sdef.max_val - sdef.min_val)
        snapped = round(raw / sdef.step) * sdef.step
        snapped = max(sdef.min_val, min(sdef.max_val, snapped))
        if sdef.step >= 1:
            snapped = int(snapped)
        setattr(settings, sdef.key, snapped)
        return settings

    def consumes_click(self, mx: int, my: int) -> bool:
        """Return True if a click at (mx, my) is inside the panel."""
        return self.visible and self.panel_rect().collidepoint(mx, my)

    # ── Drawing ───────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, settings: SimSettings, dt: float = 0.0) -> None:
        """Render the panel overlay on the right side of the screen."""
        if not self.visible:
            return

        self._ensure_fonts()

        if self._save_flash_timer > 0:
            self._save_flash_timer = max(0, self._save_flash_timer - dt)

        prect = self.panel_rect()
        px = prect.x

        # Semi-transparent background
        bg = pygame.Surface((PANEL_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
        bg.fill(PANEL_BG)
        surface.blit(bg, (px, 0))
        pygame.draw.line(surface, PANEL_BORDER, (px + PANEL_WIDTH, 0), (px + PANEL_WIDTH, config.WINDOW_HEIGHT), 2)

        # Header
        title = self._font.render("SETTINGS", True, (160, 175, 210))
        surface.blit(title, (px + 12, 14))
        hint = self._font_small.render("[S] close  |  changes apply next gen", True, (90, 95, 120))
        surface.blit(hint, (px + 12, 36))

        # Close button
        close_rect = pygame.Rect(px + PANEL_WIDTH - 30, 8, 22, 22)
        pygame.draw.rect(surface, (80, 40, 40), close_rect, border_radius=4)
        x_text = self._font_small.render("X", True, (200, 160, 160))
        surface.blit(x_text, x_text.get_rect(center=close_rect.center))

        # File picker overlay
        if self._file_picker_mode is not None:
            self._draw_file_picker(surface, px)
            return

        # Clip to scroll area
        scroll_area_h = config.WINDOW_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
        clip = pygame.Rect(px, HEADER_HEIGHT, PANEL_WIDTH, scroll_area_h)
        surface.set_clip(clip)

        mouse_pos = pygame.mouse.get_pos()
        self._hovered_tooltip = ""
        last_cat = ""

        for s in self._sliders:
            sdef = s["def"]
            draw_y = s["y"] - self.scroll_y + HEADER_HEIGHT

            if draw_y < HEADER_HEIGHT - ROW_HEIGHT or draw_y > config.WINDOW_HEIGHT - FOOTER_HEIGHT + ROW_HEIGHT:
                continue

            # Category header
            if sdef.category != last_cat:
                last_cat = sdef.category
                cat = self._font_heading.render(sdef.category, True, (90, 125, 190))
                surface.blit(cat, (px + 10, draw_y - 2))

            # Label
            label = self._font_small.render(sdef.label, True, (170, 170, 190))
            label_rect = label.get_rect(topleft=(px + 10, draw_y + 14))
            surface.blit(label, label_rect)

            if sdef.tooltip and label_rect.collidepoint(mouse_pos):
                self._hovered_tooltip = sdef.tooltip

            val = getattr(settings, sdef.key)
            sx = px + PANEL_WIDTH - SLIDER_W - 30

            if sdef.widget_type == "toggle":
                self._draw_toggle(surface, px + PANEL_WIDTH - 80, draw_y + 14, bool(val))
            else:
                sy = draw_y + 18
                t = (val - sdef.min_val) / (sdef.max_val - sdef.min_val) if sdef.max_val > sdef.min_val else 0

                # Track
                pygame.draw.rect(surface, (35, 35, 50), (sx, sy, SLIDER_W, SLIDER_H), border_radius=4)
                fill_w = int(SLIDER_W * t)
                if fill_w > 0:
                    fill_color = self._lerp_color((50, 110, 180), (80, 200, 140), t)
                    pygame.draw.rect(surface, fill_color, (sx, sy, fill_w, SLIDER_H), border_radius=4)

                # Thumb
                thumb_x = sx + int(SLIDER_W * t)
                thumb_c = (200, 210, 230) if s["dragging"] else (140, 150, 170)
                pygame.draw.circle(surface, thumb_c, (thumb_x, sy + SLIDER_H // 2), 6)

                # Value
                val_str = f"{val:{sdef.fmt}}"
                vt = self._font_small.render(val_str, True, (150, 160, 180))
                surface.blit(vt, (sx + SLIDER_W + 8, draw_y + 14))

        surface.set_clip(None)

        # Scroll bar
        max_scroll = max(1, self._content_height - scroll_area_h)
        if self._content_height > scroll_area_h:
            bar_h = max(16, int(scroll_area_h * scroll_area_h / self._content_height))
            bar_y = HEADER_HEIGHT + int((scroll_area_h - bar_h) * self.scroll_y / max_scroll)
            bar_x = px + PANEL_WIDTH - 10
            pygame.draw.rect(surface, (40, 42, 55), (bar_x, HEADER_HEIGHT, 4, scroll_area_h), border_radius=2)
            pygame.draw.rect(surface, (90, 100, 130), (bar_x, bar_y, 4, bar_h), border_radius=2)

        # Footer
        footer_y = config.WINDOW_HEIGHT - FOOTER_HEIGHT
        footer_bg = pygame.Surface((PANEL_WIDTH, FOOTER_HEIGHT), pygame.SRCALPHA)
        footer_bg.fill((18, 20, 30, 240))
        surface.blit(footer_bg, (px, footer_y))
        pygame.draw.line(surface, (50, 55, 70), (px, footer_y), (px + PANEL_WIDTH, footer_y))

        btn_y = footer_y + 10
        btn_w = (PANEL_WIDTH - 30) // 3
        btn_labels = ["Save", "Load", "Reset"]
        btn_colors = [(40, 65, 50), (45, 50, 70), (70, 40, 40)]
        btn_hover = [(55, 90, 65), (60, 68, 95), (95, 55, 55)]

        for i, (lbl, col, hcol) in enumerate(zip(btn_labels, btn_colors, btn_hover)):
            bx = px + 10 + i * (btn_w + 5)
            brect = pygame.Rect(bx, btn_y, btn_w, 32)
            hovered = brect.collidepoint(mouse_pos)
            pygame.draw.rect(surface, hcol if hovered else col, brect, border_radius=5)
            pygame.draw.rect(surface, (70, 75, 90), brect, 1, border_radius=5)
            bt = self._font_small.render(lbl, True, (190, 195, 210))
            surface.blit(bt, bt.get_rect(center=brect.center))

        # Save flash feedback
        if self._save_flash_timer > 0:
            alpha = int(255 * min(1.0, self._save_flash_timer / 0.5))
            flash = self._font_small.render("Saved!", True, (100, 220, 130))
            flash.set_alpha(alpha)
            surface.blit(flash, (px + 10, footer_y + 48))

        # Tooltip
        if self._hovered_tooltip:
            self._draw_tooltip(surface, mouse_pos, self._hovered_tooltip)

    # ── Save / Load ───────────────────────────────────────────

    def _do_save(self, settings: SimSettings) -> None:
        """Save current settings to a timestamped JSON file."""
        Path(SETTINGS_DIR).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{SETTINGS_DIR}/settings_{timestamp}.json"
        settings.save_to_file(filepath)
        self._save_flash_timer = 1.5

    def _open_file_picker(self, mode: str) -> None:
        """Open the file picker sub-panel."""
        self._file_picker_mode = mode
        self._file_list = SimSettings.list_settings_files(SETTINGS_DIR)
        self._file_scroll = 0

    def _handle_file_picker_event(
        self, event: pygame.event.Event, settings: SimSettings,
    ) -> SimSettings:
        """Handle events when the file picker is open."""
        prect = self.panel_rect()
        px = prect.x

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._file_picker_mode = None
            return settings

        if event.type == pygame.MOUSEWHEEL:
            if prect.collidepoint(pygame.mouse.get_pos()):
                self._file_scroll = max(0, self._file_scroll - event.y)
                return settings

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not prect.collidepoint(mx, my):
                self._file_picker_mode = None
                return settings

            # Close button
            close_rect = pygame.Rect(px + PANEL_WIDTH - 30, 8, 22, 22)
            if close_rect.collidepoint(mx, my):
                self.visible = False
                self._file_picker_mode = None
                return settings

            # Back button
            back_rect = pygame.Rect(px + 10, config.WINDOW_HEIGHT - 50, 80, 32)
            if back_rect.collidepoint(mx, my):
                self._file_picker_mode = None
                return settings

            # File items
            max_visible = (config.WINDOW_HEIGHT - HEADER_HEIGHT - 80) // 30
            for i in range(min(max_visible, len(self._file_list) - self._file_scroll)):
                idx = i + self._file_scroll
                item_rect = pygame.Rect(px + 10, HEADER_HEIGHT + 30 + i * 30, PANEL_WIDTH - 20, 26)
                if item_rect.collidepoint(mx, my):
                    filepath = f"{SETTINGS_DIR}/{self._file_list[idx]}"
                    try:
                        settings = SimSettings.load_from_file(filepath)
                        self._file_picker_mode = None
                    except Exception:
                        pass  # ignore bad files
                    return settings

        return settings

    def _draw_file_picker(self, surface: pygame.Surface, px: int) -> None:
        """Draw the file picker sub-panel."""
        self._ensure_fonts()
        mouse_pos = pygame.mouse.get_pos()

        mode_label = "LOAD SETTINGS" if self._file_picker_mode == "load" else "SAVE SETTINGS"
        title = self._font.render(mode_label, True, (160, 175, 210))
        surface.blit(title, (px + 12, HEADER_HEIGHT + 4))

        if not self._file_list:
            msg = self._font_small.render("No settings files found.", True, (140, 140, 160))
            surface.blit(msg, (px + 20, HEADER_HEIGHT + 50))
        else:
            max_visible = (config.WINDOW_HEIGHT - HEADER_HEIGHT - 80) // 30
            for i in range(min(max_visible, len(self._file_list) - self._file_scroll)):
                idx = i + self._file_scroll
                item_rect = pygame.Rect(px + 10, HEADER_HEIGHT + 30 + i * 30, PANEL_WIDTH - 20, 26)
                hovered = item_rect.collidepoint(mouse_pos)
                color = (45, 50, 65) if hovered else (30, 33, 45)
                pygame.draw.rect(surface, color, item_rect, border_radius=4)
                pygame.draw.rect(surface, (55, 60, 75), item_rect, 1, border_radius=4)
                name = self._file_list[idx]
                # Trim extension for display
                display = name.replace(".json", "").replace("settings_", "")
                ft = self._font_small.render(display, True, (180, 185, 200))
                surface.blit(ft, (item_rect.x + 8, item_rect.y + 5))

        # Back button
        back_rect = pygame.Rect(px + 10, config.WINDOW_HEIGHT - 50, 80, 32)
        hovered = back_rect.collidepoint(mouse_pos)
        pygame.draw.rect(surface, (60, 60, 75) if hovered else (45, 48, 60), back_rect, border_radius=5)
        bt = self._font_small.render("Back", True, (180, 185, 200))
        surface.blit(bt, bt.get_rect(center=back_rect.center))

    # ── Widget Helpers ────────────────────────────────────────

    def _draw_toggle(self, surface: pygame.Surface, x: int, y: int, on: bool) -> None:
        w, h = 44, 18
        bg = (50, 120, 70) if on else (50, 50, 60)
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=9)
        knob_x = x + w - 12 if on else x + 6
        pygame.draw.circle(surface, (210, 215, 225), (knob_x, y + h // 2), 7)
        label = self._font_small.render("ON" if on else "OFF", True, (160, 170, 190))
        surface.blit(label, (x + w + 6, y + 1))

    def _draw_tooltip(
        self, surface: pygame.Surface, pos: tuple[int, int], text: str,
    ) -> None:
        pad_x, pad_y = 8, 5
        text_surf = self._font_small.render(text, True, (225, 230, 240))
        tw, th = text_surf.get_size()
        box_w, box_h = tw + pad_x * 2, th + pad_y * 2

        tx = pos[0] + 14  # show to the right of cursor (panel is on left)
        ty = pos[1] + 14
        if tx + box_w > config.WINDOW_WIDTH - 4:
            tx = pos[0] - box_w - 10
        if ty + box_h > config.WINDOW_HEIGHT - 4:
            ty = pos[1] - box_h - 4

        bg_rect = pygame.Rect(tx, ty, box_w, box_h)
        pygame.draw.rect(surface, (15, 18, 28), bg_rect, border_radius=5)
        pygame.draw.rect(surface, (80, 90, 130), bg_rect, 1, border_radius=5)
        surface.blit(text_surf, (tx + pad_x, ty + pad_y))

    @staticmethod
    def _lerp_color(
        c1: tuple[int, int, int], c2: tuple[int, int, int], t: float,
    ) -> tuple[int, int, int]:
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )
