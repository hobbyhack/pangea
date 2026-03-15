"""
Simulation — the main game loop tying everything together.
============================================================
Initializes pygame, runs the menu, and manages the freeplay simulation.

Controls:
    SPACE  → Pause / unpause
    F      → Toggle fast-forward (multiple sim steps per frame)
    +/-    → Adjust fast-forward speed multiplier (2×–20×)
    D      → Toggle debug overlay (vision ranges, energy bars)
    E      → Toggle evolution panel (minimap, trait graphs)
    S      → Toggle settings panel (right-side overlay with save/load)
    F10    → Toggle maximized window (keeps title bar / taskbar)
    F11    → Toggle fullscreen
    1-6    → Select player tool
    ESC    → Pause menu (or close settings panel if open)
    Left-click → Use active tool
"""

from __future__ import annotations

import random
from datetime import datetime

import pygame

import pangea.config as config
from pangea.config import (
    FPS,
    NET_SNAPSHOT_INTERVAL,
)
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.evolution import (
    create_next_generation,
    select_top,
)
from pangea.menu import Menu
from pangea.renderer import Renderer
from pangea.save_load import (
    save_snapshot, save_species,
)
from pangea.settings import SimSettings
from pangea.settings_panel import SettingsPanel
from pangea.network import NetworkClient, NetworkHost
from pangea.server import EmbeddedRelay
from pangea.protocol import (
    MsgType,
    apply_full_state,
    apply_snapshot,
    full_state_from_world,
    snapshot_from_world,
    tool_action_msg,
)
from pangea.tools import TOOL_LIST, PlayerTools
from pangea.world import World


class Simulation:
    """Main simulation controller."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Pangea - Evolution Simulator")
        self.fullscreen = False
        self.maximized = False
        self._windowed_size = (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.menu = Menu(self.screen, on_toggle_fullscreen=self._toggle_fullscreen, on_toggle_maximized=self._toggle_maximized, on_resize=self._handle_resize)

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.fast_forward_multiplier = 5  # sim steps per frame in fast mode
        self.debug_mode = False
        self.show_evolution_panel = False
        self.generation_history: list[dict] = []
        self.settings = SimSettings()
        self.settings.world_width = config.WINDOW_WIDTH
        self.settings.world_height = config.WINDOW_HEIGHT
        self.tools = PlayerTools()
        self.settings_panel = SettingsPanel()
        self._active_world: World | None = None
        self._net_host: NetworkHost | None = None
        self._net_client: NetworkClient | None = None
        self._embedded_relay: EmbeddedRelay | None = None

    def _rebuild_display(self) -> None:
        """Update config, renderer, and menu after a screen size change."""
        config.WINDOW_WIDTH = self.screen.get_width()
        config.WINDOW_HEIGHT = self.screen.get_height()
        # Keep world size in sync with window (1:1, no stretching)
        self.settings.world_width = config.WINDOW_WIDTH
        self.settings.world_height = config.WINDOW_HEIGHT
        if self._active_world is not None:
            self._active_world.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.renderer = Renderer(self.screen)
        self.menu.surface = self.screen

    def _toggle_fullscreen(self) -> None:
        """Toggle between windowed and fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN,
            )
        else:
            self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        self.maximized = False
        self._rebuild_display()

    def _toggle_maximized(self) -> None:
        """Toggle between windowed and maximized (keeps title bar / taskbar)."""
        if self.fullscreen:
            self._toggle_fullscreen()
            return
        import os
        self.maximized = not self.maximized
        if self.maximized:
            info = pygame.display.Info()
            w, h = info.current_w, info.current_h - 72
            os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        else:
            os.environ.pop("SDL_VIDEO_WINDOW_POS", None)
            self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        self._rebuild_display()

    def _handle_resize(self, width: int, height: int) -> None:
        """Handle window resize events."""
        if self.fullscreen:
            return
        self._windowed_size = (width, height)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self._rebuild_display()

    def _screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        """Convert screen coordinates to world coordinates (1:1 mapping)."""
        return float(sx), float(sy)

    # ── Main Entry Point ─────────────────────────────────────

    def run(self) -> None:
        """Main application loop — show menu, run selected mode, repeat."""
        while self.running:
            choice, self.settings = self.menu.show_main_menu(self.settings)

            if choice == "quit":
                self.running = False
            elif choice == "freeplay":
                mode_result = self.menu.show_mode_select()
                if mode_result is None:
                    continue  # user pressed Back
                elif mode_result == "new":
                    self._run_freeplay()
                elif isinstance(mode_result, dict):
                    # Load a save
                    save_data = mode_result
                    loaded_settings = SimSettings.from_dict(save_data["settings"])
                    # Keep window size from current session
                    loaded_settings.world_width = self.settings.world_width
                    loaded_settings.world_height = self.settings.world_height
                    self.settings = loaded_settings

                    if save_data.get("snapshot"):
                        self._run_freeplay(loaded_snapshot=save_data)
                    else:
                        self._run_freeplay(
                            loaded_dna=save_data["creatures"],
                        )
                self._active_world = None

            elif choice == "host":
                result = self.menu.show_host_setup(self.settings)
                if result is None:
                    continue
                _, relay_url, self.settings = result
                self.menu.show_connecting("Starting server...")

                # Extract port from relay URL for the embedded server
                import re
                port_match = re.search(r":(\d+)$", relay_url)
                relay_port = int(port_match.group(1)) if port_match else 8765

                # Start embedded relay server automatically
                try:
                    self._embedded_relay = EmbeddedRelay("0.0.0.0", relay_port)
                    self._embedded_relay.start()
                except Exception as exc:
                    self.menu.show_error(f"Failed to start relay: {exc}")
                    self._embedded_relay = None
                    continue

                self.menu.show_connecting("Creating room...")
                try:
                    self._net_host = NetworkHost(relay_url)
                    room_code = self._net_host.start()
                except Exception as exc:
                    self.menu.show_error(f"Failed to host: {exc}")
                    self._net_host = None
                    if self._embedded_relay:
                        self._embedded_relay.stop()
                        self._embedded_relay = None
                    continue

                # Get host LAN IP for display
                import socket
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    host_ip = s.getsockname()[0]
                    s.close()
                except Exception:
                    host_ip = "unknown"

                # Waiting room loop
                waiting = True
                start_game = False
                while waiting:
                    # Poll network so join notifications arrive
                    self._net_host.poll_incoming()
                    wr = self.menu.show_waiting_room(
                        room_code, self._net_host.player_count, host_ip
                    )
                    if wr == "start":
                        start_game = True
                        waiting = False
                    elif wr == "cancel":
                        waiting = False
                    self.clock.tick(30)

                if start_game:
                    self._run_host_freeplay()

                if self._net_host:
                    self._net_host.stop()
                    self._net_host = None
                if self._embedded_relay:
                    self._embedded_relay.stop()
                    self._embedded_relay = None
                self._active_world = None

            elif choice == "join":
                result = self.menu.show_join_dialog()
                if result is None:
                    continue
                room_code, relay_url = result
                self.menu.show_connecting("Joining room...")
                try:
                    self._net_client = NetworkClient(relay_url, room_code)
                    self._net_client.start()
                except Exception as exc:
                    self.menu.show_error(f"Failed to join: {exc}")
                    self._net_client = None
                    continue

                self._run_client()

                if self._net_client:
                    self._net_client.stop()
                    self._net_client = None
                self._active_world = None

        pygame.quit()

    # ── Freeplay Mode ─────────────────────────────────────────

    def _run_freeplay(
        self,
        loaded_dna: list[DNA] | None = None,
        loaded_snapshot: dict | None = None,
    ) -> None:
        """Run continuous breeding mode — no generations, creatures breed in real time."""
        self.tools = PlayerTools()

        if loaded_snapshot:
            # Resume from full snapshot — restore exact simulation state
            creatures = loaded_snapshot["creatures"]
            world = World(creatures, settings=self.settings, tools=self.tools)
            world.freeplay = True
            world.generation = 0
            # Overwrite auto-generated entities with saved state
            world.food = loaded_snapshot["food"]
            world.hazards = loaded_snapshot["hazards"]
            world.biomes = loaded_snapshot["biomes"]
            self._active_world = world
            timers = loaded_snapshot.get("world_timers", {})
            world.elapsed_time = timers.get("elapsed_time", 0.0)
            world.season_time = timers.get("season_time", 0.0)
            world.day_night_time = timers.get("day_night_time", 0.0)
            world.total_births = timers.get("total_births", 0)
            world.total_deaths = timers.get("total_deaths", 0)
            world._food_spawn_accum = timers.get("food_spawn_accum", 0.0)
            # Restore tools state
            self.tools.drought_active = loaded_snapshot.get("tools_drought", False)
            self.tools.zones = loaded_snapshot.get("tools_zones", [])
            self.tools.barriers = loaded_snapshot.get("tools_barriers", [])
            # Restore freeplay tracking
            fp = loaded_snapshot.get("freeplay_state", {})
            self._freeplay_elapsed = fp.get("elapsed", 0.0)
            self._freeplay_peak_pop = fp.get("peak_pop", len(creatures))
            self._freeplay_last_births = fp.get("last_births", 0)
            self._freeplay_last_deaths = fp.get("last_deaths", 0)
            self._freeplay_births_per_min = fp.get("births_per_min", 0.0)
            self._freeplay_deaths_per_min = fp.get("deaths_per_min", 0.0)
            self._freeplay_history = fp.get("history", [])
            self._freeplay_history_timer = 0.0
            self._freeplay_stats_timer = 0.0
        elif loaded_dna:
            # Resume from save — use loaded DNA directly as starting population
            dna_list = loaded_dna
            world = self._create_world(dna_list)
            world.freeplay = True
            world.generation = 0
            self._freeplay_elapsed = 0.0
            self._freeplay_last_births = 0
            self._freeplay_last_deaths = 0
            self._freeplay_births_per_min = 0.0
            self._freeplay_deaths_per_min = 0.0
            self._freeplay_stats_timer = 0.0
            self._freeplay_peak_pop = len(dna_list)
            self._freeplay_history: list[dict] = []
            self._freeplay_history_timer = 0.0
        else:
            # Per-species initial populations
            dna_list = []
            for sp in self.settings.species_registry.all():
                for _ in range(sp.settings.freeplay_initial_population):
                    dna_list.append(DNA.random_for_species(sp.id))
            world = self._create_world(dna_list)
            world.freeplay = True
            world.generation = 0
            self._freeplay_elapsed = 0.0
            self._freeplay_last_births = 0
            self._freeplay_last_deaths = 0
            self._freeplay_births_per_min = 0.0
            self._freeplay_deaths_per_min = 0.0
            self._freeplay_stats_timer = 0.0
            self._freeplay_peak_pop = len(dna_list)
            self._freeplay_history: list[dict] = []
            self._freeplay_history_timer = 0.0

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                    continue

                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.settings_panel.toggle(self.settings)
                    continue

                if self.settings_panel.visible:
                    self.settings = self.settings_panel.handle_event(event, self.settings)
                    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
                        if self.settings_panel.consumes_click(*pygame.mouse.get_pos()):
                            continue

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    elif event.key == pygame.K_F10:
                        self._toggle_maximized()
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.fast_forward = not self.fast_forward
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.fast_forward_multiplier = min(20, self.fast_forward_multiplier + 1)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.fast_forward_multiplier = max(2, self.fast_forward_multiplier - 1)
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_e:
                        self.show_evolution_panel = not self.show_evolution_panel
                    elif event.key == pygame.K_ESCAPE:
                        if self.settings_panel.visible:
                            self.settings_panel.visible = False
                        else:
                            pause_result, self.settings = self.menu.show_pause_menu(
                                self.settings
                            )
                            if pause_result == "resume":
                                self.paused = False
                            elif pause_result == "save_quit":
                                # Save full simulation snapshot
                                freeplay_state = {
                                    "elapsed": self._freeplay_elapsed,
                                    "peak_pop": self._freeplay_peak_pop,
                                    "last_births": self._freeplay_last_births,
                                    "last_deaths": self._freeplay_last_deaths,
                                    "births_per_min": self._freeplay_births_per_min,
                                    "deaths_per_min": self._freeplay_deaths_per_min,
                                    "history": self._freeplay_history,
                                }
                                save_snapshot(
                                    world=world,
                                    settings_dict=self.settings.to_dict(),
                                    freeplay_state=freeplay_state,
                                    tools=self.tools,
                                )
                                alive = [c for c in world.creatures if c.alive]
                                if alive:
                                    top_dna = [c.dna for c in sorted(
                                        alive, key=lambda c: c.food_eaten, reverse=True,
                                    )[:self.settings.top_performers_count]]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filepath = f"species/freeplay_{timestamp}.json"
                                    save_species(top_dna, filepath, generation=0)
                                return
                            elif pause_result.startswith("import_species:"):
                                imported_id = pause_result.split(":", 1)[1]
                                sp = self.settings.species_registry.get(imported_id)
                                if sp and hasattr(sp, "_imported_dna"):
                                    for dna in sp._imported_dna:
                                        x = random.uniform(50, self.settings.world_width - 50)
                                        y = random.uniform(50, self.settings.world_height - 50)
                                        child = Creature(dna, x, y, species=sp)
                                        child.energy = sp.settings.freeplay_child_energy
                                        world.creatures.append(child)
                                    del sp._imported_dna
                                self.paused = False
                            elif pause_result in ("main_menu", "restart"):
                                if pause_result == "restart":
                                    self.tools.reset()
                                    dna_list = []
                                    for sp in self.settings.species_registry.all():
                                        for _ in range(sp.settings.freeplay_initial_population):
                                            dna_list.append(DNA.random_for_species(sp.id))
                                    world = self._create_world(dna_list)
                                    world.freeplay = True
                                    world.generation = 0
                                    self._freeplay_elapsed = 0.0
                                    self._freeplay_peak_pop = len(dna_list)
                                    self._freeplay_history.clear()
                                    self._freeplay_history_timer = 0.0
                                    self._freeplay_last_births = 0
                                    self._freeplay_last_deaths = 0
                                    self._freeplay_births_per_min = 0.0
                                    self._freeplay_deaths_per_min = 0.0
                                    self._freeplay_stats_timer = 0.0
                                    self.renderer.reset_tracking()
                                    continue
                                return
                    # Tool hotkeys (1-6)
                    elif pygame.K_1 <= event.key <= pygame.K_6:
                        tool_idx = event.key - pygame.K_1
                        if tool_idx < len(TOOL_LIST):
                            self.tools.select_tool(TOOL_LIST[tool_idx])

                # Right-click to inspect creature
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    mx, my = event.pos
                    if not self.settings_panel.consumes_click(mx, my):
                        wmx, wmy = self._screen_to_world(mx, my)
                        if not self.renderer.try_select_creature(world, wmx, wmy):
                            self.renderer.deselect_creature()

                # Left-click for tools
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if self.settings_panel.consumes_click(mx, my):
                        continue
                    toolbar_x = config.WINDOW_WIDTH - 380
                    if my < 75 and mx > toolbar_x:
                        btn_w = 58
                        gap = 4
                        for i, tool in enumerate(TOOL_LIST):
                            tx = toolbar_x + i * (btn_w + gap)
                            if tx <= mx <= tx + btn_w:
                                self.tools.select_tool(tool)
                                break
                    elif self.tools.active_tool == "none":
                        wmx, wmy = self._screen_to_world(mx, my)
                        if not self.renderer.try_select_creature(world, wmx, wmy):
                            self.renderer.deselect_creature()
                    else:
                        wmx, wmy = self._screen_to_world(mx, my)
                        food_positions = self.tools.on_mouse_down(wmx, wmy)
                        for fx, fy in food_positions:
                            world.add_food_at(fx, fy)

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    mx, my = event.pos
                    wmx, wmy = self._screen_to_world(mx, my)
                    self.tools.on_mouse_up(wmx, wmy)

            # Update settings panel dragging
            self.settings = self.settings_panel.update_dragging(self.settings)

            # Update simulation
            if not self.paused:
                if self.fast_forward:
                    # Use fixed timestep to avoid dt inflation from slow frames
                    step_dt = 1.0 / FPS
                    for _ in range(self.fast_forward_multiplier):
                        world.update(step_dt)
                        world.check_breeding()
                    sim_dt = step_dt * self.fast_forward_multiplier
                else:
                    world.update(dt)
                    world.check_breeding()
                    sim_dt = dt

                self._freeplay_elapsed += sim_dt

                # Periodic cleanup and stats
                self._freeplay_stats_timer += sim_dt
                self._freeplay_history_timer += sim_dt
                if self._freeplay_stats_timer >= 5.0:
                    world.remove_dead_creatures(min_dead_age=3.0)
                    # Update rolling birth/death rates
                    births_delta = world.total_births - self._freeplay_last_births
                    deaths_delta = world.total_deaths - self._freeplay_last_deaths
                    interval = self._freeplay_stats_timer
                    self._freeplay_births_per_min = births_delta / interval * 60
                    self._freeplay_deaths_per_min = deaths_delta / interval * 60
                    world._freeplay_births_per_min = self._freeplay_births_per_min
                    world._freeplay_deaths_per_min = self._freeplay_deaths_per_min
                    self._freeplay_last_births = world.total_births
                    self._freeplay_last_deaths = world.total_deaths
                    self._freeplay_stats_timer = 0.0

                # Record population history snapshot every 10 seconds
                if self._freeplay_history_timer >= 10.0:
                    self._freeplay_history.append(
                        self._build_freeplay_snapshot(world)
                    )
                    # Keep last 360 snapshots (~1 hour at 10s intervals)
                    if len(self._freeplay_history) > 360:
                        self._freeplay_history = self._freeplay_history[-360:]
                    self._freeplay_history_timer = 0.0

                alive = world.alive_count()
                if alive > self._freeplay_peak_pop:
                    self._freeplay_peak_pop = alive

            # Check for per-species extinction
            if not self.paused:
                from pangea.settings import (
                    EXTINCTION_RESPAWN_BEST,
                    EXTINCTION_RESPAWN_RANDOM,
                )
                for sp in self.settings.species_registry.all():
                    if world.alive_count_by_species(sp.id) > 0:
                        continue
                    if not sp.enabled:
                        continue
                    ss = sp.settings
                    if ss.extinction_mode == EXTINCTION_RESPAWN_BEST:
                        # Select from dead creatures of this species
                        sp_creatures = [
                            c for c in world.creatures if c.dna.species_id == sp.id
                        ]
                        top_dna = select_top(
                            sp_creatures,
                            min(ss.top_performers_count, len(sp_creatures)),
                            self.settings,
                        )
                        respawn_count = ss.freeplay_initial_population
                        if top_dna:
                            new_dna = create_next_generation(
                                top_dna,
                                population_size=respawn_count,
                                mutation_rate=ss.mutation_rate,
                                mutation_strength=ss.mutation_strength,
                                crossover_rate=ss.crossover_rate,
                                min_parents=ss.min_population,
                                weight_clamp=ss.weight_clamp,
                                trait_mutation_range=ss.trait_mutation_range,
                            )
                        else:
                            new_dna = [
                                DNA.random_for_species(sp.id)
                                for _ in range(respawn_count)
                            ]
                        for dna in new_dna:
                            dna.species_id = sp.id
                            x = random.uniform(50, self.settings.world_width - 50)
                            y = random.uniform(50, self.settings.world_height - 50)
                            child = Creature(dna, x, y, species=sp)
                            child.energy = ss.freeplay_child_energy
                            world.creatures.append(child)
                    elif ss.extinction_mode == EXTINCTION_RESPAWN_RANDOM:
                        respawn_count = max(1, ss.freeplay_initial_population // 2)
                        for _ in range(respawn_count):
                            dna = DNA.random_for_species(sp.id)
                            x = random.uniform(50, self.settings.world_width - 50)
                            y = random.uniform(50, self.settings.world_height - 50)
                            child = Creature(dna, x, y, species=sp)
                            child.energy = ss.freeplay_child_energy
                            world.creatures.append(child)
                    # EXTINCTION_PERMANENT: do nothing

                # Total extinction — all species gone
                if world.alive_count() == 0:
                    self.renderer.reset_tracking()

            # Render
            self.renderer.draw(
                world, self.paused,
                tools=self.tools,
                show_toolbar=True,
                fast_forward=self.fast_forward_multiplier if self.fast_forward else 0,
                debug=self.debug_mode,
            )
            self.renderer.draw_creature_stats(world)
            if self.show_evolution_panel:
                self.renderer.draw_evolution_panel(
                    world, self._freeplay_history,
                )
            if self.tools.active_tool != "none":
                self.renderer.draw_tool_cursor(self.tools)
            self.settings_panel.draw(self.screen, self.settings, dt)
            pygame.display.flip()

    # ── Host Mode (Freeplay) ─────────────────────────────────

    def _run_host_freeplay(self) -> None:
        """Run freeplay mode as network host."""
        self.tools = PlayerTools()
        # Per-species initial populations
        dna_list = []
        for sp in self.settings.species_registry.all():
            for _ in range(sp.settings.freeplay_initial_population):
                dna_list.append(DNA.random_for_species(sp.id))
        world = self._create_world(dna_list)
        world.freeplay = True
        world.generation = 0

        self._freeplay_elapsed = 0.0
        self._freeplay_last_births = 0
        self._freeplay_last_deaths = 0
        self._freeplay_births_per_min = 0.0
        self._freeplay_deaths_per_min = 0.0
        self._freeplay_stats_timer = 0.0
        self._freeplay_peak_pop = len(dna_list)
        self._freeplay_history: list[dict] = []
        self._freeplay_history_timer = 0.0
        frame_counter = 0

        # Send full state to any already-connected clients
        if self._net_host:
            self._net_host.broadcast_full_state(
                full_state_from_world(world, self.settings, self.tools, 0)
            )

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)
            frame_counter += 1

            # Process remote actions
            if self._net_host:
                for msg in self._net_host.poll_incoming():
                    self._apply_remote_action(msg, world)

            # Handle local events (identical to _run_freeplay)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.settings_panel.toggle(self.settings)
                    continue
                if self.settings_panel.visible:
                    self.settings = self.settings_panel.handle_event(event, self.settings)
                    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
                        if self.settings_panel.consumes_click(*pygame.mouse.get_pos()):
                            continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    elif event.key == pygame.K_F10:
                        self._toggle_maximized()
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.fast_forward = not self.fast_forward
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.fast_forward_multiplier = min(20, self.fast_forward_multiplier + 1)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.fast_forward_multiplier = max(2, self.fast_forward_multiplier - 1)
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_e:
                        self.show_evolution_panel = not self.show_evolution_panel
                    elif event.key == pygame.K_ESCAPE:
                        if self.settings_panel.visible:
                            self.settings_panel.visible = False
                        else:
                            pause_result, self.settings = self.menu.show_pause_menu(
                                self.settings,
                            )
                            if pause_result == "resume":
                                self.paused = False
                            elif pause_result == "main_menu":
                                return
                            elif pause_result == "save_quit":
                                freeplay_state = {
                                    "elapsed": self._freeplay_elapsed,
                                    "peak_pop": self._freeplay_peak_pop,
                                    "last_births": self._freeplay_last_births,
                                    "last_deaths": self._freeplay_last_deaths,
                                    "births_per_min": self._freeplay_births_per_min,
                                    "deaths_per_min": self._freeplay_deaths_per_min,
                                    "history": self._freeplay_history,
                                }
                                save_snapshot(
                                    world=world, settings_dict=self.settings.to_dict(),
                                    freeplay_state=freeplay_state, tools=self.tools,
                                )
                                return
                            elif pause_result.startswith("import_species:"):
                                imported_id = pause_result.split(":", 1)[1]
                                sp = self.settings.species_registry.get(imported_id)
                                if sp and hasattr(sp, "_imported_dna"):
                                    for dna in sp._imported_dna:
                                        x = random.uniform(50, self.settings.world_width - 50)
                                        y = random.uniform(50, self.settings.world_height - 50)
                                        child = Creature(dna, x, y, species=sp)
                                        child.energy = sp.settings.freeplay_child_energy
                                        world.creatures.append(child)
                                    del sp._imported_dna
                                self.paused = False
                            elif pause_result == "restart":
                                self.tools.reset()
                                dna_list = []
                                for sp in self.settings.species_registry.all():
                                    for _ in range(sp.settings.freeplay_initial_population):
                                        dna_list.append(DNA.random_for_species(sp.id))
                                world = self._create_world(dna_list)
                                world.freeplay = True
                                world.generation = 0
                                self._freeplay_elapsed = 0.0
                                self._freeplay_peak_pop = len(dna_list)
                                self._freeplay_history.clear()
                                self._freeplay_history_timer = 0.0
                                self._freeplay_last_births = 0
                                self._freeplay_last_deaths = 0
                                self._freeplay_births_per_min = 0.0
                                self._freeplay_deaths_per_min = 0.0
                                self._freeplay_stats_timer = 0.0
                                self.renderer.reset_tracking()
                                frame_counter = 0
                                continue
                    elif pygame.K_1 <= event.key <= pygame.K_6:
                        tool_idx = event.key - pygame.K_1
                        if tool_idx < len(TOOL_LIST):
                            self.tools.select_tool(TOOL_LIST[tool_idx])

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    mx, my = event.pos
                    if not self.settings_panel.consumes_click(mx, my):
                        wmx, wmy = self._screen_to_world(mx, my)
                        if not self.renderer.try_select_creature(world, wmx, wmy):
                            self.renderer.deselect_creature()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if self.settings_panel.consumes_click(mx, my):
                        continue
                    toolbar_x = config.WINDOW_WIDTH - 380
                    if my < 75 and mx > toolbar_x:
                        btn_w = 58
                        gap = 4
                        for i, tool in enumerate(TOOL_LIST):
                            tx = toolbar_x + i * (btn_w + gap)
                            if tx <= mx <= tx + btn_w:
                                self.tools.select_tool(tool)
                                break
                    elif self.tools.active_tool == "none":
                        wmx, wmy = self._screen_to_world(mx, my)
                        if not self.renderer.try_select_creature(world, wmx, wmy):
                            self.renderer.deselect_creature()
                    else:
                        wmx, wmy = self._screen_to_world(mx, my)
                        food_positions = self.tools.on_mouse_down(wmx, wmy)
                        for fx, fy in food_positions:
                            world.add_food_at(fx, fy)

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    mx, my = event.pos
                    wmx, wmy = self._screen_to_world(mx, my)
                    self.tools.on_mouse_up(wmx, wmy)

            self.settings = self.settings_panel.update_dragging(self.settings)

            if not self.paused:
                if self.fast_forward:
                    step_dt = 1.0 / FPS
                    for _ in range(self.fast_forward_multiplier):
                        world.update(step_dt)
                        world.check_breeding()
                    sim_dt = step_dt * self.fast_forward_multiplier
                else:
                    world.update(dt)
                    world.check_breeding()
                    sim_dt = dt

                self._freeplay_elapsed += sim_dt
                self._freeplay_stats_timer += sim_dt
                self._freeplay_history_timer += sim_dt

                if self._freeplay_stats_timer >= 5.0:
                    world.remove_dead_creatures(min_dead_age=3.0)
                    births_delta = world.total_births - self._freeplay_last_births
                    deaths_delta = world.total_deaths - self._freeplay_last_deaths
                    interval = self._freeplay_stats_timer
                    self._freeplay_births_per_min = births_delta / interval * 60
                    self._freeplay_deaths_per_min = deaths_delta / interval * 60
                    world._freeplay_births_per_min = self._freeplay_births_per_min
                    world._freeplay_deaths_per_min = self._freeplay_deaths_per_min
                    self._freeplay_last_births = world.total_births
                    self._freeplay_last_deaths = world.total_deaths
                    self._freeplay_stats_timer = 0.0

                if self._freeplay_history_timer >= 10.0:
                    self._freeplay_history.append(
                        self._build_freeplay_snapshot(world)
                    )
                    if len(self._freeplay_history) > 360:
                        self._freeplay_history = self._freeplay_history[-360:]
                    self._freeplay_history_timer = 0.0

                alive = world.alive_count()
                if alive > self._freeplay_peak_pop:
                    self._freeplay_peak_pop = alive

            # Per-species extinction handling
            if not self.paused:
                from pangea.settings import (
                    EXTINCTION_RESPAWN_BEST,
                    EXTINCTION_RESPAWN_RANDOM,
                )
                for sp in self.settings.species_registry.all():
                    if world.alive_count_by_species(sp.id) > 0:
                        continue
                    if not sp.enabled:
                        continue
                    ss = sp.settings
                    if ss.extinction_mode == EXTINCTION_RESPAWN_BEST:
                        sp_creatures = [
                            c for c in world.creatures if c.dna.species_id == sp.id
                        ]
                        top_dna = select_top(
                            sp_creatures,
                            min(ss.top_performers_count, len(sp_creatures)),
                            self.settings,
                        )
                        respawn_count = ss.freeplay_initial_population
                        if top_dna:
                            new_dna = create_next_generation(
                                top_dna,
                                population_size=respawn_count,
                                mutation_rate=ss.mutation_rate,
                                mutation_strength=ss.mutation_strength,
                                crossover_rate=ss.crossover_rate,
                                min_parents=ss.min_population,
                                weight_clamp=ss.weight_clamp,
                                trait_mutation_range=ss.trait_mutation_range,
                            )
                        else:
                            new_dna = [
                                DNA.random_for_species(sp.id)
                                for _ in range(respawn_count)
                            ]
                        for dna in new_dna:
                            dna.species_id = sp.id
                            x = random.uniform(50, self.settings.world_width - 50)
                            y = random.uniform(50, self.settings.world_height - 50)
                            child = Creature(dna, x, y, species=sp)
                            child.energy = ss.freeplay_child_energy
                            world.creatures.append(child)
                    elif ss.extinction_mode == EXTINCTION_RESPAWN_RANDOM:
                        respawn_count = max(1, ss.freeplay_initial_population // 2)
                        for _ in range(respawn_count):
                            dna = DNA.random_for_species(sp.id)
                            x = random.uniform(50, self.settings.world_width - 50)
                            y = random.uniform(50, self.settings.world_height - 50)
                            child = Creature(dna, x, y, species=sp)
                            child.energy = ss.freeplay_child_energy
                            world.creatures.append(child)

                if world.alive_count() == 0:
                    self.renderer.reset_tracking()

            # Broadcast snapshot
            if self._net_host and frame_counter % NET_SNAPSHOT_INTERVAL == 0:
                self._net_host.broadcast_snapshot(snapshot_from_world(world))

            # Render
            self.renderer.draw(
                world, self.paused, tools=self.tools,
                show_toolbar=True,
                fast_forward=self.fast_forward_multiplier if self.fast_forward else 0,
                debug=self.debug_mode,
            )
            self.renderer.draw_creature_stats(world)
            if self.show_evolution_panel:
                self.renderer.draw_evolution_panel(world, self._freeplay_history)
            if self.tools.active_tool != "none":
                self.renderer.draw_tool_cursor(self.tools)
            self.settings_panel.draw(self.screen, self.settings, dt)
            pygame.display.flip()

    # ── Client Mode ──────────────────────────────────────────

    def _run_client(self) -> None:
        """Run as a network client — receive snapshots, send tool actions."""
        assert self._net_client is not None

        # Wait for full state from host (no timeout — host may still be in waiting room)
        world = None
        generation = 1

        while world is None:
            self.clock.tick(30)
            self.menu.show_connecting("Waiting for host to start the game...  (ESC to cancel)")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            for msg in self._net_client.poll_incoming():
                if msg.get("t") == MsgType.FULL_STATE:
                    world, self.settings, self.tools, _, generation, self.generation_history = (
                        apply_full_state(msg)
                    )
                    break
                elif msg.get("t") == MsgType.SNAPSHOT:
                    # Got a snapshot before full_state — create minimal world
                    world = World([], settings=self.settings, tools=self.tools)
                    self._active_world = world
                    apply_snapshot(world, msg)
                    break
                elif msg.get("t") == MsgType.HOST_LEFT:
                    self.menu.show_error("Host disconnected.")
                    return

            if not self._net_client.connected:
                self.menu.show_error("Lost connection to server.")
                return

        self._active_world = world
        self.tools = PlayerTools()

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            # Process network messages
            for msg in self._net_client.poll_incoming():
                msg_type = msg.get("t")
                if msg_type == MsgType.SNAPSHOT:
                    apply_snapshot(world, msg)
                elif msg_type == MsgType.FULL_STATE:
                    world, self.settings, self.tools, _, generation, self.generation_history = (
                        apply_full_state(msg)
                    )
                    self._active_world = world
                elif msg_type == MsgType.GENERATION_END:
                    generation = msg.get("generation", generation)
                    stats = msg.get("stats")
                    if stats:
                        self.generation_history.append(stats)
                elif msg_type == MsgType.HOST_LEFT:
                    self.menu.show_error("Host disconnected.")
                    return

            if not self._net_client.connected:
                self.menu.show_error("Lost connection to server.")
                return

            # Handle local events — tools send actions to host
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    elif event.key == pygame.K_F10:
                        self._toggle_maximized()
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_e:
                        self.show_evolution_panel = not self.show_evolution_panel
                    elif event.key == pygame.K_ESCAPE:
                        return  # Disconnect and go to menu
                    elif pygame.K_1 <= event.key <= pygame.K_6:
                        tool_idx = event.key - pygame.K_1
                        if tool_idx < len(TOOL_LIST):
                            self.tools.select_tool(TOOL_LIST[tool_idx])

                # Right-click to inspect creature locally
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    mx, my = event.pos
                    wmx, wmy = self._screen_to_world(mx, my)
                    if not self.renderer.try_select_creature(world, wmx, wmy):
                        self.renderer.deselect_creature()

                # Left-click: toolbar or send tool action to host
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    toolbar_x = config.WINDOW_WIDTH - 380
                    if my < 75 and mx > toolbar_x:
                        btn_w = 58
                        gap = 4
                        for i, tool in enumerate(TOOL_LIST):
                            tx = toolbar_x + i * (btn_w + gap)
                            if tx <= mx <= tx + btn_w:
                                self.tools.select_tool(tool)
                                break
                    elif self.tools.active_tool == "none":
                        wmx, wmy = self._screen_to_world(mx, my)
                        if not self.renderer.try_select_creature(world, wmx, wmy):
                            self.renderer.deselect_creature()
                    else:
                        wmx, wmy = self._screen_to_world(mx, my)
                        self._net_client.send_tool_action(
                            tool_action_msg(self.tools.active_tool, wmx, wmy)
                        )

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.tools.active_tool == "barrier":
                        mx, my = event.pos
                        wmx, wmy = self._screen_to_world(mx, my)
                        self._net_client.send_tool_action(
                            tool_action_msg("mouse_up", wmx, wmy)
                        )

            # Render (no world.update — client is view-only)
            self.renderer.draw(
                world, False, tools=self.tools,
                show_toolbar=True,
                fast_forward=0,
                debug=self.debug_mode,
            )
            self.renderer.draw_creature_stats(world)
            if self.show_evolution_panel:
                self.renderer.draw_evolution_panel(world, self.generation_history)
            if self.tools.active_tool != "none":
                self.renderer.draw_tool_cursor(self.tools)
            pygame.display.flip()

    # ── Network Helpers ──────────────────────────────────────

    def _apply_remote_action(self, msg: dict, world: World) -> None:
        """Apply a remote client's action to the host's world."""
        msg_type = msg.get("t")

        if msg_type == MsgType.TOOL_ACTION:
            tool = msg.get("tool", "")
            x, y = msg.get("x", 0), msg.get("y", 0)

            if tool == "mouse_up":
                self.tools.on_mouse_up(x, y)
            elif tool == "food":
                # Temporarily set tool, apply, restore
                prev = self.tools.active_tool
                self.tools.active_tool = "food"
                food_positions = self.tools.on_mouse_down(x, y)
                for fx, fy in food_positions:
                    world.add_food_at(fx, fy)
                self.tools.active_tool = prev
            elif tool == "poison":
                prev = self.tools.active_tool
                self.tools.active_tool = "poison"
                self.tools.on_mouse_down(x, y)
                self.tools.active_tool = prev
            elif tool == "bounty":
                prev = self.tools.active_tool
                self.tools.active_tool = "bounty"
                self.tools.on_mouse_down(x, y)
                self.tools.active_tool = prev
            elif tool == "barrier":
                prev = self.tools.active_tool
                self.tools.active_tool = "barrier"
                self.tools.on_mouse_down(x, y)
                self.tools.active_tool = prev
            elif tool == "drought":
                self.tools.drought_active = not self.tools.drought_active

        elif msg_type == MsgType.SETTINGS_CHANGE:
            changes = msg.get("changes", {})
            for k, v in changes.items():
                if hasattr(self.settings, k):
                    setattr(self.settings, k, v)

        elif msg_type == MsgType.CLIENT_JOINED:
            # Send full state to newly joined client
            if self._net_host:
                self._net_host.broadcast_full_state(
                    full_state_from_world(
                        world, self.settings, self.tools,
                        world.generation, self.generation_history,
                    )
                )

    # ── Freeplay Snapshot Builder ─────────────────────────────

    def _build_freeplay_snapshot(self, world: World) -> dict:
        """Build a history snapshot with per-species stats for the evolution panel."""
        alive_creatures = [c for c in world.creatures if c.alive]
        n_alive = len(alive_creatures)
        registry = self.settings.species_registry

        # Group alive creatures by species
        by_species: dict[str, list] = {sp.id: [] for sp in registry.all()}
        for c in alive_creatures:
            by_species.setdefault(c.dna.species_id, []).append(c)

        def _species_stats(creatures: list) -> dict:
            n = len(creatures)
            if n == 0:
                return {
                    "count": 0, "avg_gen": 0, "avg_food": 0, "avg_energy": 0,
                    "avg_age": 0, "avg_offspring": 0,
                    "avg_speed": 0, "avg_size": 0, "avg_vision": 0,
                    "avg_efficiency": 0, "avg_lifespan": 0,
                }
            return {
                "count": n,
                "avg_gen": sum(c.generation for c in creatures) / n,
                "avg_food": sum(c.food_eaten for c in creatures) / n,
                "avg_energy": sum(c.energy for c in creatures) / n,
                "avg_age": sum(c.age for c in creatures) / n,
                "avg_offspring": sum(c.offspring_count for c in creatures) / n,
                "avg_speed": sum(c.dna.speed for c in creatures) / n,
                "avg_size": sum(c.dna.size for c in creatures) / n,
                "avg_vision": sum(c.dna.vision for c in creatures) / n,
                "avg_efficiency": sum(c.dna.efficiency for c in creatures) / n,
                "avg_lifespan": sum(c.dna.lifespan for c in creatures) / n,
            }

        snapshot = {
            "time": self._freeplay_elapsed,
            "population": n_alive,
            "births": world.total_births,
            "deaths": world.total_deaths,
            "births_per_min": self._freeplay_births_per_min,
            "deaths_per_min": self._freeplay_deaths_per_min,
            "avg_gen": (
                sum(c.generation for c in alive_creatures) / n_alive
                if n_alive else 0
            ),
        }
        # Per-species population counts and detailed stats
        for sp in registry.all():
            snapshot[sp.id] = len(by_species.get(sp.id, []))
            snapshot[f"{sp.id}_stats"] = _species_stats(by_species.get(sp.id, []))
        return snapshot

    # ── World Creation Helpers ───────────────────────────────

    def _create_world(self, dna_list: list[DNA]) -> World:
        """Create a World with creatures from a list of DNA."""
        registry = self.settings.species_registry
        creatures = []
        for dna in dna_list:
            x = random.uniform(50, self.settings.world_width - 50)
            y = random.uniform(50, self.settings.world_height - 50)
            sp = registry.get(dna.species_id)
            creatures.append(Creature(dna, x, y, species=sp))
        world = World(creatures, settings=self.settings, tools=self.tools)
        self._active_world = world
        return world
