"""
Simulation — the main game loop tying everything together.
============================================================
Initializes pygame, runs the menu, dispatches to the correct mode
(Isolation or Convergence), and manages the generation cycle.

Controls:
    SPACE  → Pause / unpause
    F      → Toggle fast-forward (multiple sim steps per frame)
    +/-    → Adjust fast-forward speed multiplier (2×–20×)
    D      → Toggle debug overlay (vision ranges, energy bars)
    E      → Toggle evolution panel (minimap, trait graphs)
    S      → Toggle settings panel (right-side overlay with save/load)
    F10    → Toggle maximized window (keeps title bar / taskbar)
    F11    → Toggle fullscreen
    1-6    → Select player tool (Isolation mode)
    ESC    → Pause menu (or close settings panel if open)
    Left-click → Use active tool
"""

from __future__ import annotations

import random
from datetime import datetime

import pygame

import pangea.config as config
from pangea.config import (
    CREATURES_PER_LINEAGE,
    FPS,
)
from pangea.creature import Creature
from pangea.dna import DNA
from pangea.evolution import (
    create_next_generation,
    evaluate_fitness,
    select_top,
)
from pangea.menu import Menu
from pangea.renderer import Renderer
from pangea.save_load import (
    load_game, load_snapshot, load_species, save_game, save_snapshot, save_species,
)
from pangea.settings import SimSettings
from pangea.settings_panel import SettingsPanel
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
            elif choice in ("isolation", "convergence", "freeplay"):
                mode_result = self.menu.show_mode_select(choice)
                if mode_result is None:
                    continue  # user pressed Back
                elif mode_result == "new":
                    # Start fresh
                    if choice == "isolation":
                        self._run_isolation()
                    elif choice == "convergence":
                        self._run_convergence()
                    elif choice == "freeplay":
                        self._run_freeplay()
                elif isinstance(mode_result, dict):
                    # Load a save
                    save_data = mode_result
                    loaded_settings = SimSettings.from_dict(save_data["settings"])
                    # Keep window size from current session
                    loaded_settings.world_width = self.settings.world_width
                    loaded_settings.world_height = self.settings.world_height
                    self.settings = loaded_settings

                    if choice == "isolation":
                        self._run_isolation(
                            loaded_dna=save_data["creatures"],
                            loaded_generation=save_data["generation"],
                        )
                    elif choice == "convergence":
                        extra = save_data.get("extra") or {}
                        self._run_convergence(
                            loaded_dna=save_data["creatures"],
                            loaded_generation=save_data["generation"],
                            loaded_extra=extra,
                        )
                    elif choice == "freeplay":
                        if save_data.get("snapshot"):
                            self._run_freeplay(loaded_snapshot=save_data)
                        else:
                            self._run_freeplay(
                                loaded_dna=save_data["creatures"],
                            )
                self._active_world = None

        pygame.quit()

    # ── Isolation Mode ───────────────────────────────────────

    def _run_isolation(
        self,
        loaded_dna: list[DNA] | None = None,
        loaded_generation: int = 0,
    ) -> None:
        """Run the simulation in isolation mode (single-user evolution)."""
        self.tools = PlayerTools()

        if loaded_dna:
            # Resume from save — expand loaded top performers into a full population
            pop = self.settings.population_size
            dna_list = create_next_generation(
                loaded_dna,
                population_size=pop,
                mutation_rate=self.settings.mutation_rate,
                mutation_strength=self.settings.mutation_strength,
                crossover_rate=self.settings.crossover_rate,
                min_parents=self.settings.min_population,
                weight_clamp=self.settings.weight_clamp,
                trait_mutation_range=self.settings.trait_mutation_range,
            )
            generation = loaded_generation + 1
        else:
            pop = self.settings.population_size
            dna_list = [DNA.random() for _ in range(pop)]
            generation = 1

        world = self._create_world(dna_list)
        world.generation = generation

        while self.running:
            # Check max generations limit
            if (self.settings.max_generations > 0
                    and generation > self.settings.max_generations):
                return

            result = self._run_generation(world, mode="isolation")

            if result == "main_menu":
                return
            elif result == "save_quit":
                top_dna = select_top(world.creatures, self.settings.top_performers_count, self.settings)
                # Save full game state for resuming later
                save_game(
                    mode="isolation",
                    dna_list=top_dna,
                    generation=generation,
                    settings_dict=self.settings.to_dict(),
                )
                # Also save species file for convergence mode use
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"species/species_{timestamp}.json"
                save_species(top_dna, filepath, generation=generation)
                return
            elif result == "restart":
                self.tools.reset()
                self.generation_history.clear()
                pop = self.settings.population_size
                dna_list = [DNA.random() for _ in range(pop)]
                world = self._create_world(dna_list)
                generation = 1
                world.generation = generation
                self.renderer.reset_tracking()
                continue

            # Evolve to next generation
            top_dna = select_top(world.creatures, self.settings.top_performers_count, self.settings)

            fitnesses = [evaluate_fitness(c, self.settings) for c in world.creatures]
            best = max(fitnesses)
            avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0

            # Record generation history for evolution panel
            all_creatures = world.creatures
            alive = [c for c in all_creatures if c.alive]
            n = len(alive) if alive else 1
            n_creatures = len(all_creatures)
            self.generation_history.append({
                "gen": generation,
                "avg_speed": sum(c.dna.speed for c in all_creatures) / n_creatures,
                "avg_size": sum(c.dna.size for c in all_creatures) / n_creatures,
                "avg_vision": sum(c.dna.vision for c in all_creatures) / n_creatures,
                "avg_efficiency": sum(c.dna.efficiency for c in all_creatures) / n_creatures,
                "avg_lifespan": sum(c.dna.lifespan for c in all_creatures) / n_creatures,
                "avg_food": sum(c.food_eaten for c in all_creatures) / n_creatures,
                "alive_pct": len(alive) / n_creatures * 100,
                "herbivores": sum(1 for c in all_creatures if c.dna.diet == 0),
                "carnivores": sum(1 for c in all_creatures if c.dna.diet == 1),
                "scavengers": sum(1 for c in all_creatures if c.dna.diet == 2),
            })

            self.renderer.draw_generation_stats(world, best, avg, mode="isolation")
            pygame.display.flip()
            pygame.time.wait(1500)

            # Create next generation (tools/zones persist across generations)
            generation += 1
            pop = self.settings.population_size
            dna_list = create_next_generation(
                top_dna,
                population_size=pop,
                mutation_rate=self.settings.mutation_rate,
                mutation_strength=self.settings.mutation_strength,
                crossover_rate=self.settings.crossover_rate,
                min_parents=self.settings.min_population,
                weight_clamp=self.settings.weight_clamp,
                trait_mutation_range=self.settings.trait_mutation_range,
            )
            world = self._create_world(dna_list)
            world.generation = generation
            self.renderer.reset_tracking()

    # ── Convergence Mode ─────────────────────────────────────

    def _run_convergence(
        self,
        loaded_dna: list[DNA] | None = None,
        loaded_generation: int = 0,
        loaded_extra: dict | None = None,
    ) -> None:
        """Run the simulation in convergence mode (two competing lineages)."""
        if loaded_dna and loaded_extra:
            # Resume from save — split DNA back into A and B lineages
            file_a = loaded_extra.get("file_a", "")
            file_b = loaded_extra.get("file_b", "")
            a_count = loaded_extra.get("a_count", len(loaded_dna) // 2)
            dna_a = loaded_dna[:a_count] if a_count > 0 else loaded_dna[:1]
            dna_b = loaded_dna[a_count:] if a_count < len(loaded_dna) else loaded_dna[-1:]

            a_total_food = loaded_extra.get("a_total_food", 0)
            b_total_food = loaded_extra.get("b_total_food", 0)
            a_survived_gens = loaded_extra.get("a_survived_gens", 0)
            b_survived_gens = loaded_extra.get("b_survived_gens", 0)
            a_alive = loaded_extra.get("a_alive", True)
            b_alive = loaded_extra.get("b_alive", True)
            generation = loaded_generation + 1
        else:
            result = self.menu.show_file_select()
            if result is None:
                return

            file_a, file_b = result
            try:
                dna_a, _ = load_species(file_a)
                dna_b, _ = load_species(file_b)
            except Exception as exc:
                self.menu.show_error(f"Failed to load species: {exc}")
                return

            if not dna_a or not dna_b:
                self.menu.show_error("Species file has no creatures.")
                return

            a_total_food = 0
            b_total_food = 0
            a_survived_gens = 0
            b_survived_gens = 0
            a_alive = True
            b_alive = True
            generation = 1

        max_gens = self.settings.convergence_max_generations

        world = self._create_convergence_world(dna_a, dna_b)
        world.generation = generation

        while self.running and generation <= max_gens:
            result = self._run_generation(world, mode="convergence")

            if result == "main_menu":
                return
            elif result == "save_quit":
                # Gather top DNA from each lineage
                a_creatures = [c for c in world.creatures if c.lineage == "A"]
                b_creatures = [c for c in world.creatures if c.lineage == "B"]
                top_n = max(1, self.settings.top_performers_count // 2)
                top_a = select_top(a_creatures, min(top_n, len(a_creatures)), self.settings)
                top_b = select_top(b_creatures, min(top_n, len(b_creatures)), self.settings)
                save_game(
                    mode="convergence",
                    dna_list=top_a + top_b,
                    generation=generation,
                    settings_dict=self.settings.to_dict(),
                    extra={
                        "file_a": file_a, "file_b": file_b,
                        "a_total_food": a_total_food, "b_total_food": b_total_food,
                        "a_survived_gens": a_survived_gens, "b_survived_gens": b_survived_gens,
                        "a_alive": a_alive, "b_alive": b_alive,
                        "a_count": len(top_a), "b_count": len(top_b),
                    },
                )
                return
            elif result == "restart":
                try:
                    dna_a, _ = load_species(file_a)
                    dna_b, _ = load_species(file_b)
                except Exception as exc:
                    self.menu.show_error(f"Failed to reload species: {exc}")
                    return
                generation = 1
                a_total_food = b_total_food = 0
                a_survived_gens = b_survived_gens = 0
                a_alive = b_alive = True
                world = self._create_convergence_world(dna_a, dna_b)
                world.generation = generation
                self.renderer.reset_tracking()
                continue

            gen_a_food = world.food_eaten_by_lineage("A")
            gen_b_food = world.food_eaten_by_lineage("B")
            a_total_food += gen_a_food
            b_total_food += gen_b_food

            a_creatures = [c for c in world.creatures if c.lineage == "A"]
            b_creatures = [c for c in world.creatures if c.lineage == "B"]

            if a_alive:
                a_survived_gens = generation
            if b_alive:
                b_survived_gens = generation

            fitnesses = [evaluate_fitness(c, self.settings) for c in world.creatures]
            best = max(fitnesses) if fitnesses else 0
            avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0

            self.renderer.draw_generation_stats(world, best, avg, mode="convergence")
            pygame.display.flip()
            pygame.time.wait(1500)

            top_n = max(1, self.settings.top_performers_count // 2)
            top_a = select_top(a_creatures, min(top_n, len(a_creatures)), self.settings)
            top_b = select_top(b_creatures, min(top_n, len(b_creatures)), self.settings)

            if not top_a:
                a_alive = False
            if not top_b:
                b_alive = False

            if not a_alive and not b_alive:
                break
            elif not a_alive:
                break
            elif not b_alive:
                break

            generation += 1
            new_a = create_next_generation(
                top_a, CREATURES_PER_LINEAGE,
                mutation_rate=self.settings.mutation_rate,
                mutation_strength=self.settings.mutation_strength,
                crossover_rate=self.settings.crossover_rate,
                min_parents=self.settings.min_population,
                weight_clamp=self.settings.weight_clamp,
                trait_mutation_range=self.settings.trait_mutation_range,
            )
            new_b = create_next_generation(
                top_b, CREATURES_PER_LINEAGE,
                mutation_rate=self.settings.mutation_rate,
                mutation_strength=self.settings.mutation_strength,
                crossover_rate=self.settings.crossover_rate,
                min_parents=self.settings.min_population,
                weight_clamp=self.settings.weight_clamp,
                trait_mutation_range=self.settings.trait_mutation_range,
            )

            all_creatures = []
            for dna in new_a:
                x = random.uniform(50, self.settings.world_width - 50)
                y = random.uniform(50, self.settings.world_height - 50)
                all_creatures.append(Creature(dna, x, y, lineage="A"))
            for dna in new_b:
                x = random.uniform(50, self.settings.world_width - 50)
                y = random.uniform(50, self.settings.world_height - 50)
                all_creatures.append(Creature(dna, x, y, lineage="B"))

            world = World(all_creatures, settings=self.settings)
            self._active_world = world
            world.generation = generation
            self.renderer.reset_tracking()

        if a_total_food > b_total_food:
            winner = "A"
        elif b_total_food > a_total_food:
            winner = "B"
        else:
            winner = "tie"

        self.menu.show_convergence_results(
            winner, a_total_food, b_total_food, a_survived_gens, b_survived_gens
        )

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
            world.predators = loaded_snapshot["predators"]
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
            world._predator_respawn_timer = timers.get("predator_respawn_timer", 0.0)
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
            pop = self.settings.freeplay_initial_population
            dna_list = [DNA.random() for _ in range(pop)]
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
                    self.settings_panel.toggle()
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
                                "freeplay", self.settings
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
                                # Also save species file for convergence use
                                alive = [c for c in world.creatures if c.alive]
                                if alive:
                                    top_dna = [c.dna for c in sorted(
                                        alive, key=lambda c: c.food_eaten, reverse=True,
                                    )[:self.settings.top_performers_count]]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filepath = f"species/freeplay_{timestamp}.json"
                                    save_species(top_dna, filepath, generation=0)
                                return
                            elif pause_result in ("main_menu", "restart"):
                                if pause_result == "restart":
                                    self.tools.reset()
                                    pop = self.settings.freeplay_initial_population
                                    dna_list = [DNA.random() for _ in range(pop)]
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
                    for _ in range(self.fast_forward_multiplier):
                        world.update(dt)
                        world.check_breeding()
                else:
                    world.update(dt)
                    world.check_breeding()

                self._freeplay_elapsed += dt

                # Periodic cleanup and stats
                self._freeplay_stats_timer += dt
                self._freeplay_history_timer += dt
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
                    alive_creatures = [c for c in world.creatures if c.alive]
                    n_alive = len(alive_creatures)
                    self._freeplay_history.append({
                        "time": self._freeplay_elapsed,
                        "population": n_alive,
                        "births": world.total_births,
                        "deaths": world.total_deaths,
                        "births_per_min": self._freeplay_births_per_min,
                        "deaths_per_min": self._freeplay_deaths_per_min,
                        "herbivores": sum(1 for c in alive_creatures if c.dna.diet == 0),
                        "carnivores": sum(1 for c in alive_creatures if c.dna.diet == 1),
                        "scavengers": sum(1 for c in alive_creatures if c.dna.diet == 2),
                        "avg_gen": (
                            sum(c.generation for c in alive_creatures) / n_alive
                            if n_alive else 0
                        ),
                    })
                    # Keep last 360 snapshots (~1 hour at 10s intervals)
                    if len(self._freeplay_history) > 360:
                        self._freeplay_history = self._freeplay_history[-360:]
                    self._freeplay_history_timer = 0.0

                alive = world.alive_count()
                if alive > self._freeplay_peak_pop:
                    self._freeplay_peak_pop = alive

            # Check for extinction
            if world.alive_count() == 0 and not self.paused:
                pop = self.settings.freeplay_initial_population
                # Carry over DNA from the best performers of the extinct population
                top_dna = select_top(
                    world.creatures,
                    min(self.settings.top_performers_count, len(world.creatures)),
                    self.settings,
                )
                if top_dna:
                    dna_list = create_next_generation(
                        top_dna,
                        population_size=pop,
                        mutation_rate=self.settings.mutation_rate,
                        mutation_strength=self.settings.mutation_strength,
                        crossover_rate=self.settings.crossover_rate,
                        min_parents=self.settings.min_population,
                        weight_clamp=self.settings.weight_clamp,
                        trait_mutation_range=self.settings.trait_mutation_range,
                    )
                else:
                    dna_list = [DNA.random() for _ in range(pop)]
                world = self._create_world(dna_list)
                world.freeplay = True
                world.generation = 0
                self.renderer.reset_tracking()

            # Render
            self.renderer.draw(
                world, "freeplay", self.paused,
                tools=self.tools,
                show_toolbar=True,
                fast_forward=self.fast_forward_multiplier if self.fast_forward else 0,
                debug=self.debug_mode,
            )
            self.renderer.draw_creature_stats(world, "freeplay")
            if self.show_evolution_panel:
                self.renderer.draw_evolution_panel(
                    world, "freeplay", self._freeplay_history,
                )
            if self.tools.active_tool != "none":
                self.renderer.draw_tool_cursor(self.tools)
            self.settings_panel.draw(self.screen, self.settings, dt)
            pygame.display.flip()

    # ── Generation Loop ──────────────────────────────────────

    def _run_generation(self, world: World, mode: str = "isolation") -> str:
        """
        Run a single generation until it ends or the user interrupts.

        Returns:
            "done", "main_menu", "save_quit", or "restart".
        """
        self.paused = False
        show_toolbar = mode == "isolation"

        while not world.is_generation_over():
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return "main_menu"

                if event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                    continue

                # S key toggles the settings panel
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.settings_panel.toggle()
                    continue

                # Route events to settings panel first (it consumes them when visible)
                if self.settings_panel.visible:
                    self.settings = self.settings_panel.handle_event(event, self.settings)
                    # If the click was inside the panel, skip normal handling
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
                                mode, self.settings
                            )
                            if pause_result == "resume":
                                self.paused = False
                            elif pause_result in ("save_quit", "main_menu", "restart"):
                                return pause_result
                    # Tool hotkeys (1-6)
                    elif mode == "isolation" and pygame.K_1 <= event.key <= pygame.K_6:
                        tool_idx = event.key - pygame.K_1
                        if tool_idx < len(TOOL_LIST):
                            self.tools.select_tool(TOOL_LIST[tool_idx])

                # Mouse events for player tools (isolation mode only)
                # Right-click to inspect creature (any mode)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    mx, my = event.pos
                    if not self.settings_panel.consumes_click(mx, my):
                        wmx, wmy = self._screen_to_world(mx, my)
                        if not self.renderer.try_select_creature(world, wmx, wmy):
                            self.renderer.deselect_creature()

                if mode == "isolation":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        if self.settings_panel.consumes_click(mx, my):
                            continue
                        # Check if click is on toolbar area (top-right)
                        toolbar_x = config.WINDOW_WIDTH - 380
                        if my < 75 and mx > toolbar_x:
                            # Toolbar click — select tool
                            btn_w = 58
                            gap = 4
                            for i, tool in enumerate(TOOL_LIST):
                                tx = toolbar_x + i * (btn_w + gap)
                                if tx <= mx <= tx + btn_w:
                                    self.tools.select_tool(tool)
                                    break
                        elif self.tools.active_tool == "none":
                            # No tool active — try to select a creature
                            wmx, wmy = self._screen_to_world(mx, my)
                            if not self.renderer.try_select_creature(world, wmx, wmy):
                                self.renderer.deselect_creature()
                        else:
                            # World click — use active tool
                            wmx, wmy = self._screen_to_world(mx, my)
                            food_positions = self.tools.on_mouse_down(wmx, wmy)
                            for fx, fy in food_positions:
                                world.add_food_at(fx, fy)

                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        mx, my = event.pos
                        wmx, wmy = self._screen_to_world(mx, my)
                        self.tools.on_mouse_up(wmx, wmy)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Convergence mode — left-click to inspect
                    mx, my = event.pos
                    if not self.settings_panel.consumes_click(mx, my):
                        wmx, wmy = self._screen_to_world(mx, my)
                        if not self.renderer.try_select_creature(world, wmx, wmy):
                            self.renderer.deselect_creature()

            # Update settings panel dragging
            self.settings = self.settings_panel.update_dragging(self.settings)

            # Update simulation (skip if paused)
            if not self.paused:
                if self.fast_forward:
                    # Run multiple simulation steps per frame for actual speedup
                    for _ in range(self.fast_forward_multiplier):
                        world.update(dt)
                        if world.is_generation_over():
                            break
                else:
                    world.update(dt)

            # Render every frame to keep UI smooth
            self.renderer.draw(
                world, mode, self.paused,
                tools=self.tools if mode == "isolation" else None,
                show_toolbar=show_toolbar,
                fast_forward=self.fast_forward_multiplier if self.fast_forward else 0,
                debug=self.debug_mode,
            )
            self.renderer.draw_creature_stats(world, mode)
            if self.show_evolution_panel:
                self.renderer.draw_evolution_panel(
                    world, mode, self.generation_history,
                )
            if mode == "isolation" and self.tools.active_tool != "none":
                self.renderer.draw_tool_cursor(self.tools)
            self.settings_panel.draw(self.screen, self.settings, dt)
            pygame.display.flip()

        return "done"

    # ── World Creation Helpers ───────────────────────────────

    def _create_world(self, dna_list: list[DNA], lineage: str = "") -> World:
        """Create a World with creatures from a list of DNA."""
        creatures = []
        for dna in dna_list:
            x = random.uniform(50, self.settings.world_width - 50)
            y = random.uniform(50, self.settings.world_height - 50)
            creatures.append(Creature(dna, x, y, lineage=lineage))
        world = World(creatures, settings=self.settings, tools=self.tools)
        self._active_world = world
        return world

    def _create_convergence_world(
        self, dna_a: list[DNA], dna_b: list[DNA],
    ) -> World:
        """Create a World with two lineages for convergence mode."""
        creatures = []

        for dna in dna_a[:CREATURES_PER_LINEAGE]:
            x = random.uniform(50, self.settings.world_width - 50)
            y = random.uniform(50, self.settings.world_height - 50)
            creatures.append(Creature(dna, x, y, lineage="A"))

        for dna in dna_b[:CREATURES_PER_LINEAGE]:
            x = random.uniform(50, self.settings.world_width - 50)
            y = random.uniform(50, self.settings.world_height - 50)
            creatures.append(Creature(dna, x, y, lineage="B"))

        while len([c for c in creatures if c.lineage == "A"]) < CREATURES_PER_LINEAGE:
            src = random.choice([c for c in creatures if c.lineage == "A"])
            x = random.uniform(50, self.settings.world_width - 50)
            y = random.uniform(50, self.settings.world_height - 50)
            creatures.append(Creature(src.dna, x, y, lineage="A"))

        while len([c for c in creatures if c.lineage == "B"]) < CREATURES_PER_LINEAGE:
            src = random.choice([c for c in creatures if c.lineage == "B"])
            x = random.uniform(50, self.settings.world_width - 50)
            y = random.uniform(50, self.settings.world_height - 50)
            creatures.append(Creature(src.dna, x, y, lineage="B"))

        world = World(creatures, settings=self.settings)
        self._active_world = world
        return world
