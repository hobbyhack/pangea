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
from pangea.save_load import load_species, save_species
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
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.menu = Menu(self.screen, on_toggle_fullscreen=self._toggle_fullscreen)

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.fast_forward_multiplier = 5  # sim steps per frame in fast mode
        self.debug_mode = False
        self.show_evolution_panel = False
        self.generation_history: list[dict] = []
        self.settings = SimSettings()
        self.tools = PlayerTools()
        self.settings_panel = SettingsPanel()

    def _toggle_fullscreen(self) -> None:
        """Toggle between windowed and fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN,
            )
        else:
            self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        # Update config so all modules see the new dimensions
        config.WINDOW_WIDTH = self.screen.get_width()
        config.WINDOW_HEIGHT = self.screen.get_height()
        self.renderer = Renderer(self.screen)
        self.menu = Menu(self.screen, on_toggle_fullscreen=self._toggle_fullscreen)

    # ── Main Entry Point ─────────────────────────────────────

    def run(self) -> None:
        """Main application loop — show menu, run selected mode, repeat."""
        while self.running:
            choice, self.settings = self.menu.show_main_menu(self.settings)

            if choice == "quit":
                self.running = False
            elif choice == "isolation":
                self._run_isolation()
            elif choice == "convergence":
                self._run_convergence()
            elif choice == "freeplay":
                self._run_freeplay()

        pygame.quit()

    # ── Isolation Mode ───────────────────────────────────────

    def _run_isolation(self) -> None:
        """Run the simulation in isolation mode (single-user evolution)."""
        self.tools = PlayerTools()
        pop = self.settings.population_size
        dna_list = [DNA.random() for _ in range(pop)]
        world = self._create_world(dna_list)
        generation = 1
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

    def _run_convergence(self) -> None:
        """Run the simulation in convergence mode (two competing lineages)."""
        result = self.menu.show_file_select()
        if result is None:
            return

        file_a, file_b = result
        dna_a, _ = load_species(file_a)
        dna_b, _ = load_species(file_b)

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
            elif result == "restart":
                dna_a, _ = load_species(file_a)
                dna_b, _ = load_species(file_b)
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
            new_a = create_next_generation(top_a, CREATURES_PER_LINEAGE)
            new_b = create_next_generation(top_b, CREATURES_PER_LINEAGE)

            all_creatures = []
            for dna in new_a:
                x = random.uniform(50, config.WINDOW_WIDTH - 50)
                y = random.uniform(50, config.WINDOW_HEIGHT - 50)
                all_creatures.append(Creature(dna, x, y, lineage="A"))
            for dna in new_b:
                x = random.uniform(50, config.WINDOW_WIDTH - 50)
                y = random.uniform(50, config.WINDOW_HEIGHT - 50)
                all_creatures.append(Creature(dna, x, y, lineage="B"))

            world = World(all_creatures, settings=self.settings)
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

    def _run_freeplay(self) -> None:
        """Run continuous breeding mode — no generations, creatures breed in real time."""
        self.tools = PlayerTools()
        pop = self.settings.freeplay_initial_population
        dna_list = [DNA.random() for _ in range(pop)]
        world = self._create_world(dna_list)
        world.freeplay = True
        world.generation = 0

        # Track rolling stats for the HUD
        self._freeplay_elapsed = 0.0
        self._freeplay_last_births = 0
        self._freeplay_last_deaths = 0
        self._freeplay_births_per_min = 0.0
        self._freeplay_deaths_per_min = 0.0
        self._freeplay_stats_timer = 0.0
        self._freeplay_peak_pop = pop
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
                                # Save all living creatures' DNA
                                alive = [c for c in world.creatures if c.alive]
                                if alive:
                                    top_dna = [c.dna for c in sorted(
                                        alive, key=lambda c: c.food_eaten, reverse=True,
                                    )[:self.settings.top_performers_count]]
                                    from datetime import datetime as _dt
                                    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
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
                                    self._freeplay_peak_pop = pop
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
                        if not self.renderer.try_select_creature(world, mx, my):
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
                        if not self.renderer.try_select_creature(world, mx, my):
                            self.renderer.deselect_creature()
                    else:
                        food_positions = self.tools.on_mouse_down(mx, my)
                        for fx, fy in food_positions:
                            world.add_food_at(fx, fy)

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    mx, my = event.pos
                    self.tools.on_mouse_up(mx, my)

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
                # Auto-respawn with random creatures
                pop = self.settings.freeplay_initial_population
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
            )
            if self.debug_mode:
                self.renderer.draw_debug(world)
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
                        if not self.renderer.try_select_creature(world, mx, my):
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
                            if not self.renderer.try_select_creature(world, mx, my):
                                self.renderer.deselect_creature()
                        else:
                            # World click — use active tool
                            food_positions = self.tools.on_mouse_down(mx, my)
                            for fx, fy in food_positions:
                                world.add_food_at(fx, fy)

                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        mx, my = event.pos
                        self.tools.on_mouse_up(mx, my)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Convergence mode — left-click to inspect
                    mx, my = event.pos
                    if not self.settings_panel.consumes_click(mx, my):
                        if not self.renderer.try_select_creature(world, mx, my):
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
            )
            if self.debug_mode:
                self.renderer.draw_debug(world)
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
            x = random.uniform(50, config.WINDOW_WIDTH - 50)
            y = random.uniform(50, config.WINDOW_HEIGHT - 50)
            creatures.append(Creature(dna, x, y, lineage=lineage))
        return World(creatures, settings=self.settings, tools=self.tools)

    def _create_convergence_world(
        self, dna_a: list[DNA], dna_b: list[DNA],
    ) -> World:
        """Create a World with two lineages for convergence mode."""
        creatures = []

        for dna in dna_a[:CREATURES_PER_LINEAGE]:
            x = random.uniform(50, config.WINDOW_WIDTH - 50)
            y = random.uniform(50, config.WINDOW_HEIGHT - 50)
            creatures.append(Creature(dna, x, y, lineage="A"))

        for dna in dna_b[:CREATURES_PER_LINEAGE]:
            x = random.uniform(50, config.WINDOW_WIDTH - 50)
            y = random.uniform(50, config.WINDOW_HEIGHT - 50)
            creatures.append(Creature(dna, x, y, lineage="B"))

        while len([c for c in creatures if c.lineage == "A"]) < CREATURES_PER_LINEAGE:
            src = random.choice([c for c in creatures if c.lineage == "A"])
            x = random.uniform(50, config.WINDOW_WIDTH - 50)
            y = random.uniform(50, config.WINDOW_HEIGHT - 50)
            creatures.append(Creature(src.dna, x, y, lineage="A"))

        while len([c for c in creatures if c.lineage == "B"]) < CREATURES_PER_LINEAGE:
            src = random.choice([c for c in creatures if c.lineage == "B"])
            x = random.uniform(50, config.WINDOW_WIDTH - 50)
            y = random.uniform(50, config.WINDOW_HEIGHT - 50)
            creatures.append(Creature(src.dna, x, y, lineage="B"))

        return World(creatures, settings=self.settings)
