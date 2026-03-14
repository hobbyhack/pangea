"""
Simulation -- the main game loop tying everything together.
============================================================
Initializes pygame, runs the menu, dispatches to the correct mode
(Isolation or Convergence), and manages the generation cycle.

Controls:
    SPACE  -> Pause / unpause
    F      -> Toggle fast-forward (skip rendering, run at max speed)
    D      -> Toggle debug overlay (vision ranges, energy bars)
    1-6    -> Select player tool (Isolation mode)
    ESC    -> Pause menu
    Left-click -> Use active tool
"""

from __future__ import annotations

import random
from datetime import datetime

import pygame

from pangea.config import (
    CREATURES_PER_LINEAGE,
    FPS,
    POPULATION_SIZE,
    TOP_PERFORMERS_COUNT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
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
from pangea.tools import TOOL_LIST, PlayerTools
from pangea.world import World


class Simulation:
    """Main simulation controller."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Pangea - Evolution Simulator")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.menu = Menu(self.screen)

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.debug_mode = False
        self.settings = SimSettings()
        self.tools = PlayerTools()

    # -- Main Entry Point -----------------------------------------------------

    def run(self) -> None:
        """Main application loop -- show menu, run selected mode, repeat."""
        while self.running:
            choice, self.settings = self.menu.show_main_menu(self.settings)

            if choice == "quit":
                self.running = False
            elif choice == "isolation":
                self._run_isolation()
            elif choice == "convergence":
                self._run_convergence()

        pygame.quit()

    # -- Isolation Mode -------------------------------------------------------

    def _run_isolation(self) -> None:
        """Run the simulation in isolation mode (single-user evolution)."""
        self.tools = PlayerTools()
        pop = self.settings.population_size
        dna_list = [DNA.random() for _ in range(pop)]
        world = self._create_world(dna_list)
        generation = 1
        world.generation = generation

        while self.running:
            result = self._run_generation(world, mode="isolation")

            if result == "main_menu":
                return
            elif result == "save_quit":
                top_dna = select_top(world.creatures, self.settings.top_performers_count)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"species/species_{timestamp}.json"
                save_species(top_dna, filepath, generation=generation)
                return
            elif result == "restart":
                self.tools.reset()
                pop = self.settings.population_size
                dna_list = [DNA.random() for _ in range(pop)]
                world = self._create_world(dna_list)
                generation = 1
                world.generation = generation
                self.renderer.reset_tracking()
                continue

            # Evolve to next generation
            top_dna = select_top(world.creatures, self.settings.top_performers_count)

            fitnesses = [evaluate_fitness(c) for c in world.creatures]
            best = max(fitnesses)
            avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0

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
                crossover=bool(self.settings.crossover_enabled),
            )
            world = self._create_world(dna_list)
            world.generation = generation
            self.renderer.reset_tracking()

    # -- Convergence Mode -----------------------------------------------------

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

            fitnesses = [evaluate_fitness(c) for c in world.creatures]
            best = max(fitnesses) if fitnesses else 0
            avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0

            self.renderer.draw_generation_stats(world, best, avg, mode="convergence")
            pygame.display.flip()
            pygame.time.wait(1500)

            top_n = max(1, self.settings.top_performers_count // 2)
            top_a = select_top(a_creatures, min(top_n, len(a_creatures)))
            top_b = select_top(b_creatures, min(top_n, len(b_creatures)))

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
                top_a,
                CREATURES_PER_LINEAGE,
                crossover=bool(self.settings.crossover_enabled),
            )
            new_b = create_next_generation(
                top_b,
                CREATURES_PER_LINEAGE,
                crossover=bool(self.settings.crossover_enabled),
            )

            all_creatures = []
            for dna in new_a:
                x = random.uniform(50, WINDOW_WIDTH - 50)
                y = random.uniform(50, WINDOW_HEIGHT - 50)
                all_creatures.append(Creature(dna, x, y, lineage="A"))
            for dna in new_b:
                x = random.uniform(50, WINDOW_WIDTH - 50)
                y = random.uniform(50, WINDOW_HEIGHT - 50)
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

    # -- Generation Loop ------------------------------------------------------

    def _run_generation(self, world: World, mode: str = "isolation") -> str:
        """
        Run a single generation until it ends or the user interrupts.

        Returns:
            "done", "main_menu", "save_quit", or "restart".
        """
        self.paused = False
        show_toolbar = mode == "isolation"

        while not world.is_generation_over():
            dt = self.clock.tick(FPS if not self.fast_forward else 0) / 1000.0
            dt = min(dt, 0.05)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return "main_menu"

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.fast_forward = not self.fast_forward
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_ESCAPE:
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
                if mode == "isolation":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        # Check if click is on toolbar area (top-right)
                        toolbar_x = WINDOW_WIDTH - 380
                        if my < 75 and mx > toolbar_x:
                            # Toolbar click -- select tool
                            btn_w = 58
                            gap = 4
                            for i, tool in enumerate(TOOL_LIST):
                                tx = toolbar_x + i * (btn_w + gap)
                                if tx <= mx <= tx + btn_w:
                                    self.tools.select_tool(tool)
                                    break
                        else:
                            # World click -- use active tool
                            food_positions = self.tools.on_mouse_down(mx, my)
                            for fx, fy in food_positions:
                                world.add_food_at(fx, fy)

                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        mx, my = event.pos
                        self.tools.on_mouse_up(mx, my)

            # Update simulation (skip if paused)
            if not self.paused:
                world.update(dt)

            # Render
            if not self.fast_forward:
                self.renderer.draw(
                    world, mode, self.paused,
                    tools=self.tools if mode == "isolation" else None,
                    show_toolbar=show_toolbar,
                )
                if self.debug_mode:
                    self.renderer.draw_debug(world)
                if mode == "isolation" and self.tools.active_tool != "none":
                    self.renderer.draw_tool_cursor(self.tools)
                pygame.display.flip()
            else:
                if int(world.elapsed_time * 10) % 5 == 0:
                    self.renderer.draw(
                        world, mode, self.paused,
                        tools=self.tools if mode == "isolation" else None,
                        show_toolbar=show_toolbar,
                    )
                    pygame.display.flip()

        return "done"

    # -- World Creation Helpers -----------------------------------------------

    def _create_world(self, dna_list: list[DNA], lineage: str = "") -> World:
        """Create a World with creatures from a list of DNA."""
        creatures = []
        for dna in dna_list:
            x = random.uniform(50, WINDOW_WIDTH - 50)
            y = random.uniform(50, WINDOW_HEIGHT - 50)
            creatures.append(Creature(dna, x, y, lineage=lineage))
        return World(creatures, settings=self.settings, tools=self.tools)

    def _create_convergence_world(
        self, dna_a: list[DNA], dna_b: list[DNA],
    ) -> World:
        """Create a World with two lineages for convergence mode."""
        creatures = []

        for dna in dna_a[:CREATURES_PER_LINEAGE]:
            x = random.uniform(50, WINDOW_WIDTH - 50)
            y = random.uniform(50, WINDOW_HEIGHT - 50)
            creatures.append(Creature(dna, x, y, lineage="A"))

        for dna in dna_b[:CREATURES_PER_LINEAGE]:
            x = random.uniform(50, WINDOW_WIDTH - 50)
            y = random.uniform(50, WINDOW_HEIGHT - 50)
            creatures.append(Creature(dna, x, y, lineage="B"))

        while len([c for c in creatures if c.lineage == "A"]) < CREATURES_PER_LINEAGE:
            src = random.choice([c for c in creatures if c.lineage == "A"])
            x = random.uniform(50, WINDOW_WIDTH - 50)
            y = random.uniform(50, WINDOW_HEIGHT - 50)
            creatures.append(Creature(src.dna, x, y, lineage="A"))

        while len([c for c in creatures if c.lineage == "B"]) < CREATURES_PER_LINEAGE:
            src = random.choice([c for c in creatures if c.lineage == "B"])
            x = random.uniform(50, WINDOW_WIDTH - 50)
            y = random.uniform(50, WINDOW_HEIGHT - 50)
            creatures.append(Creature(src.dna, x, y, lineage="B"))

        return World(creatures, settings=self.settings)
