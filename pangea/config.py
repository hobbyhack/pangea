"""
Configuration constants for Pangea Evolution Simulator.
============================================================
Tweak these values to change the simulation behavior!
All tunable parameters are gathered here for easy access.
"""

# ── Window Settings ──────────────────────────────────────────
WINDOW_WIDTH = 1200          # Simulation window width in pixels
WINDOW_HEIGHT = 800          # Simulation window height in pixels
FPS = 60                     # Frames per second (rendering speed)

# ── Population & Generations ─────────────────────────────────
POPULATION_SIZE = 50         # Number of creatures per generation
GENERATION_TIME_LIMIT = 30.0 # Seconds before a generation is forced to end
TOP_PERFORMERS_COUNT = 10    # How many top creatures survive to reproduce

# ── Genetic Budget ───────────────────────────────────────────
EVOLUTION_POINTS = 100       # Total points each creature distributes among traits
# Default trait allocation for the initial random generation:
DEFAULT_SPEED = 25           # Points allocated to speed
DEFAULT_SIZE = 25            # Points allocated to size
DEFAULT_VISION = 25          # Points allocated to vision range
DEFAULT_EFFICIENCY = 25      # Points allocated to energy efficiency

# ── Mutation ─────────────────────────────────────────────────
MUTATION_RATE = 0.1          # Probability of mutating each weight (0.0 to 1.0)
MUTATION_STRENGTH = 0.3      # Standard deviation of Gaussian mutation noise
TRAIT_MUTATION_RANGE = 5     # Max +/- change to each trait per mutation

# ── World / Food ─────────────────────────────────────────────
FOOD_SPAWN_RATE = 0.5        # Average food items spawning per second
FOOD_ENERGY = 30.0           # Energy gained by eating one food item
FOOD_RADIUS = 4.0            # Radius of food items (for drawing & collision)
INITIAL_FOOD_COUNT = 30      # Food items present at the start of each generation
WORLD_WRAP = False           # True = toroidal wrap-around, False = bounded walls

# ── Creature Physics ─────────────────────────────────────────
BASE_ENERGY = 100.0          # Starting energy for each creature
ENERGY_COST_PER_THRUST = 0.1 # Base energy cost multiplier per frame of movement
SIZE_SPEED_PENALTY = 0.02    # Speed reduction per unit of effective radius

# Trait scaling formulas (how evolution points translate to physics):
#   effective_speed    = speed_points * SPEED_SCALE
#   effective_radius   = RADIUS_BASE + size_points * RADIUS_SCALE
#   effective_vision   = VISION_BASE + vision_points * VISION_SCALE
#   effective_efficiency = EFFICIENCY_BASE + efficiency_points * EFFICIENCY_SCALE
SPEED_SCALE = 0.1            # 25 pts → max velocity 2.5 px/frame
RADIUS_BASE = 3.0            # Minimum creature radius in pixels
RADIUS_SCALE = 0.15          # 25 pts → radius ~6.75 px
VISION_BASE = 50.0           # Minimum vision range in pixels
VISION_SCALE = 4.0           # 25 pts → 150 px vision range
EFFICIENCY_BASE = 0.5        # Minimum efficiency multiplier
EFFICIENCY_SCALE = 0.02      # 25 pts → efficiency 1.0 (neutral)

# ── Neural Network ───────────────────────────────────────────
NN_INPUT_SIZE = 4            # Inputs: food distance, food angle, wall distance, energy
NN_HIDDEN_SIZE = 8           # Hidden layer neurons
NN_OUTPUT_SIZE = 2           # Outputs: turn angle, thrust

# ── Fitness Weights ──────────────────────────────────────────
FITNESS_FOOD_WEIGHT = 10.0   # Points per food item eaten
FITNESS_TIME_WEIGHT = 0.1    # Points per second survived
FITNESS_ENERGY_WEIGHT = 0.05 # Points per unit of remaining energy

# ── Convergence Mode ─────────────────────────────────────────
CONVERGENCE_MAX_GENERATIONS = 50  # Max generations before declaring a winner
CREATURES_PER_LINEAGE = 25        # Creatures per lineage in convergence mode

# ── Day/Night Cycle ────────────────────────────────────────
DAY_NIGHT_CYCLE_LENGTH = 90.0     # Seconds for one full day/night cycle
NIGHT_VISION_MULTIPLIER = 0.3     # Vision multiplier at darkest point (0.0–1.0)

# ── Hazards / Obstacles ─────────────────────────────────────
HAZARD_COUNT = 3               # Number of hazard zones per generation
HAZARD_MIN_RADIUS = 30.0       # Minimum hazard zone radius in pixels
HAZARD_MAX_RADIUS = 60.0       # Maximum hazard zone radius in pixels
HAZARD_DAMAGE = 2.0            # Energy drained per second at the hazard center

# ── Colors (RGB) ─────────────────────────────────────────────
COLOR_BACKGROUND = (15, 15, 25)
COLOR_FOOD = (50, 205, 50)
COLOR_LINEAGE_A = (220, 60, 60)    # Red team
COLOR_LINEAGE_B = (60, 120, 220)   # Blue team
COLOR_HUD_TEXT = (220, 220, 220)
COLOR_MENU_BG = (20, 20, 35)
COLOR_BUTTON = (50, 60, 90)
COLOR_BUTTON_HOVER = (70, 85, 130)
COLOR_BUTTON_TEXT = (230, 230, 240)

# ── Predators ───────────────────────────────────────────────
PREDATOR_COUNT = 2               # Number of NPC predators in the world
PREDATOR_SPEED = 2.0             # Predator movement speed (px/frame)
PREDATOR_VISION = 150.0          # How far predators can detect creatures (px)
PREDATOR_DAMAGE = 5.0            # Energy drained per second on contact
PREDATOR_RADIUS = 8.0            # Predator body radius in pixels
COLOR_PREDATOR = (255, 50, 50)   # Red color for predator rendering
COLOR_HAZARD_LAVA = (220, 80, 30)
COLOR_HAZARD_COLD = (80, 150, 255)
