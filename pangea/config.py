"""
Configuration constants for Pangea Evolution Simulator.
============================================================
Tweak these values to change the simulation behavior!
All tunable parameters are gathered here for easy access.
"""

# ── Window Settings ──────────────────────────────────────────
WINDOW_WIDTH = 1024          # Simulation window width in pixels
WINDOW_HEIGHT = 768          # Simulation window height in pixels
FPS = 60                     # Frames per second (rendering speed)

# ── Play Area (World) Dimensions ────────────────────────────
WORLD_WIDTH = WINDOW_WIDTH   # Play area matches window (resizes dynamically)
WORLD_HEIGHT = WINDOW_HEIGHT

# ── Population & Generations ─────────────────────────────────
POPULATION_SIZE = 50         # Number of creatures per generation
GENERATION_TIME_LIMIT = 30.0 # Seconds before a generation is forced to end
TOP_PERFORMERS_COUNT = 10    # How many top creatures survive to reproduce

# ── Genetic Budget ───────────────────────────────────────────
EVOLUTION_POINTS = 100       # Total points each creature distributes among traits
# Default trait allocation for the initial random generation:
DEFAULT_SPEED = 20           # Points allocated to speed
DEFAULT_SIZE = 20            # Points allocated to size
DEFAULT_VISION = 20          # Points allocated to vision range
DEFAULT_EFFICIENCY = 20      # Points allocated to energy efficiency
DEFAULT_LIFESPAN = 20        # Points allocated to lifespan

# ── Lifespan Scaling ────────────────────────────────────────
LIFESPAN_BASE = 10.0         # Minimum lifespan in seconds
LIFESPAN_SCALE = 0.5         # Additional seconds per trait point

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
FOOD_DECAY_TIME = 15.0       # Seconds before uneaten food disappears
FOOD_CLUSTER_SIZE = 4        # Number of food items spawned per cluster
FOOD_RESPAWN_CHANCE = 0.0    # Chance (0-1) to spawn a new food when one is eaten
FOOD_MIN = 0                 # Minimum food items on the map (0 = no minimum)
FOOD_MAX = 0                 # Maximum food items on the map (0 = no limit)
SEASON_LENGTH = 60.0         # Seconds per full seasonal cycle
SEASON_MIN_RATE = 0.2        # Multiplier at seasonal minimum
SEASON_ENABLED = True        # Whether seasonal food oscillation is active

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
NN_INPUT_SIZE = 12           # Inputs: food dist, food angle, wall dist, energy,
                             #         nearest creature dist, nearest creature angle,
                             #         own speed, predator dist, predator angle,
                             #         under_attack, biome_speed, biome_danger
NN_HIDDEN_SIZE = 8           # Hidden layer neurons
NN_OUTPUT_SIZE = 2           # Outputs: turn angle, thrust

# ── Fitness Weights ──────────────────────────────────────────
FITNESS_FOOD_WEIGHT = 10.0   # Points per food item eaten
FITNESS_TIME_WEIGHT = 0.1    # Points per second survived
FITNESS_ENERGY_WEIGHT = 0.05 # Points per unit of remaining energy
FITNESS_OFFSPRING_WEIGHT = 5.0  # Points per offspring produced


# ── Day/Night Cycle ────────────────────────────────────────
DAY_NIGHT_CYCLE_LENGTH = 90.0     # Seconds for one full day/night cycle
DAY_NIGHT_ENABLED = True          # Whether day/night cycle is active
NIGHT_VISION_MULTIPLIER = 0.3     # Vision multiplier at darkest point (0.0–1.0)

# ── Hazards / Obstacles ─────────────────────────────────────
HAZARD_COUNT = 3               # Number of hazard zones per generation
HAZARD_MIN_RADIUS = 30.0       # Minimum hazard zone radius in pixels
HAZARD_MAX_RADIUS = 60.0       # Maximum hazard zone radius in pixels
HAZARD_DAMAGE = 2.0            # Energy drained per second at the hazard center

# ── Biomes / Terrain ────────────────────────────────────────
BIOMES_ENABLED = True        # Whether biome regions are generated
BIOME_COUNT = 4              # Number of biome regions generated per world
BIOME_MIN_RADIUS = 80.0      # Minimum biome circle radius in pixels
BIOME_MAX_RADIUS = 150.0     # Maximum biome circle radius in pixels
BIOME_SPEED_MULTIPLIERS = {  # Movement speed multipliers by biome type
    "normal": 1.0,
    "water": 0.5,
    "road": 1.5,
    "forest": 0.7,
    "desert": 1.3,
    "swamp": 0.4,
    "mountain": 0.3,
}

# Biome special effects
BIOME_ENERGY_DRAIN = {       # Extra energy drain per second inside biome
    "desert": 1.0,
    "swamp": 0.5,
}
BIOME_FOOD_MULTIPLIER = {    # Food spawn rate multiplier inside biome
    "forest": 2.0,
}
BIOME_PREDATOR_BLOCKED = {"mountain"}  # Predators cannot enter these biomes

# ── Territory ──────────────────────────────────────────────
TERRITORY_GRID_SIZE = 80           # Pixels per territory grid cell

# ── Colors (RGB) ─────────────────────────────────────────────
COLOR_BACKGROUND = (15, 15, 25)
COLOR_FOOD = (50, 205, 50)
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
SIZE_ARMOR_SCALE = 0.03          # Damage reduction per unit of effective_radius (3% per px)
COLOR_PREDATOR = (255, 50, 50)   # Red color for predator rendering
COLOR_HAZARD_LAVA = (220, 80, 30)
COLOR_HAZARD_COLD = (80, 150, 255)
COLOR_BIOME_WATER = (30, 60, 120)   # Water biome tint color
COLOR_BIOME_ROAD = (100, 90, 70)    # Road biome tint color
COLOR_BIOME_FOREST = (20, 80, 30)   # Forest biome tint color
COLOR_BIOME_DESERT = (140, 120, 50) # Desert biome tint color
COLOR_BIOME_SWAMP = (50, 70, 40)    # Swamp biome tint color
COLOR_BIOME_MOUNTAIN = (100, 100, 110)  # Mountain biome tint color

# ── Creature Diets (legacy — kept for backward-compatible save loading) ──
DIET_HERBIVORE = 0               # Eats plants, bonus food energy
DIET_CARNIVORE = 1               # Attacks creatures, reduced plant food
DIET_SCAVENGER = 2               # Normal food + energy from nearby deaths
DIET_NAMES = {0: "herbivore", 1: "carnivore", 2: "scavenger"}

HERBIVORE_FOOD_BONUS = 1.5      # Food energy multiplier for herbivores
CARNIVORE_FOOD_PENALTY = 0.5    # Food energy multiplier for carnivores
CARNIVORE_ATTACK_RANGE = 2.0    # Attack range as multiplier of own radius
CARNIVORE_ATTACK_DAMAGE = 3.0   # Energy drained from victim per second
CARNIVORE_ENERGY_STEAL = 0.5    # Fraction of damage converted to own energy
SCAVENGER_FOOD_PENALTY = 0.7    # Food energy multiplier for scavengers
SCAVENGER_DEATH_RADIUS = 60.0   # Range to detect nearby deaths
SCAVENGER_DEATH_ENERGY = 8.0    # Energy gained per nearby death
CORPSE_ENERGY = 15.0            # Energy in a corpse food item
CORPSE_DECAY_TIME = 10.0        # Seconds before a corpse disappears
CORPSE_RADIUS = 5.0             # Radius of corpse food items

# ── Freeplay Mode ──────────────────────────────────────────
FREEPLAY_INITIAL_POPULATION = 40     # Starting population size
FREEPLAY_CARRYING_CAPACITY = 80      # Soft population cap (food slows above this)
FREEPLAY_HARD_CAP = 120              # Absolute max population (births blocked)
FREEPLAY_BREED_MIN_AGE = 5.0         # Min seconds alive before breeding
FREEPLAY_BREED_MIN_FOOD = 3          # Min food eaten to qualify for breeding
FREEPLAY_BREED_ENERGY_THRESHOLD = 0.6  # Energy must be above this fraction of BASE_ENERGY
FREEPLAY_BREED_COOLDOWN = 10.0       # Seconds between successive breedings
FREEPLAY_BREED_ENERGY_COST = 30.0    # Energy deducted from parent when breeding
FREEPLAY_CHILD_ENERGY = 60.0         # Starting energy for newborns
FREEPLAY_CHILD_SPAWN_RADIUS = 30.0   # Max pixel distance child spawns from parent
FREEPLAY_OVERCAPACITY_FOOD_PENALTY = 0.5  # Food spawn rate multiplier when above soft cap

COLOR_HERBIVORE = (80, 200, 80)     # Green tint for herbivores
COLOR_CARNIVORE = (200, 60, 60)     # Red tint for carnivores
COLOR_SCAVENGER = (180, 140, 50)    # Yellow/brown tint for scavengers

# ── Network / Multiplayer ─────────────────────────────────────
NET_SNAPSHOT_INTERVAL = 6           # Frames between snapshot broadcasts (60fps/6 = 10/sec)
NET_DEFAULT_RELAY = "ws://localhost:8765"
NET_DEFAULT_API = "http://localhost:8000"
NET_HEARTBEAT_INTERVAL = 15.0       # Seconds between heartbeat pings
