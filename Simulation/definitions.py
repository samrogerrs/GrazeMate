import numpy as np

# Constants for Boid behavior
SEPARATION_RADIUS = 50
ALIGNMENT_RADIUS = 100
COHESION_RADIUS = 100
SEPARATION_WEIGHT = 1.0
ALIGNMENT_WEIGHT = 0.5
COHESION_WEIGHT = 1
MAX_SPEED = 2.0
MAX_FORCE = 0.01

# DBSCAN parameters
DBSCAN_EPS = 180
DBSCAN_MIN_SAMPLES = 1


# Define the target position for the herd
target_pos = np.array([1600, 100])  # Goal for the herd to move toward

# Simulation parameters
SCREEN_WIDTH = 1700
SCREEN_HEIGHT = 1000
HERD_SIZE = 50
TERRAIN_RES = 50
FPS = 60

# LSM parameters
STRESS_RECOVERY_RATE = 0.02
STRESS_PRESSURE_RATE = 0.05
STRAGGLER_DISTANCE = 300
OPTIMAL_OFFSET = (100, 50)

# Terrain effects
TERRAIN_EFFECTS = {
    'grass': {'speed': 1.0, 'type': 'grass'},
    'water': {'speed': 0.5, 'type': 'water'},
    'mud': {'speed': 0.7, 'type': 'mud'},
    'rocky': {'speed': 0.6, 'type': 'rocky'}
}


# Colors
STRESS_COLORS = [(255, 255, 255), (255, 0, 0)]  # White to red gradient


# Constants for pressure/release and stress.
PRESSURE_PUSH_RADIUS = 100       # Within this distance, the drone applies pressure.
CATTLE_SPEED_THRESHOLD = 1.5     # Above this speed, cows are considered moving well.
WRANGLER_RELEASE_THRESHOLD = 1.0  # If target cow's speed is above this (with proper direction), drone stops.
STRESS_PROBABILITY = 0.1         # 10% chance per frame for a cow to get stressed if conditions are met.
STRESS_DURATION_RANGE = (5, 10)  # Frames (at 60 fps, ~1-2 seconds) for which a cow remains stressed.
COLOR_NORMAL = (0, 0, 0)
COLOR_STRESSED = (255, 165, 0)     # Orange
COLOR_FLY_HIGH = (255, 200, 200)   # Light pink