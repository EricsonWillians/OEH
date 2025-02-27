import math
from typing import Final

# === Simulation Constants ===

# Camera settings - set for an immersive first-person view
CAMERA_POSITION: Final[tuple[float, float]] = (4e6, 0)  # 4,000,000 cm, or 40 km away
VIEW_ANGLE: Final[float] = math.atan2(0.0 - 0.0, 0.0 - 4e6)  # Automatically points toward (0,0)
FIELD_OF_VIEW: Final[float] = math.radians(90)  # 90° FOV for a wide, immersive view

# Black hole parameters
BLACK_HOLE_MASS: Final[float] = 1.0  # In our simulation's geometric units, mass=1 gives event horizon ~2
EVENT_HORIZON_RADIUS: Final[float] = 2.0 * BLACK_HOLE_MASS  # Schwarzschild radius

# Integration parameters tuned for quality
TIME_STEP: Final[float] = 0.005  # Smaller time step for smoother ray integration
NUM_STEPS: Final[int] = 1000     # Increased number of steps for higher accuracy
ESCAPE_RADIUS: Final[float] = 100.0  # Capture extended details in the background

# === Rendering Parameters ===

# Window settings - increased resolution for better visual fidelity
WINDOW_WIDTH: Final[int] = 640
WINDOW_HEIGHT: Final[int] = 360
WINDOW_TITLE: Final[str] = "Open Events in the Horizon"

# OpenGL texture settings
TEXTURE_INTERNAL_FORMAT: Final[int] = 0x8227  # GL_RGB32F

# === Debug & Logging Settings ===

DEBUG_MODE: Final[bool] = False
LOG_LEVEL: Final[str] = "INFO"
