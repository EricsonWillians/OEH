"""
oeh/config.py

Optimized configuration for GPU-accelerated black hole simulation in Pretty Mode.
This configuration prioritizes visual quality and smooth ray integration
over raw performance.
"""

import math
from typing import Final

# === Simulation Constants ===

# Camera settings - repositioned for a balanced, dramatic view
CAMERA_POSITION: Final[tuple[float, float]] = (25.0, 10.0)  # Further back for a wider view
VIEW_ANGLE: Final[float] = math.atan2(-10.0, -25.0)  # Adjusted to point toward the black hole
FIELD_OF_VIEW: Final[float] = math.radians(70)  # Slightly narrower FOV for improved detail

# Black hole parameters
BLACK_HOLE_MASS: Final[float] = 1.0  # Geometric units remain the same
EVENT_HORIZON_RADIUS: Final[float] = 2.0 * BLACK_HOLE_MASS  # Schwarzschild radius

# Integration parameters tuned for quality
TIME_STEP: Final[float] = 0.005  # Smaller time step for smoother ray integration
NUM_STEPS: Final[int] = 1000    # Increased number of steps for higher accuracy
ESCAPE_RADIUS: Final[float] = 100.0  # Larger escape radius to capture extended details

# === Rendering Parameters ===

# Window settings - increased resolution for better visual fidelity
WINDOW_WIDTH: Final[int] = 1280
WINDOW_HEIGHT: Final[int] = 720
WINDOW_TITLE: Final[str] = "Open Events in the Horizon - Pretty Mode"

# OpenGL texture settings
TEXTURE_INTERNAL_FORMAT: Final[int] = 0x8227  # GL_RGB32F

# === Debug & Logging Settings ===

DEBUG_MODE: Final[bool] = False  # Turn off debug mode for final visuals
LOG_LEVEL: Final[str] = "INFO"
