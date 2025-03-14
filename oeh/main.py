#!/usr/bin/env python3
import sys
import signal
import time
import os
import datetime
import argparse
import math
import logging

import glfw
import numpy as np
from PIL import Image
from OpenGL.GL import *

from oeh.config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    CAMERA_POSITION,
    BLACK_HOLE_MASS,
    FIELD_OF_VIEW,
    VIEW_ANGLE
)
from oeh.rendering.opengl_renderer import OpenGLRenderer
from oeh.simulation.raytracer import get_current_fps, run_simulation
from utils.logger import get_logger, setup_file_logging

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Simulation state
# -----------------------------------------------------------------------------
class SimulationState:
    def __init__(self) -> None:
        # Core simulation parameters
        self.camera_position = CAMERA_POSITION
        self.black_hole_mass = BLACK_HOLE_MASS
        self.fov = FIELD_OF_VIEW
        self.b_field_exponent = 1.25
        self.integrator_choice = 2
        self.view_angle = VIEW_ANGLE

        # Post-processing parameters
        self.pp_exposure = 1.5
        self.pp_contrast = 1.2
        self.pp_saturation = 1.4
        self.pp_gamma = 2.2
        self.pp_bloom = 0.3
        self.pp_vignette = True

        # Rendering & control state
        self.paused = False
        self.auto_rotate = True
        self.rotation_speed = 0.005
        self.rotation_angle = 0.0
        self.show_help = False
        self.show_info = True

    def reset(self) -> None:
        self.__init__()

sim_state = SimulationState()
renderer = None  # Global renderer reference

# -----------------------------------------------------------------------------
# Argument parsing and signal handling
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open Events in the Horizon - Black Hole Simulation"
    )
    parser.add_argument("--width", type=int, default=WINDOW_WIDTH, help="Window width")
    parser.add_argument("--height", type=int, default=WINDOW_HEIGHT, help="Window height")
    parser.add_argument("--mass", type=float, default=BLACK_HOLE_MASS, help="Black hole mass")
    parser.add_argument("--camera-x", type=float, default=CAMERA_POSITION[0], help="Camera X position")
    parser.add_argument("--camera-y", type=float, default=CAMERA_POSITION[1], help="Camera Y position")
    parser.add_argument("--fov", type=float, default=FIELD_OF_VIEW, help="Field of view (radians)")
    parser.add_argument("--b", type=float, default=1.25, help="Magnetic field exponent")
    parser.add_argument("--integrator", type=int, choices=[0, 1, 2], default=2,
                        help="Integrator: 0=Euler, 1=RK4, 2=Verlet")
    parser.add_argument("--log-file", type=str, help="Log to file")
    parser.add_argument("--no-vsync", action="store_true", help="Disable VSync")
    parser.add_argument("--cpu", action="store_true", help="Force CPU rendering (not implemented)")
    parser.add_argument("--force-color", action="store_true", help="Force a bright color output for debugging")
    return parser.parse_args()

def signal_handler(signum: int, frame: object) -> None:
    logger.info("Received signal %d, shutting down...", signum)
    if renderer:
        renderer.cleanup()
    glfw.terminate()
    sys.exit(0)

# -----------------------------------------------------------------------------
# GLFW callbacks
# -----------------------------------------------------------------------------

def key_callback(window, key, scancode, action, mods) -> None:
    global sim_state, renderer
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    now = time.time()
    if hasattr(renderer, 'last_key_time'):
        if key in renderer.last_key_time and (now - renderer.last_key_time[key]) < 0.1:
            return
        renderer.last_key_time[key] = now

    # Handle basic controls in the main file
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_SPACE:
        sim_state.paused = not sim_state.paused
        renderer.paused = sim_state.paused  # Sync with renderer
        logger.info("Simulation %s", "paused" if sim_state.paused else "resumed")
    elif key == glfw.KEY_H:
        sim_state.show_help = not sim_state.show_help
        renderer.show_help = sim_state.show_help  # Sync with renderer
    elif key == glfw.KEY_I:
        sim_state.show_info = not sim_state.show_info
        renderer.show_info = sim_state.show_info  # Sync with renderer
    elif key == glfw.KEY_R:
        sim_state.auto_rotate = not sim_state.auto_rotate
        renderer.auto_rotate = sim_state.auto_rotate  # Sync with renderer
        logger.info("Auto-rotation %s", "enabled" if sim_state.auto_rotate else "disabled")
    
    # Camera movement
    elif key == glfw.KEY_W:
        sim_state.camera_position = (sim_state.camera_position[0],
                                     sim_state.camera_position[1] + 0.5)
        # Fixed: Use the tuple directly
        renderer.camera_position = sim_state.camera_position
    elif key == glfw.KEY_S:
        sim_state.camera_position = (sim_state.camera_position[0],
                                     sim_state.camera_position[1] - 0.5)
        renderer.camera_position = sim_state.camera_position
    elif key == glfw.KEY_A:
        sim_state.camera_position = (sim_state.camera_position[0] - 0.5,
                                     sim_state.camera_position[1])
        renderer.camera_position = sim_state.camera_position
    elif key == glfw.KEY_D:
        sim_state.camera_position = (sim_state.camera_position[0] + 0.5,
                                     sim_state.camera_position[1])
        renderer.camera_position = sim_state.camera_position
    
    # Black hole parameters
    elif key == glfw.KEY_UP:
        sim_state.black_hole_mass += 0.5
        renderer.black_hole_mass = sim_state.black_hole_mass  # Sync with renderer
        logger.info("Black hole mass: %.2f Msun", sim_state.black_hole_mass)
    elif key == glfw.KEY_DOWN:
        sim_state.black_hole_mass = max(1.0, sim_state.black_hole_mass - 0.5)
        renderer.black_hole_mass = sim_state.black_hole_mass  # Sync with renderer
        logger.info("Black hole mass: %.2f Msun", sim_state.black_hole_mass)
    
    # Field of view
    elif key == glfw.KEY_LEFT:
        sim_state.fov = max(0.1, sim_state.fov - 0.05)
        renderer.fov = sim_state.fov  # Sync with renderer
        logger.info("FOV: %.2f radians", sim_state.fov)
    elif key == glfw.KEY_RIGHT:
        sim_state.fov += 0.05
        renderer.fov = sim_state.fov  # Sync with renderer
        logger.info("FOV: %.2f radians", sim_state.fov)
    
    # Magnetic field exponent
    elif key == glfw.KEY_B:
        sim_state.b_field_exponent = max(0.75, sim_state.b_field_exponent - 0.05)
        renderer.b_field_exponent = sim_state.b_field_exponent  # Sync with renderer
        logger.info("B-field exponent: %.2f", sim_state.b_field_exponent)
    elif key == glfw.KEY_N:
        sim_state.b_field_exponent = min(1.5, sim_state.b_field_exponent + 0.05)
        renderer.b_field_exponent = sim_state.b_field_exponent  # Sync with renderer
        logger.info("B-field exponent: %.2f", sim_state.b_field_exponent)
    
    # Integrator selection
    elif key == glfw.KEY_1:
        sim_state.integrator_choice = 0
        renderer.integrator_choice = sim_state.integrator_choice  # Sync with renderer
        logger.info("Integrator: Euler")
    elif key == glfw.KEY_2:
        sim_state.integrator_choice = 1
        renderer.integrator_choice = sim_state.integrator_choice  # Sync with renderer
        logger.info("Integrator: RK4")
    elif key == glfw.KEY_3:
        sim_state.integrator_choice = 2
        renderer.integrator_choice = sim_state.integrator_choice  # Sync with renderer
        logger.info("Integrator: Velocity Verlet")
    
    # Post-processing adjustments - handle them as in renderer
    elif key == glfw.KEY_E:
        sim_state.pp_exposure = max(0.1, sim_state.pp_exposure - 0.1)
        renderer.pp_exposure = sim_state.pp_exposure  # Sync with renderer
        renderer._update_post_processing_uniforms()
        logger.info("Exposure: %.1f", sim_state.pp_exposure)
    elif key == glfw.KEY_Q:
        sim_state.pp_exposure += 0.1
        renderer.pp_exposure = sim_state.pp_exposure  # Sync with renderer
        renderer._update_post_processing_uniforms()
        logger.info("Exposure: %.1f", sim_state.pp_exposure)
    elif key == glfw.KEY_C:
        sim_state.pp_contrast = max(0.5, sim_state.pp_contrast - 0.1)
        renderer.pp_contrast = sim_state.pp_contrast  # Sync with renderer
        renderer._update_post_processing_uniforms()
        logger.info("Contrast: %.1f", sim_state.pp_contrast)
    elif key == glfw.KEY_V:
        sim_state.pp_contrast += 0.1
        renderer.pp_contrast = sim_state.pp_contrast  # Sync with renderer
        renderer._update_post_processing_uniforms()
        logger.info("Contrast: %.1f", sim_state.pp_contrast)
    elif key == glfw.KEY_G:
        sim_state.pp_gamma = max(0.5, sim_state.pp_gamma - 0.1)
        renderer.pp_gamma = sim_state.pp_gamma  # Sync with renderer
        renderer._update_post_processing_uniforms()
        logger.info("Gamma: %.1f", sim_state.pp_gamma)
    elif key == glfw.KEY_T:
        sim_state.pp_gamma += 0.1
        renderer.pp_gamma = sim_state.pp_gamma  # Sync with renderer
        renderer._update_post_processing_uniforms()
        logger.info("Gamma: %.1f", sim_state.pp_gamma)
    elif key == glfw.KEY_F:
        sim_state.pp_vignette = not sim_state.pp_vignette
        renderer.pp_vignette = sim_state.pp_vignette  # Sync with renderer
        renderer._update_post_processing_uniforms()
        logger.info("Vignette: %s", "On" if sim_state.pp_vignette else "Off")
    
    # Screenshots
    elif key == glfw.KEY_P:
        save_screenshot()
    
    # Reset all parameters to defaults
    elif key == glfw.KEY_0:
        sim_state.reset()
        renderer._reset_parameters()  # Use renderer's reset
        logger.info("Parameters reset to defaults")

def mouse_button_callback(window, button, action, mods) -> None:
    global renderer
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        x, y = glfw.get_cursor_pos(window)
        if renderer and renderer.width > 0 and renderer.height > 0:
            x_norm = x / renderer.width * 2.0 - 1.0
            y_norm = 1.0 - y / renderer.height * 2.0
            logger.debug("Mouse clicked at (%.2f, %.2f)", x_norm, y_norm)

def scroll_callback(window, x_offset, y_offset) -> None:
    global sim_state
    zoom_speed = 0.05
    sim_state.fov = max(0.1, min(3.0, sim_state.fov - y_offset * zoom_speed))
    logger.debug("FOV adjusted to %.2f", sim_state.fov)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
# And also fix the update_renderer_postprocessing function:

def update_renderer_postprocessing() -> None:
    """Synchronizes post-processing parameters between simulation state and renderer."""
    global renderer, sim_state
    if renderer:
        # Update renderer with values from simulation state
        renderer.pp_exposure = sim_state.pp_exposure
        renderer.pp_contrast = sim_state.pp_contrast
        renderer.pp_saturation = sim_state.pp_saturation
        renderer.pp_gamma = sim_state.pp_gamma
        renderer.pp_bloom = sim_state.pp_bloom
        renderer.pp_vignette = sim_state.pp_vignette
        
        # Update the uniforms in the shader
        renderer._update_post_processing_uniforms()
        
        # Also update other renderer properties to ensure they're in sync
        renderer.camera_position = sim_state.camera_position  # Fixed: Pass the tuple directly
        renderer.black_hole_mass = sim_state.black_hole_mass
        renderer.fov = sim_state.fov
        renderer.b_field_exponent = sim_state.b_field_exponent
        renderer.integrator_choice = sim_state.integrator_choice
        renderer.paused = sim_state.paused
        renderer.auto_rotate = sim_state.auto_rotate
        renderer.show_help = sim_state.show_help
        renderer.show_info = sim_state.show_info

def update_camera() -> None:
    global sim_state
    if sim_state.auto_rotate:
        sim_state.rotation_angle += sim_state.rotation_speed
        radius = math.hypot(sim_state.camera_position[0], sim_state.camera_position[1])
        sim_state.camera_position = (
            radius * math.cos(sim_state.rotation_angle),
            radius * math.sin(sim_state.rotation_angle)
        )

def update_view_angle() -> None:
    global sim_state
    cam_x, cam_y = sim_state.camera_position
    if (cam_x != 0 or cam_y != 0) and not sim_state.auto_rotate:
        sim_state.view_angle = math.atan2(-cam_y, -cam_x)

def process_simulation() -> np.ndarray:
    """
    Returns the image data from the simulation or, if force_color is active,
    a dummy bright red image.
    """
    if args.force_color:
        # Create a dummy bright red image
        image = np.zeros((renderer.height, renderer.width, 3), dtype=np.float32)
        image[..., 0] = 1.0
    else:
        image = run_simulation(
            custom_camera_position=sim_state.camera_position,
            custom_black_hole_mass=sim_state.black_hole_mass,
            custom_fov=sim_state.fov,
            b_field_exponent=sim_state.b_field_exponent,
            integrator_choice=sim_state.integrator_choice
        )
    return image

def update_texture(image: np.ndarray) -> None:
    glBindTexture(GL_TEXTURE_2D, renderer.texture)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, renderer.width, renderer.height, GL_RGB, GL_FLOAT, image)

def save_screenshot() -> None:
    global renderer
    if renderer:
        renderer.take_screenshot()
    else:
        logger.error("Cannot take screenshot - renderer not initialized")

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main_loop() -> None:
    """Main rendering loop that lets the renderer handle everything."""
    while not glfw.window_should_close(renderer.window):
        # Process events - this will call the renderer's key_callback
        glfw.poll_events()
        
        # Let the renderer handle everything
        renderer.render_frame()
        
        # Swap buffers
        glfw.swap_buffers(renderer.window)
        
        # Update FPS display
        fps = get_current_fps()
        now = time.time()
        if now - renderer.last_frame_time > renderer.fps_update_interval:
            glfw.set_window_title(
                renderer.window, 
                f"{WINDOW_TITLE} - FPS: {fps:.2f} | "
                f"Mass: {renderer.black_hole_mass:.1f} | "
                f"FOV: {renderer.fov:.2f} | "
                f"B-field: {renderer.b_field_exponent:.2f}"
            )
            renderer.last_frame_time = now

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main() -> None:
    global renderer, args, sim_state
    args = parse_args()

    if args.log_file:
        setup_file_logging(args.log_file)

    # Setup signal handling for graceful exit.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Update simulation state from command-line args.
    sim_state.camera_position = (args.camera_x, args.camera_y)
    sim_state.black_hole_mass = args.mass
    sim_state.fov = args.fov
    sim_state.b_field_exponent = args.b
    sim_state.integrator_choice = args.integrator

    logger.info("Initializing simulation with parameters:")
    logger.info("  Window: %dx%d", args.width, args.height)
    logger.info("  Camera: (%.2f, %.2f)", sim_state.camera_position[0], sim_state.camera_position[1])
    logger.info("  BH Mass: %.2f Msun", sim_state.black_hole_mass)
    logger.info("  FOV: %.2f radians", sim_state.fov)
    logger.info("  B-field exponent: %.2f", sim_state.b_field_exponent)
    logger.info("  Integrator: %d", sim_state.integrator_choice)
    logger.info("  Force Color Test: %s", "Yes" if args.force_color else "No")

    # Initialize renderer.
    renderer = OpenGLRenderer(width=args.width, height=args.height,
                              title=WINDOW_TITLE, vsync=not args.no_vsync)
    renderer.force_color = args.force_color if hasattr(args, 'force_color') else False

    try:
        renderer.initialize()
    except Exception as e:
        logger.exception("Failed to initialize renderer: %s", e)
        sys.exit(1)

    # Important: Keep renderer's callback intact - DO NOT override
    # DO NOT call setup_callbacks() here
    
    # Transfer simulation state values to renderer
    sync_simulation_to_renderer()
    
    logger.info("Entering main loop.")
    main_loop()

    try:
        if renderer:
            renderer.cleanup()
    except Exception:
        pass
    glfw.terminate()
    logger.info("Simulation terminated.")

# Add a new function to sync state between simulation and renderer
def sync_simulation_to_renderer() -> None:
    """Synchronizes all state between simulation and renderer."""
    global renderer, sim_state
    if renderer:
        renderer.camera_position = sim_state.camera_position
        renderer.black_hole_mass = sim_state.black_hole_mass
        renderer.fov = sim_state.fov
        renderer.b_field_exponent = sim_state.b_field_exponent
        renderer.integrator_choice = sim_state.integrator_choice
        renderer.auto_rotate = sim_state.auto_rotate
        renderer.paused = sim_state.paused
        renderer.show_help = sim_state.show_help
        renderer.show_info = sim_state.show_info

        # Post-processing parameters
        renderer.pp_exposure = sim_state.pp_exposure
        renderer.pp_contrast = sim_state.pp_contrast
        renderer.pp_saturation = sim_state.pp_saturation
        renderer.pp_gamma = sim_state.pp_gamma
        renderer.pp_bloom = sim_state.pp_bloom
        renderer.pp_vignette = sim_state.pp_vignette
        
        # Update the shader uniforms
        renderer._update_post_processing_uniforms()

if __name__ == "__main__":
    main()
