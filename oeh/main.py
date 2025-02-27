import sys
import signal
import time
import os
import datetime
import argparse
import glfw
import numpy as np
from typing import Tuple
import logging
import math
from PIL import Image

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
from oeh.custom_types import Vector2D
from OpenGL.GL import *

logger = get_logger(__name__)

# Global state for interactive controls
class SimulationState:
    """Encapsulates the mutable state of the simulation."""
    def __init__(self) -> None:
        self.camera_position: Vector2D = CAMERA_POSITION
        self.black_hole_mass: float = BLACK_HOLE_MASS
        self.fov: float = FIELD_OF_VIEW
        self.b_field_exponent: float = 1.25  # Optimized default for disk visibility
        self.integrator_choice: int = 2  # Default: Velocity Verlet
        self.view_angle: float = VIEW_ANGLE
        # Post-processing parameters
        self.pp_exposure: float = 1.5
        self.pp_contrast: float = 1.2
        self.pp_saturation: float = 1.4 
        self.pp_gamma: float = 2.2
        self.pp_bloom: float = 0.3
        self.pp_vignette: bool = True
        # Rendering state
        self.paused: bool = False
        self.auto_rotate: bool = True  # Enable auto-rotation by default
        self.rotation_speed: float = 0.005
        self.show_help: bool = False
        self.show_info: bool = True

# Create the global state
sim_state = SimulationState()
renderer = None  # Will be initialized in main

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Open Events in the Horizon - Black Hole Simulation')
    parser.add_argument('--width', type=int, default=WINDOW_WIDTH, help='Window width')
    parser.add_argument('--height', type=int, default=WINDOW_HEIGHT, help='Window height')
    parser.add_argument('--mass', type=float, default=BLACK_HOLE_MASS, help='Black hole mass')
    parser.add_argument('--camera-x', type=float, default=CAMERA_POSITION[0], help='Camera X position')
    parser.add_argument('--camera-y', type=float, default=CAMERA_POSITION[1], help='Camera Y position')
    parser.add_argument('--fov', type=float, default=FIELD_OF_VIEW, help='Field of view (radians)')
    parser.add_argument('--b', type=float, default=1.25, help='Magnetic field exponent')
    parser.add_argument('--integrator', type=int, default=2, choices=[0,1,2], 
                        help='Integrator: 0=Euler, 1=RK4, 2=Verlet')
    parser.add_argument('--log-file', type=str, help='Log to file')
    parser.add_argument('--no-vsync', action='store_true', help='Disable VSync')
    parser.add_argument('--cpu', action='store_true', help='Force CPU rendering')
    return parser.parse_args()

def signal_handler(signum: int, frame: object) -> None:
    """Handles OS signals."""
    logger.info("Received signal %d, shutting down...", signum)
    if renderer:
        renderer.cleanup()
    glfw.terminate()
    sys.exit(0)

def save_screenshot() -> None:
    """Save a screenshot using the renderer's functionality."""
    global renderer
    if renderer:
        renderer.take_screenshot()
    else:
        logger.error("Cannot take screenshot - renderer not initialized")

def key_callback(window, key, scancode, action, mods):
    """GLFW key callback."""
    global sim_state, renderer
    
    current_time = time.time()
    if hasattr(renderer, 'last_key_time'):
        if key in renderer.last_key_time and current_time - renderer.last_key_time[key] < 0.1:
            return
        renderer.last_key_time[key] = current_time
    
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            sim_state.paused = not sim_state.paused
            logger.info(f"Simulation {'paused' if sim_state.paused else 'resumed'}")
        elif key == glfw.KEY_H:
            sim_state.show_help = not sim_state.show_help
        elif key == glfw.KEY_I:
            sim_state.show_info = not sim_state.show_info
        elif key == glfw.KEY_R:
            sim_state.auto_rotate = not sim_state.auto_rotate
            logger.info(f"Auto-rotation {'enabled' if sim_state.auto_rotate else 'disabled'}")
        elif key == glfw.KEY_W:
            sim_state.camera_position = (
                sim_state.camera_position[0],
                sim_state.camera_position[1] + 0.5
            )
        elif key == glfw.KEY_S:
            sim_state.camera_position = (
                sim_state.camera_position[0],
                sim_state.camera_position[1] - 0.5
            )
        elif key == glfw.KEY_A:
            sim_state.camera_position = (
                sim_state.camera_position[0] - 0.5,
                sim_state.camera_position[1]
            )
        elif key == glfw.KEY_D:
            sim_state.camera_position = (
                sim_state.camera_position[0] + 0.5,
                sim_state.camera_position[1]
            )
        elif key == glfw.KEY_Q:
            sim_state.view_angle -= 0.1  # Rotate left
        elif key == glfw.KEY_E:
            sim_state.view_angle += 0.1  # Rotate right
        elif key == glfw.KEY_UP:
            sim_state.black_hole_mass += 0.5
            logger.info(f"Black hole mass: {sim_state.black_hole_mass} Msun")
        elif key == glfw.KEY_DOWN:
            sim_state.black_hole_mass = max(1.0, sim_state.black_hole_mass - 0.5)
            logger.info(f"Black hole mass: {sim_state.black_hole_mass} Msun")
        elif key == glfw.KEY_LEFT:
            sim_state.fov = max(0.1, sim_state.fov - 0.05)
            logger.info(f"FOV: {sim_state.fov:.2f} radians")
        elif key == glfw.KEY_RIGHT:
            sim_state.fov += 0.05
            logger.info(f"FOV: {sim_state.fov:.2f} radians")
        elif key == glfw.KEY_B:
            sim_state.b_field_exponent -= 0.1
            logger.info(f"B-field exponent: {sim_state.b_field_exponent:.2f}")
        elif key == glfw.KEY_N:
            sim_state.b_field_exponent += 0.1
            logger.info(f"B-field exponent: {sim_state.b_field_exponent:.2f}")
        elif key == glfw.KEY_1:
            sim_state.integrator_choice = 0
            logger.info("Integrator: Euler")
        elif key == glfw.KEY_2:
            sim_state.integrator_choice = 1
            logger.info("Integrator: RK4")
        elif key == glfw.KEY_3:
            sim_state.integrator_choice = 2
            logger.info("Integrator: Velocity Verlet")
        elif key == glfw.KEY_F1:
            sim_state.pp_exposure = max(0.1, sim_state.pp_exposure - 0.1)
            if renderer:
                renderer.pp_exposure = sim_state.pp_exposure
                renderer._update_post_processing_uniforms()
            logger.info(f"Exposure: {sim_state.pp_exposure:.1f}")
        elif key == glfw.KEY_F2:
            sim_state.pp_exposure += 0.1
            if renderer:
                renderer.pp_exposure = sim_state.pp_exposure
                renderer._update_post_processing_uniforms()
            logger.info(f"Exposure: {sim_state.pp_exposure:.1f}")
        elif key == glfw.KEY_F3:
            sim_state.pp_contrast = max(0.5, sim_state.pp_contrast - 0.1)
            if renderer:
                renderer.pp_contrast = sim_state.pp_contrast
                renderer._update_post_processing_uniforms()
            logger.info(f"Contrast: {sim_state.pp_contrast:.1f}")
        elif key == glfw.KEY_F4:
            sim_state.pp_contrast += 0.1
            if renderer:
                renderer.pp_contrast = sim_state.pp_contrast
                renderer._update_post_processing_uniforms()
            logger.info(f"Contrast: {sim_state.pp_contrast:.1f}")
        elif key == glfw.KEY_F5:
            sim_state.pp_vignette = not sim_state.pp_vignette
            if renderer:
                renderer.pp_vignette = sim_state.pp_vignette
                renderer._update_post_processing_uniforms()
            logger.info(f"Vignette: {'On' if sim_state.pp_vignette else 'Off'}")
        elif key == glfw.KEY_P:
            save_screenshot()
        elif key == glfw.KEY_0:
            _reset_parameters()
            logger.info("Parameters reset to defaults")

def mouse_button_callback(window, button, action, mods):
    """Handles mouse button input."""
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        x, y = glfw.get_cursor_pos(window)
        x_norm = x / renderer.width * 2.0 - 1.0
        y_norm = 1.0 - y / renderer.height * 2.0  # Flip Y coordinate
        logger.debug(f"Mouse clicked at ({x_norm:.2f}, {y_norm:.2f})")

def scroll_callback(window, x_offset, y_offset):
    """Handles mouse wheel scrolling."""
    global sim_state
    zoom_speed = 0.05
    sim_state.fov = max(0.1, min(3.0, sim_state.fov - y_offset * zoom_speed))
    logger.debug(f"FOV adjusted to {sim_state.fov:.2f}")

def _reset_parameters():
    """Resets all parameters to their default values."""
    global sim_state, renderer
    sim_state.camera_position = CAMERA_POSITION
    sim_state.black_hole_mass = BLACK_HOLE_MASS
    sim_state.fov = FIELD_OF_VIEW
    sim_state.b_field_exponent = 1.25
    sim_state.integrator_choice = 2
    sim_state.view_angle = VIEW_ANGLE
    sim_state.pp_exposure = 1.5
    sim_state.pp_contrast = 1.2
    sim_state.pp_saturation = 1.4
    sim_state.pp_gamma = 2.2
    sim_state.pp_bloom = 0.3
    sim_state.pp_vignette = True
    if renderer:
        renderer.pp_exposure = sim_state.pp_exposure
        renderer.pp_contrast = sim_state.pp_contrast
        renderer.pp_saturation = sim_state.pp_saturation
        renderer.pp_gamma = sim_state.pp_gamma
        renderer.pp_bloom = sim_state.pp_bloom
        renderer.pp_vignette = sim_state.pp_vignette
        renderer._update_post_processing_uniforms()

def update_camera_for_rotation():
    """Updates camera position for auto-rotation."""
    global sim_state
    if sim_state.auto_rotate:
        if not hasattr(sim_state, 'rotation_angle'):
            sim_state.rotation_angle = 0.0
        sim_state.rotation_angle += sim_state.rotation_speed
        radius = math.sqrt(sim_state.camera_position[0]**2 + sim_state.camera_position[1]**2)
        sim_state.camera_position = (
            radius * math.cos(sim_state.rotation_angle),
            radius * math.sin(sim_state.rotation_angle)
        )

def main() -> None:
    """Main entry point."""
    global sim_state, renderer
    args = parse_arguments()
    if args.log_file:
        setup_file_logging(args.log_file)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set initial state from command-line args
    sim_state.camera_position = (args.camera_x, args.camera_y)
    sim_state.black_hole_mass = args.mass
    sim_state.fov = args.fov
    sim_state.b_field_exponent = args.b
    sim_state.integrator_choice = args.integrator

    logger.info("Initializing simulation.")

    try:
        renderer = OpenGLRenderer(
            width=args.width,
            height=args.height,
            title=WINDOW_TITLE,
            vsync=not args.no_vsync
        )
        renderer.initialize()
        
        renderer.pp_exposure = sim_state.pp_exposure
        renderer.pp_contrast = sim_state.pp_contrast
        renderer.pp_saturation = sim_state.pp_saturation
        renderer.pp_gamma = sim_state.pp_gamma
        renderer.pp_bloom = sim_state.pp_bloom
        renderer.pp_vignette = sim_state.pp_vignette
        renderer._update_post_processing_uniforms()
        
        glfw.set_key_callback(renderer.window, key_callback)
        glfw.set_mouse_button_callback(renderer.window, mouse_button_callback)
        glfw.set_scroll_callback(renderer.window, scroll_callback)
        
        renderer.last_key_time = {}
        last_image = None

        logger.info("Entering simulation loop.")
        logger.info("Controls: WASD=Move, QE=Rotate, Arrows=Mass/FOV, BN=Magnetic field, 123=Integrator")
        logger.info("          Space=Pause, R=Auto-rotate, P=Screenshot, H=Help, I=Info")
        logger.info("          F1-F5=Post-processing, 0=Reset parameters")

        while not glfw.window_should_close(renderer.window):
            glfw.poll_events()
            cam_x, cam_y = sim_state.camera_position
            if cam_x != 0 or cam_y != 0:
                sim_state.view_angle = math.atan2(-cam_y, -cam_x)
            if sim_state.auto_rotate:
                update_camera_for_rotation()
            if not sim_state.paused or last_image is None:
                image = run_simulation(
                    custom_camera_position=sim_state.camera_position,
                    custom_black_hole_mass=sim_state.black_hole_mass,
                    custom_fov=sim_state.fov,
                    b_field_exponent=sim_state.b_field_exponent,
                    integrator_choice=sim_state.integrator_choice,
                )
                last_image = image
            else:
                image = last_image

            glBindTexture(GL_TEXTURE_2D, renderer.texture)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, renderer.width, renderer.height, GL_RGB, GL_FLOAT, image)
            renderer.render_frame()
            
            fps = get_current_fps()
            current_time = time.time()
            if current_time - renderer.last_frame_time > renderer.fps_update_interval:
                glfw.set_window_title(renderer.window, f"{WINDOW_TITLE} - {fps:.2f} FPS")
                renderer.last_frame_time = current_time

    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
    finally:
        try:
            if renderer:
                renderer.cleanup()
        except Exception:
            pass
        glfw.terminate()
        logger.info("Simulation terminated.")

if __name__ == "__main__":
    main()
