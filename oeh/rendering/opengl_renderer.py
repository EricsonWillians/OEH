# oeh/rendering/opengl_renderer.py

import sys
import os
import ctypes
import time
import datetime
import numpy as np
import glfw
from OpenGL.GL import *

from oeh.config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    CAMERA_POSITION,
    BLACK_HOLE_MASS,
    FIELD_OF_VIEW
)
from oeh.simulation.raytracer import run_simulation, get_current_fps
from oeh.custom_types import Vector2D
from utils.logger import get_logger

logger = get_logger(__name__)

# Constants for post-processing parameters
PP_DEFAULT_EXPOSURE = 1.5
PP_DEFAULT_CONTRAST = 1.2
PP_DEFAULT_SATURATION = 1.4
PP_DEFAULT_GAMMA = 2.2
PP_DEFAULT_BLOOM = 0.3
PP_DEFAULT_VIGNETTE = True


def compile_shader(source, shader_type):
    """Compiles an OpenGL shader from source."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        logger.error(f"Shader compilation error: {error}")
        raise RuntimeError(f"Shader compilation error: {error}")
    return shader


def framebuffer_size_callback(window, width, height):
    """GLFW callback for window resizing."""
    glViewport(0, 0, width, height)


class OpenGLRenderer:
    """
    Professional OpenGL renderer for the black hole simulation.
    
    Features:
    - High-performance GPU rendering pipeline
    - Advanced post-processing effects (exposure, contrast, bloom, etc.)
    - Real-time parameter adjustment
    - FPS monitoring and display
    - Screenshot capability with timestamp
    - Thorough error handling and logging
    """
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, title=WINDOW_TITLE, vsync=True):
        # Window parameters
        self.width = width
        self.height = height
        self.title = title
        self.vsync = vsync
        self.window = None
        
        # OpenGL objects
        self.shader_program = None
        self.post_processing_program = None
        self.framebuffer = None
        self.render_texture = None
        self.VAO = None
        self.screen_VAO = None
        self.texture = None
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
        self.fps_display_value = 0.0
        self.render_time_ms = 0.0
        
        # Simulation parameters
        self.camera_position = CAMERA_POSITION
        self.black_hole_mass = BLACK_HOLE_MASS
        self.fov = FIELD_OF_VIEW
        self.b_field_exponent = 1.25  # Default magnetic field exponent from paper
        self.integrator_choice = 2  # Default to Velocity Verlet

        # Post-processing parameters
        self.pp_exposure = PP_DEFAULT_EXPOSURE
        self.pp_contrast = PP_DEFAULT_CONTRAST 
        self.pp_saturation = PP_DEFAULT_SATURATION
        self.pp_gamma = PP_DEFAULT_GAMMA
        self.pp_bloom = PP_DEFAULT_BLOOM
        self.pp_vignette = PP_DEFAULT_VIGNETTE
        
        # User interface state
        self.show_help = False
        self.show_info = True
        self.auto_rotate = False
        self.rotation_speed = 0.005
        self.rotation_angle = 0.0
        self.last_key_time = {}  # For key repeat rate limiting
        
        # Simulation state
        self.paused = False
        self.last_image = None  # Cache for paused state

    def initialize(self):
        """Initializes GLFW, creates window, and sets up OpenGL context and resources."""
        if not glfw.init():
            logger.error("Failed to initialize GLFW")
            raise RuntimeError("Failed to initialize GLFW")

        # Request OpenGL 3.3 Core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)  # Enable MSAA
        
        # For macOS compatibility
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Create window
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            logger.error("Failed to create GLFW window")
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)
        glfw.swap_interval(1 if self.vsync else 0)  # Enable/disable VSync

        # Set up input callbacks
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_position_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        # Log OpenGL information
        logger.info(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        logger.info(f"OpenGL Vendor: {glGetString(GL_VENDOR).decode()}")
        logger.info(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode()}")
        logger.info(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")

        # Enable OpenGL features
        glEnable(GL_MULTISAMPLE)  # Enable MSAA
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up rendering resources
        self._setup_shaders()
        self._setup_framebuffer()
        self._setup_vertex_buffers()
        self._setup_texture()

        glClearColor(0.0, 0.0, 0.1, 1.0)  # Dark blue background
        logger.info("Renderer initialized successfully")

    def _setup_shaders(self):
        """Compiles and links the vertex and fragment shaders for rendering and post-processing."""
        # Shader for displaying the texture on screen
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
        """
        
        # Post-processing fragment shader with advanced effects
        fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;

        uniform sampler2D screenTexture;
        
        // Post-processing parameters
        uniform float exposure;
        uniform float contrast;
        uniform float saturation;
        uniform float gamma;
        uniform float bloomStrength;
        uniform bool enableVignette;

        vec3 adjustExposure(vec3 color, float exposure) {
            // Exposure tone mapping - simulates camera exposure
            return vec3(1.0) - exp(-color * exposure);
        }

        vec3 adjustContrast(vec3 color, float contrast) {
            // Improved contrast adjustment with proper luminance preservation
            const vec3 luminanceWeights = vec3(0.2126, 0.7152, 0.0722);
            float luminance = dot(color, luminanceWeights);
            return mix(vec3(luminance), color, contrast);
        }

        vec3 adjustSaturation(vec3 color, float saturation) {
            // Saturation adjustment with proper luminance preservation
            const vec3 luminanceWeights = vec3(0.2126, 0.7152, 0.0722);
            float luminance = dot(color, luminanceWeights);
            return mix(vec3(luminance), color, saturation);
        }

        vec3 adjustGamma(vec3 color, float gamma) {
            // Gamma correction for proper color space transformation
            return pow(max(color, vec3(0.0001)), vec3(1.0 / gamma));
        }

        vec3 applyVignette(vec3 color, vec2 texCoord) {
            // Improved smooth vignette effect
            vec2 center = vec2(0.5);
            float dist = length(texCoord - center);
            float radius = 1.3;  // Adjust for vignette size
            float softness = 0.8; // Adjust for vignette softness
            float vignette = smoothstep(radius, radius - softness, dist);
            return color * vignette;
        }
        
        vec3 applyBloom(vec3 color, vec2 texCoord, float strength) {
            // Advanced bloom effect using multi-pass sampling
            const vec3 luminanceWeights = vec3(0.2126, 0.7152, 0.0722);
            float brightness = dot(color, luminanceWeights);
            
            // Only apply bloom to bright areas
            if (brightness > 0.7) {
                vec2 texSize = textureSize(screenTexture, 0);
                vec2 texelSize = 1.0 / texSize;
                
                // Multi-sample blur
                vec3 bloomColor = vec3(0.0);
                float totalWeight = 0.0;
                
                // Two-pass Gaussian approximation
                for (int x = -3; x <= 3; x++) {
                    for (int y = -3; y <= 3; y++) {
                        // Gaussian weight
                        float weight = exp(-(x*x + y*y) / 8.0);
                        vec2 offset = vec2(float(x), float(y)) * texelSize * 2.0;
                        bloomColor += texture(screenTexture, texCoord + offset).rgb * weight;
                        totalWeight += weight;
                    }
                }
                
                bloomColor /= totalWeight;
                
                // Adaptive bloom factor based on brightness
                float bloomFactor = smoothstep(0.7, 1.0, brightness) * strength;
                
                // Add bloom to original color
                return color + bloomColor * bloomFactor;
            }
            
            return color;
        }

        void main() {
            // Sample the texture
            vec3 color = texture(screenTexture, TexCoord).rgb;
            
            // Apply post-processing pipeline
            color = applyBloom(color, TexCoord, bloomStrength);
            color = adjustExposure(color, exposure);
            color = adjustContrast(color, contrast);
            color = adjustSaturation(color, saturation);
            if (enableVignette) {
                color = applyVignette(color, TexCoord);
            }
            color = adjustGamma(color, gamma);
            
            // Final color output
            FragColor = vec4(color, 1.0);
        }
        """

        # Compile shaders
        vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

        # Create and link the shader program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)

        # Check for linking errors
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            logger.error(f"Shader linking error: {error}")
            glDeleteProgram(self.shader_program)
            raise RuntimeError(f"Shader linking error: {error}")

        # Clean up shader objects
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        # Set up uniforms for post-processing
        glUseProgram(self.shader_program)
        glUniform1i(glGetUniformLocation(self.shader_program, "screenTexture"), 0)
        self._update_post_processing_uniforms()

    def _update_post_processing_uniforms(self):
        """Updates all post-processing uniform values in the shader."""
        glUseProgram(self.shader_program)
        glUniform1f(glGetUniformLocation(self.shader_program, "exposure"), self.pp_exposure)
        glUniform1f(glGetUniformLocation(self.shader_program, "contrast"), self.pp_contrast)
        glUniform1f(glGetUniformLocation(self.shader_program, "saturation"), self.pp_saturation)
        glUniform1f(glGetUniformLocation(self.shader_program, "gamma"), self.pp_gamma)
        glUniform1f(glGetUniformLocation(self.shader_program, "bloomStrength"), self.pp_bloom)
        glUniform1i(glGetUniformLocation(self.shader_program, "enableVignette"), 
                   GL_TRUE if self.pp_vignette else GL_FALSE)

    def _setup_framebuffer(self):
        """Creates a framebuffer for off-screen rendering with multisampling support."""
        # Create framebuffer
        self.framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
        
        # Create texture attachment
        self.render_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.render_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, self.width, self.height, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.render_texture, 0)
        
        # Create renderbuffer for depth and stencil
        rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)
        
        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.error("Framebuffer is not complete")
            raise RuntimeError("Framebuffer creation failed")
            
        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _setup_vertex_buffers(self):
        """Sets up vertex buffers for rendering a screen-sized quad."""
        vertices = np.array([
            # positions      # texture coords
            -1.0,  1.0,     0.0, 1.0,  # top left
            -1.0, -1.0,     0.0, 0.0,  # bottom left
             1.0, -1.0,     1.0, 0.0,  # bottom right
             1.0,  1.0,     1.0, 1.0   # top right
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,  # first triangle
            0, 2, 3   # second triangle
        ], dtype=np.uint32)

        # Generate and bind a VAO
        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        # Bind and set up the VAO
        glBindVertexArray(self.VAO)
        
        # Bind and fill the VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Bind and fill the EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, 
                             ctypes.c_void_p(2 * vertices.itemsize))
        glEnableVertexAttribArray(1)
        
        # Unbind VAO
        glBindVertexArray(0)

    def _setup_texture(self):
        """Creates a texture for rendering the simulation results."""
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Allocate storage for high dynamic range floating point texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, self.width, self.height, 0, GL_RGB, GL_FLOAT, None)

    def key_callback(self, window, key, scancode, action, mods):
        """Handles keyboard input."""
        # Rate limiting for smoother adjustments
        current_time = time.time()
        if key in self.last_key_time and current_time - self.last_key_time[key] < 0.1:
            return
        self.last_key_time[key] = current_time
        
        if action == glfw.PRESS or action == glfw.REPEAT:
            # Application control
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                self.paused = not self.paused
                logger.info(f"Simulation {'paused' if self.paused else 'resumed'}")
            elif key == glfw.KEY_H:
                self.show_help = not self.show_help
            elif key == glfw.KEY_I:
                self.show_info = not self.show_info
            elif key == glfw.KEY_R:
                self.auto_rotate = not self.auto_rotate
                logger.info(f"Auto-rotation {'enabled' if self.auto_rotate else 'disabled'}")
            
            # Camera movement
            elif key == glfw.KEY_W:
                self.camera_position = Vector2D(
                    self.camera_position[0],
                    self.camera_position[1] + 0.5
                )
            elif key == glfw.KEY_S:
                self.camera_position = Vector2D(
                    self.camera_position[0],
                    self.camera_position[1] - 0.5
                )
            elif key == glfw.KEY_A:
                self.camera_position = Vector2D(
                    self.camera_position[0] - 0.5,
                    self.camera_position[1]
                )
            elif key == glfw.KEY_D:
                self.camera_position = Vector2D(
                    self.camera_position[0] + 0.5,
                    self.camera_position[1]
                )
            
            # Black hole parameters
            elif key == glfw.KEY_UP:
                self.black_hole_mass += 0.5
                logger.info(f"Black hole mass: {self.black_hole_mass} Msun")
            elif key == glfw.KEY_DOWN:
                self.black_hole_mass = max(1.0, self.black_hole_mass - 0.5)
                logger.info(f"Black hole mass: {self.black_hole_mass} Msun")
            
            # Field of view
            elif key == glfw.KEY_LEFT:
                self.fov = max(0.1, self.fov - 0.05)
                logger.info(f"FOV: {self.fov:.2f} radians")
            elif key == glfw.KEY_RIGHT:
                self.fov += 0.05
                logger.info(f"FOV: {self.fov:.2f} radians")
            
            # Magnetic field exponent
            elif key == glfw.KEY_B:
                self.b_field_exponent = max(0.75, self.b_field_exponent - 0.05)
                logger.info(f"B-field exponent: {self.b_field_exponent:.2f}")
            elif key == glfw.KEY_N:
                self.b_field_exponent = min(1.5, self.b_field_exponent + 0.05)
                logger.info(f"B-field exponent: {self.b_field_exponent:.2f}")
            
            # Integrator selection
            elif key == glfw.KEY_1:
                self.integrator_choice = 0
                logger.info("Integrator: Euler")
            elif key == glfw.KEY_2:
                self.integrator_choice = 1
                logger.info("Integrator: RK4")
            elif key == glfw.KEY_3:
                self.integrator_choice = 2
                logger.info("Integrator: Velocity Verlet")
            
            # Post-processing adjustments
            elif key == glfw.KEY_E:
                self.pp_exposure = max(0.1, self.pp_exposure - 0.1)
                self._update_post_processing_uniforms()
                logger.info(f"Exposure: {self.pp_exposure:.1f}")
            elif key == glfw.KEY_Q:
                self.pp_exposure += 0.1
                self._update_post_processing_uniforms()
                logger.info(f"Exposure: {self.pp_exposure:.1f}")
            elif key == glfw.KEY_C:
                self.pp_contrast = max(0.5, self.pp_contrast - 0.1)
                self._update_post_processing_uniforms()
                logger.info(f"Contrast: {self.pp_contrast:.1f}")
            elif key == glfw.KEY_V:
                self.pp_contrast += 0.1
                self._update_post_processing_uniforms()
                logger.info(f"Contrast: {self.pp_contrast:.1f}")
            elif key == glfw.KEY_G:
                self.pp_gamma = max(0.5, self.pp_gamma - 0.1)
                self._update_post_processing_uniforms()
                logger.info(f"Gamma: {self.pp_gamma:.1f}")
            elif key == glfw.KEY_T:
                self.pp_gamma += 0.1
                self._update_post_processing_uniforms()
                logger.info(f"Gamma: {self.pp_gamma:.1f}")
            elif key == glfw.KEY_F:
                self.pp_vignette = not self.pp_vignette
                self._update_post_processing_uniforms()
                logger.info(f"Vignette: {'On' if self.pp_vignette else 'Off'}")
            
            # Screenshots
            elif key == glfw.KEY_P:
                self.take_screenshot()
            
            # Reset all parameters to defaults
            elif key == glfw.KEY_0:
                self._reset_parameters()
                logger.info("Parameters reset to defaults")

    def mouse_button_callback(self, window, button, action, mods):
        """Handles mouse button input."""
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            # Get cursor position
            x, y = glfw.get_cursor_pos(window)
            # Convert to normalized coordinates
            x_norm = x / self.width * 2.0 - 1.0
            y_norm = 1.0 - y / self.height * 2.0  # Flip Y coordinate
            
            # Example: Click to set camera focus point
            logger.debug(f"Mouse clicked at ({x_norm:.2f}, {y_norm:.2f})")

    def cursor_position_callback(self, window, x, y):
        """Handles mouse movement."""
        # Only track if a button is pressed
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            # Example: Camera orbit control
            pass

    def scroll_callback(self, window, x_offset, y_offset):
        """Handles mouse wheel scrolling."""
        # Zoom control
        zoom_speed = 0.05
        self.fov = max(0.1, min(3.0, self.fov - y_offset * zoom_speed))
        logger.debug(f"FOV adjusted to {self.fov:.2f}")

    def _reset_parameters(self):
        """Resets all parameters to their default values."""
        # Reset simulation parameters
        self.camera_position = Vector2D(*CAMERA_POSITION)
        self.black_hole_mass = BLACK_HOLE_MASS
        self.fov = FIELD_OF_VIEW
        self.b_field_exponent = 1.25
        self.integrator_choice = 2
        
        # Reset post-processing parameters
        self.pp_exposure = PP_DEFAULT_EXPOSURE
        self.pp_contrast = PP_DEFAULT_CONTRAST
        self.pp_saturation = PP_DEFAULT_SATURATION
        self.pp_gamma = PP_DEFAULT_GAMMA
        self.pp_bloom = PP_DEFAULT_BLOOM
        self.pp_vignette = PP_DEFAULT_VIGNETTE
        
        # Update shader uniforms
        self._update_post_processing_uniforms()

    def _update_camera_for_rotation(self):
        """Updates camera position for auto-rotation."""
        if self.auto_rotate:
            # Update rotation angle
            self.rotation_angle += self.rotation_speed
            
            # Calculate new camera position in a circular orbit
            radius = np.sqrt(self.camera_position[0]**2 + self.camera_position[1]**2)
            self.camera_position = Vector2D(
                radius * np.cos(self.rotation_angle),
                radius * np.sin(self.rotation_angle)
            )

    def render_frame(self):
        """Renders a single frame by running the raytracer and displaying the result."""
        # Apply auto-rotation if enabled
        self._update_camera_for_rotation()
        
        # Run the simulation only if not paused
        if not self.paused or self.last_image is None:
            t_start = time.time()
            
            # Run the simulation
            image = run_simulation(
                custom_camera_position=self.camera_position,
                custom_black_hole_mass=self.black_hole_mass,
                custom_fov=self.fov,
                b_field_exponent=self.b_field_exponent,
                integrator_choice=self.integrator_choice
            )
            
            self.last_image = image
            self.render_time_ms = (time.time() - t_start) * 1000
        else:
            # Use the cached image when paused
            image = self.last_image
        
        # Upload the image data to the texture
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_FLOAT, image)
        
        # First pass: Render to framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render the texture directly to the framebuffer
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        # Second pass: Render framebuffer to screen with post-processing
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.render_texture)
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        # Display on-screen information if enabled
        if self.show_info:
            self._draw_info()
        
        if self.show_help:
            self._draw_help()
            
        # Swap buffers
        glfw.swap_buffers(self.window)
        
        # Update performance counters
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # Update FPS display periodically
        if elapsed >= self.fps_update_interval:
            self.fps_display_value = self.frame_count / elapsed
            simulation_fps = get_current_fps()
            
            # Update window title with performance info
            glfw.set_window_title(
                self.window, 
                f"{self.title} - Display: {self.fps_display_value:.1f} FPS | "
                f"Simulation: {simulation_fps:.1f} FPS | "
                f"Render: {self.render_time_ms:.1f} ms"
            )
            
            self.frame_count = 0
            self.last_frame_time = current_time

    def _draw_info(self):
        """
        Draws on-screen information about current parameters.
        Note: In a real implementation this would use a text rendering library.
        """
        # This is a placeholder for text rendering
        # In a real implementation, you would use a proper text rendering system 
        # like freetype-gl, ImGui, or your own implementation using textured quads
        pass

    def _draw_help(self):
        """
        Draws help information overlay.
        Note: In a real implementation this would use a text rendering library.
        """
        # This is a placeholder for text rendering
        # In a real implementation, you would use a proper text rendering system
        pass

    def take_screenshot(self):
        """Saves a screenshot of the current frame."""
        try:
            import PIL.Image
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/blackhole_{timestamp}.png"
            
            # Get the framebuffer size
            width, height = glfw.get_framebuffer_size(self.window)
            
            # Read the pixel data from the framebuffer
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            
            # Create a PIL Image from the pixel data
            image = PIL.Image.frombytes("RGB", (width, height), data)
            
            # Flip the image vertically (OpenGL has origin at bottom-left)
            image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            
            # Save the image
            image.save(filename)
            logger.info(f"Screenshot saved to {filename}")
        except ImportError:
            logger.error("PIL (Pillow) is required for screenshots. Install with 'pip install Pillow'")
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")

    def run(self):
        """Main application loop."""
        logger.info("Starting simulation loop")
        
        try:
            while not glfw.window_should_close(self.window):
                # Process events
                glfw.poll_events()
                
                # Render current frame
                self.render_frame()
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            raise
        finally:
            # Ensure cleanup happens
            self.cleanup()

    def cleanup(self):
        """Cleans up OpenGL resources."""
        logger.info("Cleaning up resources")
        
        # Delete OpenGL objects
        if self.VAO:
            glDeleteVertexArrays(1, [self.VAO])
        if self.shader_program:
            glDeleteProgram(self.shader_program)
        if self.texture:
            glDeleteTextures(1, [self.texture])
        if self.render_texture:
            glDeleteTextures(1, [self.render_texture])
        if self.framebuffer:
            glDeleteFramebuffers(1, [self.framebuffer])
        
        # Terminate GLFW
        glfw.terminate()
        logger.info("Renderer shutdown complete")