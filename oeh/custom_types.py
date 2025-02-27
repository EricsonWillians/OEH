"""
This module defines the centralized type definitions used throughout
the OEH project. It leverages Python’s advanced type hints, generics, and protocols
to create a robust, scalable, and maintainable codebase.
"""

from __future__ import annotations
from typing import Any, Callable, Protocol, runtime_checkable, TypeVar, TYPE_CHECKING
try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Number",
    "Vector2D",
    "Vector3D",
    "Color",
    "ArrayLike",
    "StateArray",
    "DerivativeFunc",
    "IntegratorFunc",
    "RayState",
    "RayStateSequence",
    "Renderable",
    "ShaderProgramID",
    "TextureID",
    "VAOID",
    "CudaKernel",
    "LogCallback",
    "T",
    "AsyncCallback",
    "StateModifier",
    "Simulatable",
    "ImageSimulatable",
]

# === Basic Scalar Types ===

# For simulation precision we use float throughout.
Number: TypeAlias = float

# === Geometric Types ===

# Two-dimensional and three-dimensional vectors.
Vector2D: TypeAlias = tuple[Number, Number]
Vector3D: TypeAlias = tuple[Number, Number, Number]

# A color is represented as an (R, G, B) tuple, where each component is in [0, 1].
Color: TypeAlias = tuple[Number, Number, Number]

# === Numpy Array Types ===

# A generic alias for NumPy arrays of floats (e.g. state vectors, image frames).
ArrayLike: TypeAlias = NDArray[np.float64]

# The simulation state is represented as a NumPy array.
StateArray: TypeAlias = NDArray[np.float64]

# === Simulation Functions and Types ===

# A derivative function calculates the derivative of the state.
DerivativeFunc: TypeAlias = Callable[[StateArray], StateArray]

# A function for integrating the state over time.
IntegratorFunc: TypeAlias = Callable[[StateArray, Number, int, DerivativeFunc], StateArray]

# === Ray State Dataclass ===

@dataclass(frozen=True)
class RayState:
    """
    Represents the state of a ray in the simulation.
    
    Attributes:
        position: A 2D coordinate (x, y) representing the ray's current position.
        velocity: A normalized 2D vector (x, y) representing the ray's direction.
        time: The simulation time associated with this state.
    """
    position: Vector2D
    velocity: Vector2D
    time: Number = 0.0

# A sequence of RayState objects can represent an entire trajectory.
RayStateSequence: TypeAlias = list[RayState]

# === Rendering Protocols and OpenGL Types ===

@runtime_checkable
class Renderable(Protocol):
    """
    Protocol for objects that can render themselves within an OpenGL context.
    """
    def render(self) -> None:
        """
        Render the object to the current OpenGL framebuffer.
        """
        ...

# OpenGL resource identifiers.
ShaderProgramID: TypeAlias = int
TextureID: TypeAlias = int
VAOID: TypeAlias = int

# === CUDA Kernel Types ===

# Although Numba's CUDA kernels are dynamically typed, we alias them as Any for clarity.
CudaKernel: TypeAlias = Any

# === Logging and Callback Types ===

# A callback function used for logging messages.
LogCallback: TypeAlias = Callable[[str], None]

# A generic asynchronous callback that processes a value of type T.
T = TypeVar("T")
AsyncCallback: TypeAlias = Callable[[T], None]

# A function that modifies a simulation state.
StateModifier: TypeAlias = Callable[[StateArray], StateArray]

# === Simulation Protocols ===

class Simulatable(Protocol[T]):
    """
    A protocol for simulation components that return a result of type T.
    """
    def run_simulation(self) -> T:
        """
        Executes the simulation step and returns a result.
        """
        ...

class ImageSimulatable(Simulatable[NDArray[np.float64]], Protocol):
    """
    A simulation component that returns an image (as a NumPy array) upon execution.
    """
    def run_simulation(self) -> NDArray[np.float64]:
        ...
