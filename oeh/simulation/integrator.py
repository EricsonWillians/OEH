"""
oeh/simulation/integrator.py

Provides numerical integrators for ray tracing in the vicinity of a black hole.
This revised version includes safeguards against division by zero or extremely small
values during the gravitational force calculation.
"""

import math
from numba import cuda

@cuda.jit(device=True)
def rk4_integrate(x, y, vx, vy, mass, dt):
    eps = 1e-12

    # k1
    r_squared1 = x * x + y * y
    r1 = math.sqrt(r_squared1) if r_squared1 > eps else 1e-6
    force_magnitude1 = 3.0 * mass / (r_squared1 * r1) if r_squared1 > eps else 0.0
    force_x1 = -force_magnitude1 * (x / r1) if r1 != 0.0 else 0.0
    force_y1 = -force_magnitude1 * (y / r1) if r1 != 0.0 else 0.0
    k1_vx = force_x1 * dt
    k1_vy = force_y1 * dt
    k1_x = vx * dt
    k1_y = vy * dt

    # k2
    x2 = x + k1_x * 0.5
    y2 = y + k1_y * 0.5
    vx2 = vx + k1_vx * 0.5
    vy2 = vy + k1_vy * 0.5
    r_squared2 = x2 * x2 + y2 * y2
    r2 = math.sqrt(r_squared2) if r_squared2 > eps else 1e-6
    force_magnitude2 = 3.0 * mass / (r_squared2 * r2) if r_squared2 > eps else 0.0
    force_x2 = -force_magnitude2 * (x2 / r2) if r2 != 0.0 else 0.0
    force_y2 = -force_magnitude2 * (y2 / r2) if r2 != 0.0 else 0.0
    k2_vx = force_x2 * dt
    k2_vy = force_y2 * dt
    k2_x = vx2 * dt
    k2_y = vy2 * dt

    # k3
    x3 = x + k2_x * 0.5
    y3 = y + k2_y * 0.5
    vx3 = vx + k2_vx * 0.5
    vy3 = vy + k2_vy * 0.5
    r_squared3 = x3 * x3 + y3 * y3
    r3 = math.sqrt(r_squared3) if r_squared3 > eps else 1e-6
    force_magnitude3 = 3.0 * mass / (r_squared3 * r3) if r_squared3 > eps else 0.0
    force_x3 = -force_magnitude3 * (x3 / r3) if r3 != 0.0 else 0.0
    force_y3 = -force_magnitude3 * (y3 / r3) if r3 != 0.0 else 0.0
    k3_vx = force_x3 * dt
    k3_vy = force_y3 * dt
    k3_x = vx3 * dt
    k3_y = vy3 * dt

    # k4
    x4 = x + k3_x
    y4 = y + k3_y
    vx4 = vx + k3_vx
    vy4 = vy + k3_vy
    r_squared4 = x4 * x4 + y4 * y4
    r4 = math.sqrt(r_squared4) if r_squared4 > eps else 1e-6
    force_magnitude4 = 3.0 * mass / (r_squared4 * r4) if r_squared4 > eps else 0.0
    force_x4 = -force_magnitude4 * (x4 / r4) if r4 != 0.0 else 0.0
    force_y4 = -force_magnitude4 * (y4 / r4) if r4 != 0.0 else 0.0
    k4_vx = force_x4 * dt
    k4_vy = force_y4 * dt
    k4_x = vx4 * dt
    k4_y = vy4 * dt

    new_vx = vx + (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) / 6.0
    new_vy = vy + (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) / 6.0
    new_x = x + (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0
    new_y = y + (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y) / 6.0

    return new_x, new_y, new_vx, new_vy


@cuda.jit(device=True)
def velocity_verlet_integrate(x, y, vx, vy, mass, dt):
    eps = 1e-12

    # First half-step: calculate force at the initial position.
    r_squared = x * x + y * y
    r = math.sqrt(r_squared) if r_squared > eps else 1e-6
    force_magnitude = 3.0 * mass / (r_squared * r) if r_squared > eps else 0.0
    force_x = -force_magnitude * (x / r) if r != 0.0 else 0.0
    force_y = -force_magnitude * (y / r) if r != 0.0 else 0.0

    vx_half = vx + 0.5 * force_x * dt
    vy_half = vy + 0.5 * force_y * dt

    # Full step for position.
    new_x = x + vx_half * dt
    new_y = y + vy_half * dt

    # Second half-step: calculate force at the new position.
    r_squared_new = new_x * new_x + new_y * new_y
    r_new = math.sqrt(r_squared_new) if r_squared_new > eps else 1e-6
    force_magnitude_new = 3.0 * mass / (r_squared_new * r_new) if r_squared_new > eps else 0.0
    force_x_new = -force_magnitude_new * (new_x / r_new) if r_new != 0.0 else 0.0
    force_y_new = -force_magnitude_new * (new_y / r_new) if r_new != 0.0 else 0.0

    new_vx = vx_half + 0.5 * force_x_new * dt
    new_vy = vy_half + 0.5 * force_y_new * dt

    return new_x, new_y, new_vx, new_vy


@cuda.jit(device=True)
def euler_integrate(x, y, vx, vy, mass, dt):
    eps = 1e-12

    r_squared = x * x + y * y
    r = math.sqrt(r_squared) if r_squared > eps else 1e-6
    force_magnitude = 3.0 * mass / (r_squared * r) if r_squared > eps else 0.0
    force_x = -force_magnitude * (x / r) if r != 0.0 else 0.0
    force_y = -force_magnitude * (y / r) if r != 0.0 else 0.0

    new_vx = vx + force_x * dt
    new_vy = vy + force_y * dt
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt

    return new_x, new_y, new_vx, new_vy
