"""
Miscellaneous utilities.
"""
from __future__ import annotations
from mujoco_warp._src import math
from mujoco_warp._src import types
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import WrapType
from mujoco_warp._src.types import vec10f as vec10
import warp as wp
import warp._src.context
__all__: list[str] = ['GeomType', 'MJ_MAXVAL', 'MJ_MINVAL', 'WrapType', 'dcmotor_slots', 'dcmotor_voltage', 'halton', 'inside_geom', 'is_intersect', 'length_circle', 'lugre_stribeck', 'math', 'muscle_bias', 'muscle_dynamics', 'muscle_dynamics_timescale', 'muscle_gain', 'muscle_gain_length', 'poly_potential', 'types', 'unpack_servo_inputs', 'vec10', 'wp', 'wrap', 'wrap_circle', 'wrap_inside']
MJ_MAXVAL: float = 10000000000.0
MJ_MINVAL: float = 1e-15
_poly_force: warp._src.context.Function
_poly_force_deriv: warp._src.context.Function
_sigmoid: warp._src.context.Function
dcmotor_slots: warp._src.context.Function
dcmotor_voltage: warp._src.context.Function
halton: warp._src.context.Function
inside_geom: warp._src.context.Function
is_intersect: warp._src.context.Function
length_circle: warp._src.context.Function
lugre_stribeck: warp._src.context.Function
muscle_bias: warp._src.context.Function
muscle_dynamics: warp._src.context.Function
muscle_dynamics_timescale: warp._src.context.Function
muscle_gain: warp._src.context.Function
muscle_gain_length: warp._src.context.Function
poly_potential: warp._src.context.Function
unpack_servo_inputs: warp._src.context.Function
wrap: warp._src.context.Function
wrap_circle: warp._src.context.Function
wrap_inside: warp._src.context.Function
