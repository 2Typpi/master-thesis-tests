from __future__ import annotations
from mujoco_warp._src import collision_driver
from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import history
from mujoco_warp._src import island
from mujoco_warp._src import math
from mujoco_warp._src import passive
from mujoco_warp._src import sensor
from mujoco_warp._src import sleep
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src.support import xfrc_accumulate
import mujoco_warp._src.types
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import mat66f as mat66
from mujoco_warp._src.types import vec10f as vec10
from mujoco_warp._src.types import vec6f as vec6
from mujoco_warp._src import util_misc
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import typing
import warp as wp
import warp._src.context
import warp._src.types
__all__: list[str] = ['BiasType', 'Data', 'DisableBit', 'DynType', 'EnableBit', 'GainType', 'IntegratorType', 'JointType', 'MJ_MINVAL', 'Model', 'OverflowType', 'TrnType', 'cache_kernel', 'collision_driver', 'constraint', 'derivative', 'euler', 'event_scope', 'forward', 'fwd_acceleration', 'fwd_actuation', 'fwd_kinematics', 'fwd_position', 'fwd_velocity', 'history', 'implicit', 'island', 'mat66', 'math', 'next_act', 'passive', 'rungekutta4', 'sensor', 'sleep', 'smooth', 'solver', 'step', 'step1', 'step2', 'util_misc', 'vec10', 'vec6', 'wp', 'xfrc_accumulate']
def _advance(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, qacc: warp._src.types.array, qvel: warp._src.types.array | None = None):
    """
    Advance state and time given activation derivatives and acceleration.
    """
def _energy_pos(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    ...
def _energy_vel(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    ...
def _launch_implicit_free_body_solve(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, qacc: wp.array2d[wp.float32]):
    ...
def _next_time_builder(*args):
    ...
def _qfrc_smooth(*args):
    ...
def _rk_accumulate(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, scale: float, qvel_rk: wp.array2d[wp.float32], qacc_rk: wp.array2d[wp.float32], act_dot_rk: warp._src.types.array | None = None):
    """
    Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4.
    """
def _rk_perturb_state(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, scale: float, qpos_t0: wp.array2d[wp.float32], qvel_t0: wp.array2d[wp.float32], act_t0: warp._src.types.array | None = None):
    ...
def euler(*args, **kwargs):
    """
    Euler integrator, semi-implicit in velocity.
    """
def forward(*args, **kwargs):
    """
    Forward dynamics.
    """
def fwd_acceleration(*args, **kwargs):
    """
    Add up all non-constraint forces, compute qacc_smooth.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      factorize: Flag to factorize inertia matrix.
    """
def fwd_actuation(*args, **kwargs):
    """
    Actuation-dependent computations.
    """
def fwd_kinematics(*args, **kwargs):
    """
    Kinematics-dependent computations.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
    """
def fwd_position(*args, **kwargs):
    """
    Position-dependent computations.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      factorize: Flag to factorize inertia matrix.
    """
def fwd_velocity(*args, **kwargs):
    """
    Velocity-dependent computations.
    """
def implicit(*args, **kwargs):
    """
    Integrates fully implicit in velocity.
    """
def rungekutta4(*args, **kwargs):
    """
    Runge-Kutta explicit order 4 integrator.
    """
def step(*args, **kwargs):
    """
    Advance simulation.
    """
def step1(*args, **kwargs):
    """
    Advance simulation in two phases: before input is set by user.
    """
def step2(*args, **kwargs):
    """
    Advance simulation in two phases: after input is set by user.
    """
MJ_MINVAL: float = 1e-15
_actuator_force: warp._src.context.Kernel
_actuator_velocity: warp._src.context.Kernel
_compute_damping_deriv: warp._src.context.Kernel
_euler_damp_qfrc: warp._src.context.Kernel
_implicit_free_body_reset_m: warp._src.context.Kernel
_implicit_free_body_solve: warp._src.context.Kernel
_map_m2d: warp._src.context.Kernel
_next_activation: warp._src.context.Kernel
_next_position: warp._src.context.Kernel
_next_velocity: warp._src.context.Kernel
_project_spatial_B: warp._src.context.Function
_qfrc_actuator: warp._src.context.Kernel
_qfrc_actuator_gravcomp_limits: warp._src.context.Kernel
_rk_accumulate_activation_velocity: warp._src.context.Kernel
_rk_accumulate_velocity_acceleration: warp._src.context.Kernel
_tendon_actuator_force: warp._src.context.Kernel
_tendon_actuator_force_clamp: warp._src.context.Kernel
_tendon_velocity: warp._src.context.Kernel
next_act: warp._src.context.Function
