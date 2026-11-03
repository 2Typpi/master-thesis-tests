from __future__ import annotations
from mujoco_warp._src import history
from mujoco_warp._src import math
from mujoco_warp._src import ray
from mujoco_warp._src import smooth
from mujoco_warp._src import support
import mujoco_warp._src.types
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DataType
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SensorType
from mujoco_warp._src.types import Stage
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec5f as vec5
from mujoco_warp._src.types import vec6f as vec6
from mujoco_warp._src.types import vec8f as vec8
from mujoco_warp._src.types import vec8i
from mujoco_warp._src.types import vec_pluginattr
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
from typing import Any
import warp as wp
import warp._src.context
__all__: list[str] = ['Any', 'ConeType', 'ConstraintType', 'ContactType', 'Data', 'DataType', 'DisableBit', 'GeomType', 'JointType', 'MJ_MAXCONPAIR', 'MJ_MAXVAL', 'MJ_MINVAL', 'Model', 'ObjType', 'OverflowType', 'SensorType', 'Stage', 'TrnType', 'cache_kernel', 'energy_pos', 'energy_vel', 'event_scope', 'get_sdf_params', 'history', 'inside_geom', 'math', 'poly_potential', 'ray', 'sdf', 'sensor_acc', 'sensor_pos', 'sensor_vel', 'smooth', 'support', 'vec5', 'vec6', 'vec8', 'vec8i', 'vec_pluginattr', 'wp']
def _contact_match_builder(*args):
    ...
def _contact_sort(*args):
    ...
def _energy_vel_kinetic(*args):
    ...
def _preprocess_tactile_contacts_kernel(*args):
    ...
def energy_pos(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Position-dependent energy (potential).
    """
def energy_vel(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Velocity-dependent energy (kinetic).
    """
def sensor_acc(*args, **kwargs):
    """
    Compute acceleration-dependent sensor values.
    """
def sensor_pos(*args, **kwargs):
    """
    Compute position-dependent sensor values.
    """
def sensor_vel(*args, **kwargs):
    """
    Compute velocity-dependent sensor values.
    """
MJ_MAXCONPAIR: int = 50
MJ_MAXVAL: float = 10000000000.0
MJ_MINVAL: float = 1e-15
_accelerometer: warp._src.context.Function
_actuator_force: warp._src.context.Function
_actuator_pos: warp._src.context.Function
_actuator_vel: warp._src.context.Function
_ball_ang_vel: warp._src.context.Function
_ball_quat: warp._src.context.Function
_cam_projection: warp._src.context.Function
_check_match: warp._src.context.Function
_clock: warp._src.context.Function
_cvel_offset: warp._src.context.Function
_energy_pos_gravity: warp._src.context.Kernel
_energy_pos_passive_joint: warp._src.context.Kernel
_energy_pos_passive_tendon: warp._src.context.Kernel
_energy_pos_zero: warp._src.context.Kernel
_force: warp._src.context.Function
_frame_angvel: warp._src.context.Function
_frame_axis: warp._src.context.Function
_frame_linvel: warp._src.context.Function
_frame_pos: warp._src.context.Function
_frame_quat: warp._src.context.Function
_frameangacc: warp._src.context.Function
_framelinacc: warp._src.context.Function
_get_body_id: warp._src.context.Function
_get_mat: warp._src.context.Function
_get_pos: warp._src.context.Function
_get_quat: warp._src.context.Function
_gyro: warp._src.context.Function
_joint_actuator_force: warp._src.context.Function
_joint_pos: warp._src.context.Function
_joint_vel: warp._src.context.Function
_limit_frc: warp._src.context.Kernel
_limit_pos: warp._src.context.Kernel
_limit_vel: warp._src.context.Kernel
_magnetometer: warp._src.context.Function
_sensor_acc: warp._src.context.Kernel
_sensor_collision: warp._src.context.Kernel
_sensor_pos: warp._src.context.Kernel
_sensor_rangefinder_init: warp._src.context.Kernel
_sensor_tactile: warp._src.context.Kernel
_sensor_touch: warp._src.context.Kernel
_sensor_vel: warp._src.context.Kernel
_subtree_angmom: warp._src.context.Function
_subtree_com: warp._src.context.Function
_subtree_linvel: warp._src.context.Function
_tendon_actuator_force: warp._src.context.Kernel
_tendon_actuator_force_cutoff: warp._src.context.Kernel
_tendon_pos: warp._src.context.Function
_tendon_vel: warp._src.context.Function
_torque: warp._src.context.Function
_transform_spatial: warp._src.context.Function
_velocimeter: warp._src.context.Function
_write_scalar: warp._src.context.Function
_write_vector: warp._src.context.Function
get_sdf_params: warp._src.context.Function
inside_geom: warp._src.context.Function
poly_potential: warp._src.context.Function
sdf: warp._src.context.Function
