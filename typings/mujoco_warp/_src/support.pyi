from __future__ import annotations
import mujoco_warp._src.types
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import State
from mujoco_warp._src.types import vec10f as vec10
from mujoco_warp._src.types import vec5f as vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import typing
import warp as wp
import warp._src.context
import warp._src.types
__all__: list[str] = ['ConeType', 'Data', 'DynType', 'JointType', 'MJ_MINVAL', 'Model', 'State', 'apply_ft', 'cache_kernel', 'compute_interp_cell_quat', 'compute_interp_face_quat', 'contact_force', 'contact_force_fn', 'contact_force_kernel', 'dphi2D', 'eval_basis_trilinear', 'event_scope', 'flex_dphi', 'flex_phi', 'gather_face_node_index', 'gather_face_node_index_fast', 'get_face_metadata', 'get_state', 'jac', 'jac_dof', 'jac_dot_dof', 'mat33_to_quat_polar', 'motion_cross', 'mul_m', 'mul_m_dense', 'mul_m_kernel', 'next_act', 'select_top4_weights', 'set_state', 'transform_force', 'vec10', 'vec5', 'wp', 'xfrc_accumulate']
def _make_jac_kernel(*args):
    ...
def apply_ft(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ft: wp.array2d[wp.spatial_vectorf], qfrc: wp.array2d[wp.float32], flg_add: bool):
    ...
def contact_force(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, contact_ids: wp.array[wp.int32], to_world_frame: bool, force: wp.array[wp.spatial_vectorf]):
    """
    Compute forces for contacts in Data.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      contact_ids: IDs for each contact.
      to_world_frame: If True, map force from contact to world frame.
      force: Contact forces.
    """
def get_state(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, state: wp.array2d[wp.float32], sig: int, active: warp._src.types.array | None = None):
    """
    Copy concatenated state components specified by sig from Data into state.
    
    The bits of the integer sig correspond to element fields of State.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output information (device).
      state: Concatenation of state components.
      sig: Bitflag specifying state components.
      active: Per-world bitmask for getting state.
    """
def jac(*args, **kwargs):
    """
    Compute translational and rotational Jacobian for point on body.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state (device).
      jacp: Output translational Jacobian (optional).
      jacr: Output rotational Jacobian (optional).
      point: 3D point in global coordinates.
      body: Body ID for each world.
    """
def mul_m(*args, **kwargs):
    """
    Multiply vectors by inertia matrix; optionally skip per world.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      res: Result: M @ vec.
      vec: Input vector to multiply by M.
      skip: Per-world bitmask to skip computing output.
      M: Input matrix: M @ vec.
    """
def mul_m_dense(*args):
    ...
def mul_m_kernel(*args):
    ...
def set_state(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, state: wp.array2d[wp.float32], sig: int, active: warp._src.types.array | None = None):
    """
    Copy concatenated state components specified by sig from state into Data.
    
    The bits of the integer sig correspond to element fields of State.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output information (device).
      state: Concatenation of state components.
      sig: Bitflag specifying state components.
      active: Per-world bitmask for setting state.
    """
def xfrc_accumulate(*args, **kwargs):
    """
    Map applied forces at each body via Jacobians to dof space and accumulate.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      qfrc: Total applied force mapped to dof space.
    """
MJ_MINVAL: float = 1e-15
_apply_ft: warp._src.context.Kernel
_compute_jacp: warp._src.context.Function
_compute_jacr: warp._src.context.Function
_decode_pyramid: warp._src.context.Function
_phi: warp._src.context.Function
compute_interp_cell_quat: warp._src.context.Function
compute_interp_face_quat: warp._src.context.Function
contact_force_fn: warp._src.context.Function
contact_force_kernel: warp._src.context.Kernel
dphi2D: warp._src.context.Function
eval_basis_trilinear: warp._src.context.Function
flex_dphi: warp._src.context.Function
flex_phi: warp._src.context.Function
gather_face_node_index: warp._src.context.Function
gather_face_node_index_fast: warp._src.context.Function
get_face_metadata: warp._src.context.Function
jac_dof: warp._src.context.Function
jac_dot_dof: warp._src.context.Function
mat33_to_quat_polar: warp._src.context.Function
motion_cross: warp._src.context.Function
next_act: warp._src.context.Function
select_top4_weights: warp._src.context.Function
transform_force: warp._src.context.Function
