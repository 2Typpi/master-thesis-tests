from __future__ import annotations
import mujoco as mujoco
from mujoco_warp._src import math as mjmath
from mujoco_warp._src import smooth
import mujoco_warp._src.types
from mujoco_warp._src import types
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec10f as vec10
import numpy as np
import warp as wp
import warp._src.context
__all__: list[str] = ['BiasType', 'MJ_MINVAL', 'TrnType', 'mjmath', 'mujoco', 'np', 'set_const', 'set_const_0', 'set_const_fixed', 'set_const_spring', 'set_length_range', 'smooth', 'types', 'vec10', 'wp']
def set_const(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, restore: bool = True):
    """
    Recomputes qpos0-dependent constant model fields.
    
    This function propagates changes from some model fields to derived fields,
    allowing modifications that would otherwise be unsafe. It should be called
    after modifying model parameters at runtime.
    
    Model fields that can be modified safely with set_const:
    
    ==================================  ==============================================
    Field                               Notes
    ==================================  ==============================================
    qpos0, qpos_spring
    body_mass, body_inertia,            Mass and inertia are usually scaled together
    body_ipos, body_iquat               since inertia is sum(m * r^2).
    body_pos, body_quat                 Unsafe for static bodies (invalidates BVH).
    body_gravcomp                       If changing from 0 to >0 bodies, required.
    dof_armature
    eq_data                             For connect/weld, offsets computed if not set.
    hfield_size
    tendon_stiffness, tendon_damping    Only if changing from/to zero.
    actuator_gainprm, actuator_biasprm  For position actuators with dampratio.
    ==================================  ==============================================
    
    For selective updates, use the sub-functions directly based on what changed:
    
    ==============  ===============
    Modified Field  Call
    ==============  ===============
    body_mass       set_const
    body_gravcomp   set_const_fixed
    body_inertia    set_const_0
    qpos0           set_const_0
    ==============  ===============
    
    Computes:
      - Fixed quantities (via set_const_fixed):
        - body_subtreemass: mass of body and all descendants
      - qpos0-dependent quantities (via set_const_0):
        - tendon_length0: tendon resting lengths
        - dof_invweight0: inverse inertia for DOFs
        - body_invweight0: inverse spatial inertia for bodies
        - tendon_invweight0: inverse weight for tendons
        - cam_pos0, cam_poscom0, cam_mat0: camera references
        - light_pos0, light_poscom0, light_dir0: light references
        - actuator_acc0: acceleration from unit actuator force
        - actuator_biasprm[2] (dampratio resolution)
    
    Skips: actuator_length0 (not in mjwarp).
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      restore: Whether to restore state fields to correspond to d.qpos.
    """
def set_const_0(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, restore: bool = True):
    """
    Compute quantities that depend on qpos0.
    
    Computes:
      - tendon_length0: tendon resting lengths
      - eq_data: connect/weld anchor data, recomputed so the constraint is
        satisfied at qpos0
      - dof_invweight0: inverse inertia for DOFs
      - body_invweight0: inverse spatial inertia for bodies
      - tendon_invweight0: inverse weight for tendons
      - cam_pos0, cam_poscom0, cam_mat0: camera references
      - light_pos0, light_poscom0, light_dir0: light references
      - actuator_acc0: acceleration from unit actuator force
      - actuator_biasprm[2] (dampratio resolution): for position actuators where
        gainprm[0] == -biasprm[1] and biasprm[2] > 0, converts dampratio to
        damping via biasprm[2] = -dampratio * 2 * sqrt(kp * reflected_mass)
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      restore: Whether to restore state fields to correspond to d.qpos.
    """
def set_const_fixed(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Compute fixed quantities (independent of qpos0).
    
    Computes:
      - body_subtreemass: mass of body and all descendants (depends on body_mass)
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
    """
def set_const_spring(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, restore: bool = True):
    """
    Compute quantities that depend on qpos_spring.
    
    Computes:
      - tendon_lengthspring: spring resting length range
    """
def set_length_range(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, index: int = -1):
    """
    Compute feasible actuator length ranges from joint/tendon limits.
    
    For joint and tendon transmissions with limits, copies the range directly
    from jnt_range or tendon_range scaled by gear. Actuators without limits
    keep (0, 0). This covers the common robotics use case; simulation-based
    computation for general transmissions is not yet implemented.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object (unused, kept for API compatibility with MuJoCo C).
      index: Actuator index to compute for, or -1 for all actuators.
    """
MJ_MINVAL: float = 1e-15
_accumulate_subtreemass: warp._src.context.Kernel
_compute_actuator_acc0: warp._src.context.Kernel
_compute_body_A_diag_entry: warp._src.context.Kernel
_compute_body_jac_row: warp._src.context.Kernel
_compute_cam_pos0: warp._src.context.Kernel
_compute_dof_M0: warp._src.context.Kernel
_compute_eq_data0: warp._src.context.Kernel
_compute_light_pos0: warp._src.context.Kernel
_compute_meaninertia: warp._src.context.Kernel
_compute_tendon_dot_product: warp._src.context.Kernel
_copy_actuator_moment: warp._src.context.Kernel
_copy_qpos0_to_qpos: warp._src.context.Kernel
_copy_tendon_jacobian: warp._src.context.Kernel
_copy_tendon_length0: warp._src.context.Kernel
_extract_dof_A_diag: warp._src.context.Kernel
_finalize_body_invweight0: warp._src.context.Kernel
_finalize_dof_invweight0: warp._src.context.Kernel
_init_subtreemass: warp._src.context.Kernel
_resolve_dampratio: warp._src.context.Kernel
_resolve_tendon_lengthspring: warp._src.context.Kernel
_set_length_range: warp._src.context.Kernel
_set_unit_vector: warp._src.context.Kernel
