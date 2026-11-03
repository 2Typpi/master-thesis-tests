from __future__ import annotations
from mujoco_warp._src import math
import mujoco_warp._src.types
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10f as vec10
from mujoco_warp._src import util_misc
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.context
__all__: list[str] = ['BiasType', 'Data', 'DisableBit', 'DynType', 'GainType', 'IntegratorType', 'MJ_MINVAL', 'Model', 'deriv_rne_body2jnt_sparse', 'deriv_rne_cacc_cfrcbody_forward', 'deriv_rne_cfrcbody_backward', 'deriv_rne_cvel_cdof_dot', 'deriv_rne_vel', 'deriv_smooth_vel', 'ellipsoid_max_moment', 'event_scope', 'geom_semiaxes', 'math', 'next_act', 'util_misc', 'vec10', 'wp']
def deriv_rne_vel(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, out: wp.array2d[wp.float32]):
    """
    Compute RNE velocity derivatives and add to the output.
    
    Implements the analytical derivative of inverse-dynamics Coriolis/centrifugal
    forces with respect to joint velocities.
    
    Args:
      m: The model (device).
      d: The data (device).
      out: D-structure output array (nworld, nD) to accumulate RNE terms into.
    """
def deriv_smooth_vel(*args, **kwargs):
    """
    Analytical derivative of smooth forces w.r.t. velocities.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      out: M - dt * qDeriv (derivatives of smooth forces w.r.t velocities).
    """
MJ_MINVAL: float = 1e-15
_deriv_box_fluid: warp._src.context.Function
_deriv_ellipsoid_fluid: warp._src.context.Function
_geom_ellipsoid_fluid_B: warp._src.context.Function
_get_jac_column_local: warp._src.context.Function
_nonzero_mask: warp._src.context.Function
_qderiv_actuator_passive: warp._src.context.Kernel
_qderiv_actuator_passive_actuation_sparse: warp._src.context.Kernel
_qderiv_actuator_passive_vel: warp._src.context.Kernel
_qderiv_box_fluid: warp._src.context.Kernel
_qderiv_ellipsoid_fluid: warp._src.context.Kernel
_qderiv_tendon_damping: warp._src.context.Kernel
deriv_rne_body2jnt_sparse: warp._src.context.Kernel
deriv_rne_cacc_cfrcbody_forward: warp._src.context.Kernel
deriv_rne_cfrcbody_backward: warp._src.context.Kernel
deriv_rne_cvel_cdof_dot: warp._src.context.Kernel
ellipsoid_max_moment: warp._src.context.Function
geom_semiaxes: warp._src.context.Function
next_act: warp._src.context.Function
