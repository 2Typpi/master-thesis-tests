from __future__ import annotations
from mujoco_warp._src import math
from mujoco_warp._src import support
import mujoco_warp._src.types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import mat43f as mat43
from mujoco_warp._src import util_misc
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.context
__all__: list[str] = ['Data', 'DisableBit', 'GeomType', 'JointType', 'MJ_MINVAL', 'Model', 'cache_kernel', 'ellipsoid_max_moment', 'event_scope', 'geom_semiaxes', 'mat43', 'math', 'passive', 'support', 'util_misc', 'wp']
def _fluid(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    ...
def _qfrc_passive_kernel(*args):
    ...
def passive(*args, **kwargs):
    """
    Adds all passive forces.
    """
MJ_MINVAL: float = 1e-15
_apply_face_forces: warp._src.context.Function
_flex_bending: warp._src.context.Kernel
_flex_elasticity: warp._src.context.Kernel
_flex_passive_bend_interp: warp._src.context.Kernel
_flex_passive_interp: warp._src.context.Kernel
_fluid_force: warp._src.context.Kernel
_gravity_force: warp._src.context.Kernel
_pow2: warp._src.context.Function
_pow4: warp._src.context.Function
_qfrc_adhesion: warp._src.context.Kernel
_spring_damper_dof_passive: warp._src.context.Kernel
_spring_damper_tendon_passive: warp._src.context.Kernel
ellipsoid_max_moment: warp._src.context.Function
geom_semiaxes: warp._src.context.Function
