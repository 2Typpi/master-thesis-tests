from __future__ import annotations
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import sensor
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import support
from mujoco_warp._src.support import mul_m
import mujoco_warp._src.types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
from mujoco_warp._src import util_misc
import warp as wp
import warp._src.context
__all__: list[str] = ['Data', 'DisableBit', 'EnableBit', 'IntegratorType', 'Model', 'derivative', 'discrete_acc', 'forward', 'inv_constraint', 'inverse', 'mul_m', 'sensor', 'smooth', 'solver', 'support', 'util_misc', 'wp']
def discrete_acc(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, qacc: wp.array2d[wp.float32]):
    """
    Convert discrete-time qacc to continuous-time qacc.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      qacc: Acceleration.
    """
def inv_constraint(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Inverse constraint solver.
    """
def inverse(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Inverse dynamics.
    """
_qfrc_eulerdamp: warp._src.context.Kernel
_qfrc_inverse: warp._src.context.Kernel
_zero_qfrc_constraint_nefc: warp._src.context.Kernel
