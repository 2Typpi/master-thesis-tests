from __future__ import annotations
import mujoco_warp._src.types
from mujoco_warp._src.types import CtrlChart
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import Model
import typing
from typing import Union
import warp as wp
import warp._src.context
import warp._src.types
__all__: list[str] = ['CtrlChart', 'Data', 'GainType', 'MJ_MINVAL', 'Model', 'Union', 'apply_sensor_delay', 'init_ctrl_history', 'init_sensor_history', 'insert_ctrl_history', 'read_ctrl', 'read_ctrl_delayed', 'read_sensor', 'reset_history', 'wp']
def apply_sensor_delay(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, sensorid: wp.array[wp.int32]):
    """
    Apply delay/interval logic for given sensors after computation.
    
    Matches MuJoCo C architecture where the delayed read (mj_sensorPos) occurs
    before the fresh value insert (mj_advance). We save fresh sensordata,
    overwrite with delayed values, then insert the saved fresh values.
    """
def init_ctrl_history(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctrlid: int, times: warp._src.types.array | None, values: wp.array2d[wp.float32]):
    """
    Initialize history buffer for 1 actuator across all worlds.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      ctrlid: actuator index.
      times: timestamps or None (nsample,).
      values: ctrl values (nworld, nsample * dim) or (nworld, nsample, dim).
    
    Raises:
      ValueError: If times are not strictly increasing.
    """
def init_sensor_history(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, sensorid: int, times: warp._src.types.array | None, values: wp.array2d[wp.float32], phase: float | warp._src.types.array | None = None):
    """
    Initialize history buffer for 1 sensor across all worlds.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      sensorid: sensor index.
      times: timestamps or None (nsample,).
      values: sensor values (nworld, nsample * dim).
      phase: user slot value per world (nworld,) or scalar float. If None,
        preserves the existing user slot in the buffer.
    
    Raises:
      ValueError: If times are not strictly increasing.
    """
def insert_ctrl_history(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Insert current ctrl values into history buffers.
    """
def read_ctrl(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctrlid: int, time: wp.array[wp.float32], interp: int, result: warp._src.types.array):
    """
    Read delayed ctrl for 1 actuator across all worlds.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      ctrlid: actuator index.
      time: query time per world (nworld,).
      interp: interpolation mode (-1=model default, 0=ZOH, 1=linear, 2=cubic).
      result: output buffer (nworld, dim) or (nworld,) if dim == 1.
    """
def read_ctrl_delayed(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctrl: wp.array2d[wp.float32]):
    """
    Read delayed ctrl values for all actuators.
    """
def read_sensor(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, sensorid: int, time: wp.array[wp.float32], interp: int, result: wp.array2d[wp.float32]):
    """
    Read delayed sensor for 1 sensor across all worlds.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      sensorid: sensor index.
      time: query time per world (nworld,).
      interp: interpolation mode (-1=model default, 0=ZOH, 1=linear, 2=cubic).
      result: output buffer (nworld, dim).
    """
def reset_history(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, reset: warp._src.types.array | None = None):
    """
    Reset all delay and history buffers to reference initial state matching MuJoCo C.
    """
MJ_MINVAL: float = 1e-15
_apply_sensor_delay_kernel: warp._src.context.Kernel
_history_find_index: warp._src.context.Function
_history_insert_scalar: warp._src.context.Function
_history_insert_vector: warp._src.context.Function
_history_physical_index: warp._src.context.Function
_history_read_scalar: warp._src.context.Function
_history_read_vector: warp._src.context.Function
_init_ctrl_history_kernel: warp._src.context.Kernel
_init_sensor_history_kernel: warp._src.context.Kernel
_insert_ctrl_history_kernel: warp._src.context.Kernel
_insert_sensor_history_stage: warp._src.context.Kernel
_read_ctrl_delayed_kernel: warp._src.context.Kernel
_read_ctrl_kernel: warp._src.context.Kernel
_read_sensor_kernel: warp._src.context.Kernel
_reset_actuator_history_kernel: warp._src.context.Kernel
_reset_sensor_history_kernel: warp._src.context.Kernel
