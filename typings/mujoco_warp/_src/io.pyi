from __future__ import annotations
import dataclasses as dataclasses
import mujoco as mujoco
from mujoco_warp._src import history
from mujoco_warp._src.set_const import set_const
from mujoco_warp._src.set_const import set_const_0
from mujoco_warp._src.set_const import set_const_fixed
from mujoco_warp._src.set_const import set_const_spring
from mujoco_warp._src.set_const import set_length_range
from mujoco_warp._src import sleep
from mujoco_warp._src import support
import mujoco_warp._src.types
from mujoco_warp._src import types
from mujoco_warp._src import warp_util
import numpy
import numpy as np
import typing
from typing import Any
import warnings as warnings
import warp as wp
import warp._src.types
__all__: list[str] = ['Any', 'MJ_COLLISION_TABLE', 'dataclasses', 'find_keys', 'get_data_into', 'history', 'is_sparse', 'load_trajectory', 'm_block_layout', 'make_data', 'make_trajectory', 'mujoco', 'np', 'override_model', 'put_data', 'put_model', 'reset_data', 'reset_data_keyframe', 'set_const', 'set_const_0', 'set_const_fixed', 'set_const_spring', 'set_length_range', 'sleep', 'support', 'types', 'warnings', 'warp_util', 'wp']
def __annotate__(format):
    ...
def _allocate_compact_arrays(mjm: mujoco._structs.MjModel, d: mujoco_warp._src.types.Data, nworld: int, nvmax_pad: int, njmax_pad: int, compact: bool):
    """
    Allocate workspace for the compacted dense factor/solve (when nvmax is requested).
    
    Mirrors the island-local ``i*`` Data fields with a ``c*`` (compact) prefix. The
    constant model-shadows (tolerances, dof-pair indices) are derived on the host since
    they depend only on nvmax_pad; the workspace shadows are sized by nvmax_pad so the
    blocked Cholesky never reads out of bounds on its partial tile. When the user does not
    request compaction (nvmax is None) everything is allocated empty.
    
    TODO(team): once the compact path replaces the island solver, the whole forward
    pipeline can run in compacted space and ``d.M`` / ``d.qacc`` etc. become nvmax-sized
    directly, collapsing these ``c*`` shadows into the primary Data fields.
    """
def _allocate_island_arrays(mjm: mujoco._structs.MjModel, d: mujoco_warp._src.types.Data, nworld: int, njmax: int, mjd: mujoco._structs.MjData):
    ...
def _body_pair_nnz(mjm: mujoco._structs.MjModel, body1: int, body2: int) -> int:
    """
    Returns the number of unique DOFs in the kinematic tree union of two bodies.
    """
def _body_set_nnz(mjm: mujoco._structs.MjModel, bodies) -> int:
    """
    Returns the number of unique DOFs in the kinematic tree union of a set of bodies.
    """
def _calculate_max_contact_nnz(mjm: mujoco._structs.MjModel) -> int:
    """
    Returns the maximum number of non-zeros for a single contact constraint.
    """
def _create_array(data: typing.Any, spec, sizes: dict[str, int], batch_size: int = 1) -> warp._src.types.array | None:
    """
    Creates a warp array and populates it with data.
    
    The array shape is determined by a field spec referencing MjModel / MjData array sizes.
    """
def _create_constraint(mjm, nworld: int, njmax: int, sizes: dict, mjd = None) -> mujoco_warp._src.types.Constraint:
    """
    Construct a types.Constraint with standard and island local fields allocated properly.
    """
def _default_nconmax(mjm: mujoco._structs.MjModel, mjd: mujoco._structs.MjData | None = None) -> int:
    """
    Returns a default guess for an ideal nconmax given a Model and optional Data.
    
    This guess is based off a very simple heuristic, and may need to be manually raised if MJWarp
    reports ncon overflow, or lowered in order to get the very best performance.
    """
def _default_njmax(mjm: mujoco._structs.MjModel, mjd: mujoco._structs.MjData | None = None) -> int:
    """
    Returns a default guess for an ideal njmax given a Model and optional Data.
    
    This guess is based off a very simple heuristic, and may need to be manually raised if MJWarp
    reports ncon overflow, or lowered in order to get the very best performance.
    """
def _default_njmax_nnz(mjm: mujoco._structs.MjModel, nconmax: int, njmax: int) -> int:
    """
    Returns a heuristic estimate for the number of non-zeros in the sparse constraint Jacobian.
    
    Assumes all equality, friction, and limit constraints are active and computes
    their non-zeros. For contacts, assumes njmax contact rows at the maximum
    body-pair non-zeros from all enabled collision pairs.
    
    Args:
      mjm: The model containing kinematic and dynamic information (host).
      nconmax: Maximum number of contacts per world.
      njmax: Maximum number of constraint rows per world.
    
    Returns:
      Estimated number of non-zeros in the constraint Jacobian.
    """
def _get_nflexface(mjm: mujoco._structs.MjModel) -> int:
    ...
def _get_nflexintcell(mjm: mujoco._structs.MjModel) -> int:
    ...
def _get_padded_sizes(nv: int, njmax: int, is_sparse: bool, tile_size: int, augment_cholesky: bool = False):
    ...
def _initial_body_awake(mjm: mujoco._structs.MjModel, nworld: int, init_asleep: bool) -> numpy.ndarray:
    """
    Returns the initial body awake array.
    """
def _jtdaj_groups(mjd: mujoco._structs.MjData) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Group loaded efc rows into JTDAJ blocks: maximal runs sharing (efc_type, efc_id).
    
    MuJoCo lays each constraint's rows out contiguously, so this reproduces the block list
    make_constraint builds in-kernel. Returns block start rows (adr) and lengths (nrow).
    """
def _m_blocks(mjm: mujoco._structs.MjModel):
    """
    The (start, size) diagonal blocks of M: the kinematic trees, each a contiguous dof range.
    
    M couples a dof only with its tree ancestors, so its diagonal blocks are exactly the trees.
    (dof_simplenum is not used to classify blocks: it is a contiguous-suffix run-length, so it
    misses interspersed decoupled dofs; the M_rownnz coupling check in m_block_layout catches those.)
    """
def _nvmax_pad(nvmax: int) -> int:
    """
    Reserve an augmented column and round nvmax up to the dense tile size.
    """
def _resolve_batch_size(na: int | None, n: int | None, nworld: int, default: int) -> int:
    ...
def find_keys(model: mujoco._structs.MjModel, keyname_prefix: str) -> list[int]:
    """
    Finds keyframes that start with keyname_prefix.
    """
def get_data_into(result: mujoco._structs.MjData, mjm: mujoco._structs.MjModel, d: mujoco_warp._src.types.Data, world_id: int = 0):
    """
    Gets data from a device into an existing mujoco.MjData.
    
    Args:
      result: The data object containing the current state and output arrays (host).
      mjm: The model containing kinematic and dynamic information (host).
      d: The data object containing the current state and output arrays (device).
      world_id: The id of the world to get the data from.
    """
def is_sparse(mjm: mujoco._structs.MjModel) -> bool:
    ...
def load_trajectory(npz_path: str, mjm: mujoco._structs.MjModel, mjd: mujoco._structs.MjData) -> numpy.ndarray:
    """
    Load ctrl sequence from NPZ and interpolate to model timestep.
    
    Controls are sampled on the model timestep using zero-order hold. ``times``
    may contain one timestamp per control or interval boundaries with one extra
    final timestamp. For per-control timestamps, the final interval repeats the
    preceding interval duration, or one model timestep for a single control.
    Sub-timestep control intervals may be skipped when no model-timestep sample
    falls within them.
    
    The NPZ file should contain:
      - 'ctrl': array of shape (nstep, nu) with ctrl values
      - 'times': array of shape (nstep,) or (nstep + 1,) with timestamps
      - 'qpos' (optional): array of shape (1, nq) - initial state
      - 'qvel' (optional): array of shape (1, nv) - initial state
    """
def m_block_layout(mjm: mujoco._structs.MjModel) -> dict:
    """
    Per-block dense/sparse layout for M's diagonal blocks.
    
    Blocks use scalar Cholesky through six DOFs, tile Cholesky through M_BLOCK_DENSE_MAX, and sparse
    LDL beyond that. Compact diagonal blocks also use the scalar path without allocating a factor.
    """
def make_data(mjm: mujoco._structs.MjModel, nworld: int = 1, nconmax: int | None = None, nccdmax: int | None = None, njmax: int | None = None, njmax_nnz: int | None = None, naconmax: int | None = None, naccdmax: int | None = None, nvmax: int | None = None) -> mujoco_warp._src.types.Data:
    """
    Creates a data object on device.
    
    Args:
      mjm: The model containing kinematic and dynamic information (host).
      nworld: Number of worlds.
      nconmax: Number of contacts to allocate per world. Contacts exist in large
               heterogeneous arrays: one world may have more than nconmax contacts.
      nccdmax: Number of CCD contacts to allocate per world. Same semantics as nconmax.
      njmax: Number of constraints to allocate per world. Constraint arrays are
             batched by world: no world may have more than njmax constraints.
      njmax_nnz: Number of non-zeros in constraint Jacobian (sparse). Defaults to njmax * nv.
      naconmax: Number of contacts to allocate for all worlds. Overrides nconmax.
      naccdmax: Maximum number of CCD contacts. Defaults to naconmax.
      nvmax: Capacity for compacted active DOFs per world. Defaults to nv.
    
    Returns:
      The data object containing the current state and output arrays (device).
    """
def make_trajectory(model: mujoco._structs.MjModel, keys: list[int]) -> numpy.ndarray:
    """
    Make a ctrl trajectory with linear interpolation.
    """
def override_model(model: mujoco_warp._src.types.Model | mujoco._structs.MjModel, overrides: dict[str, typing.Any] | typing.Sequence[str]):
    """
    Overrides model parameters.
    
    Overrides are of the format:
      opt.iterations = 1
      opt.cone = pyramidal
      opt.disableflags = contact | spring
    """
def put_data(mjm: mujoco._structs.MjModel, mjd: mujoco._structs.MjData, nworld: int = 1, nconmax: int | None = None, nccdmax: int | None = None, njmax: int | None = None, njmax_nnz: int | None = None, naconmax: int | None = None, naccdmax: int | None = None, nvmax: int | None = None) -> mujoco_warp._src.types.Data:
    """
    Moves data from host to a device.
    
    Args:
      mjm: The model containing kinematic and dynamic information (host).
      mjd: The data object containing current state and output arrays (host).
      nworld: The number of worlds.
      nconmax: Number of contacts to allocate per world.  Contacts exist in large
               heterogenous arrays: one world may have more than nconmax contacts.
      nccdmax: Number of CCD contacts to allocate per world. Same semantics as nconmax.
      njmax: Number of constraints to allocate per world.  Constraint arrays are
             batched by world: no world may have more than njmax constraints.
      njmax_nnz: Number of non-zeros in constraint Jacobian (sparse). Defaults to njmax * nv.
      naconmax: Number of contacts to allocate for all worlds. Overrides nconmax.
      naccdmax: Maximum number of CCD contacts. Defaults to naconmax.
      nvmax: Capacity for compacted active DOFs per world. Defaults to nv.
    
    Returns:
      The data object containing the current state and output arrays (device).
    """
def put_model(mjm: mujoco._structs.MjModel, batch_sizes: dict[str, int] | None = None) -> mujoco_warp._src.types.Model:
    """
    Creates a model on device.
    
    Args:
      mjm: The model containing kinematic and dynamic information (host).
      batch_sizes: Optional per-field leading batch sizes for `Model` fields whose
        array spec starts with `*`. Fields not listed here keep the default shared
        leading dimension of 1.
    
    Returns:
      The model containing kinematic and dynamic information (device).
    """
def reset_data(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, reset: warp._src.types.array | None = None):
    """
    Clear data, set defaults; optionally by world.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      reset: Per-world bitmask (bool or integer array). Reset if nonzero/True.
    
    Raises:
      ValueError: If reset is specified but its shape is not (d.nworld,) or its
        dtype is not bool or integer.
    """
def reset_data_keyframe(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, key: int | warp._src.types.array):
    """
    Reset data, set fields from specified keyframe.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      key: The keyframe index to initialize the data with. If a plain integer is
        given, all worlds are reset to that keyframe (in this case, if keyframe
        index is < 0 or >= m.nkey, a ValueError is raised). If an array of
        integers is given, each value of the array indicates the target
        keyframe of the corresponding world (in this case, worlds whose
        keyframe index is < 0 or >= m.nkey are not reset).
    
    Raises:
      ValueError: If key is an int and key<0 or key>=m.nkey.
      ValueError: If key is a wp.array but its shape is not (d.nworld,) or its
        dtype is not int.
    """
MJ_COLLISION_TABLE: dict
_COMPACT_DATA_FIELDS: tuple = ('ctol', 'cls_tol', 'cdof_tri_row', 'cdof_tri_col', 'cM', 'cqLD', 'crhs', 'cx', 'cJ', 'cMa', 'cqfrc_smooth', 'cqacc_smooth', 'cqacc_warmstart', 'cqacc', 'cqfrc_constraint')
__conditional_annotations__: set = {0}
