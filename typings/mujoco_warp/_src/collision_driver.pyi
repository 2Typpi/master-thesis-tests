from __future__ import annotations
from mujoco_warp._src.collision_convex import convex_narrowphase
import mujoco_warp._src.collision_core
from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.collision_core import create_collision_context
from mujoco_warp._src.collision_flex import flex_collision
from mujoco_warp._src.collision_primitive import primitive_narrowphase
from mujoco_warp._src.collision_sdf import sdf_narrowphase
import mujoco_warp._src.types
from mujoco_warp._src.types import BroadphaseFilter
from mujoco_warp._src.types import BroadphaseType
from mujoco_warp._src.types import CollisionType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import SleepState
from mujoco_warp._src.types import mat23f as mat23
from mujoco_warp._src.types import mat63f as mat63
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.context
__all__: list[str] = ['BroadphaseFilter', 'BroadphaseType', 'CollisionContext', 'CollisionType', 'Data', 'DisableBit', 'EnableBit', 'GeomType', 'MJ_COLLISION_TABLE', 'MJ_FLEX_COLLISION_TABLE', 'MJ_MAXVAL', 'Model', 'SleepState', 'cache_kernel', 'collision', 'convex_narrowphase', 'create_collision_context', 'event_scope', 'flex_collision', 'mat23', 'mat63', 'nxn_broadphase', 'primitive_narrowphase', 'sap_binary_search', 'sap_broadphase', 'sap_range', 'sdf_narrowphase', 'upper_tri_index', 'wp']
def _broadphase_filter(opt_broadphase_filter: int, ngeom_aabb: int, ngeom_rbound: int, ngeom_margin: int, ngeom_gap: int):
    ...
def _narrowphase(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.collision_core.CollisionContext):
    ...
def _nxn_broadphase(*args):
    ...
def _sap_broadphase(*args):
    ...
def _sap_project(*args):
    ...
def _segmented_sort(*args):
    ...
def collision(*args, **kwargs):
    """
    Runs the full collision detection pipeline.
    
    This function orchestrates the broadphase and narrowphase collision detection stages. It
    first identifies potential collision pairs using a broadphase algorithm (either N-squared
    or Sweep-and-Prune, based on `m.opt.broadphase`). Then, for each potential pair, it
    performs narrowphase collision detection to compute detailed contact information like
    distance, position, and frame.
    
    The results are used to populate the `d.contact` array, and the total number of contacts
    is stored in `d.nacon`.  If `d.nacon` is larger than `d.naconmax` then an overflow has
    occurred and the remaining contacts will be skipped.  If this happens, raise the `nconmax`
    parameter in `io.make_data` or `io.put_data`.
    
    This function will do nothing except zero out arrays if collision detection is disabled
    via `m.opt.disableflags` or if `d.nacon` is 0.
    
    Passing `awake_prev` (the awake state snapshotted before the post-collision wake) runs the
    incremental sleeping pass: contacts are appended to the existing buffer and only pairs involving
    a newly-awakened body are emitted.
    """
def nxn_broadphase(*args, **kwargs):
    """
    Runs broadphase collision detection using a brute-force N-squared approach.
    
    This function iterates through a pre-filtered list of all possible geometry pairs and
    performs a quick bounding sphere check to identify potential collisions.
    
    For each pair that passes the sphere check, it populates the collision arrays in `d`
    (`d.collision_pair`, `d.collision_pairid`, etc.), which are then consumed by the
    narrowphase.
    
    The initial list of pairs is filtered at model creation time to exclude pairs based on
    `contype`/`conaffinity`, parent-child relationships, and explicit `<exclude>` tags.
    
    Passing ``awake_prev`` runs the incremental sleeping pass: only pairs involving a newly-awakened
    body are emitted. When graph conditionals are available the launch is wrapped in one gated on
    whether any body woke since pass 1 (``awake_prev != body_awake``), so the broadphase is skipped
    wholesale on steps where nothing woke; otherwise it runs unconditionally and the per-pair filter
    restricts the emitted pairs.
    """
def sap_broadphase(*args, **kwargs):
    """
    Runs broadphase collision detection using a sweep-and-prune (SAP) algorithm.
    
    This method is more efficient than the N-squared approach for large numbers of
    objects. It works by projecting the bounding spheres of all geoms onto a
    single axis and sorting them. It then sweeps along the axis, only checking
    for overlaps between geoms whose projections are close to each other.
    
    For each potentially colliding pair identified by the sweep, a more precise
    bounding sphere check is performed. If this check passes, the pair is added
    to the collision arrays in `d` for the narrowphase stage.
    
    Two sorting strategies are supported, controlled by `m.opt.broadphase`
    
    - `SAP_TILE`: Uses a tile-based sort.
    - `SAP_SEGMENTED`: Uses a segmented sort.
    
    Unlike `nxn_broadphase`, SAP cannot be wrapped in a CUDA graph conditional to skip the
    incremental sleeping pass: its sort/scan utilities allocate scratch internally, which is
    not allowed inside a conditional body. The incremental filter in the sweep kernel still
    restricts the emitted pairs to those involving newly-awakened bodies.
    """
MJ_COLLISION_TABLE: dict
MJ_FLEX_COLLISION_TABLE: dict
MJ_MAXVAL: float = 10000000000.0
_aabb_filter: warp._src.context.Function
_add_geom_pair: warp._src.context.Function
_any_awake_changed: warp._src.context.Kernel
_obb_filter: warp._src.context.Function
_plane_filter: warp._src.context.Function
_sphere_filter: warp._src.context.Function
sap_binary_search: warp._src.context.Function
sap_range: warp._src.context.Kernel
upper_tri_index: warp._src.context.Function
