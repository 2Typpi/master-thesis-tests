from __future__ import annotations
from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import mat43f as mat43
from mujoco_warp._src.types import mat63f as mat63
from mujoco_warp._src.types import vec5f as vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.codegen
import warp._src.context
__all__: list[str] = ['CollisionContext', 'Data', 'DisableBit', 'Geom', 'GeomType', 'MJ_MAXCONPAIR', 'MJ_MAXVAL', 'MJ_MAX_EPAFACES', 'MJ_MAX_EPAHORIZON', 'Model', 'OverflowType', 'cache_kernel', 'ccd', 'ccd_hfield_kernel_builder', 'ccd_kernel_builder', 'contact_margin_gap', 'contact_material_params', 'contact_params', 'convex_narrowphase', 'epa_phase', 'event_scope', 'geom_collision_pair', 'geom_collision_pair_from_types', 'gjk_phase', 'make_frame', 'mat43', 'mat63', 'multicontact', 'support', 'upper_trid_index', 'vec5', 'wp', 'write_contact']
def _ccd_grid_size(kernel, naconmax: int, device) -> int:
    ...
def ccd_hfield_kernel_builder(*args):
    """
    Kernel builder for heightfield CCD collisions (no multiccd args).
    """
def ccd_kernel_builder(*args):
    """
    Kernel builder for non-heightfield CCD collisions (no hfield args).
    """
def convex_narrowphase(*args, **kwargs):
    """
    Runs narrowphase collision detection for convex geom pairs.
    
    This function handles collision detection for pairs of convex geometries that were
    identified during the broadphase. It uses the Gilbert-Johnson-Keerthi (GJK) algorithm to
    determine the distance between shapes and the Expanding Polytope Algorithm (EPA) to find
    the penetration depth and contact normal for colliding pairs.
    
    The convex geom types handled by this function are `SPHERE`, `CAPSULE`, `ELLIPSOID`, `CYLINDER`,
    `BOX`, `MESH`, `HFIELD`.
    
    To optimize performance, this function dynamically builds and launches a specialized
    kernel for each type of convex collision pair present in the model, avoiding unnecessary
    computations for non-existent pair types.
    """
Geom: warp._src.codegen.Struct
MJ_MAXCONPAIR: int = 50
MJ_MAXVAL: float = 10000000000.0
MJ_MAX_EPAFACES: int = 5
MJ_MAX_EPAHORIZON: int = 24
_CCD_MIN_BLOCKS: int = 8
_CCD_OVERSUBSCRIBE_WAVES: int = 2
_hfield_filter: warp._src.context.Function
ccd: warp._src.context.Function
contact_margin_gap: warp._src.context.Function
contact_material_params: warp._src.context.Function
contact_params: warp._src.context.Function
epa_phase: warp._src.context.Function
geom_collision_pair: warp._src.context.Function
geom_collision_pair_from_types: warp._src.context.Function
gjk_phase: warp._src.context.Function
make_frame: warp._src.context.Function
multicontact: warp._src.context.Function
support: warp._src.context.Function
upper_trid_index: warp._src.context.Function
write_contact: warp._src.context.Function
