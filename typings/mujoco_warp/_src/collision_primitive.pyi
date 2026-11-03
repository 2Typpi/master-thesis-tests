from __future__ import annotations
from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import mat43f as mat43
from mujoco_warp._src.types import vec5f as vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.codegen
import warp._src.context
__all__: list[str] = ['CollisionContext', 'Data', 'Geom', 'GeomType', 'MJ_MAXVAL', 'Model', 'box_box', 'box_box_wrapper', 'cache_kernel', 'capsule_box', 'capsule_box_wrapper', 'capsule_capsule', 'capsule_capsule_wrapper', 'contact_params', 'event_scope', 'geom_collision_pair', 'make_frame', 'mat43', 'plane_box', 'plane_box_wrapper', 'plane_capsule', 'plane_capsule_wrapper', 'plane_convex', 'plane_convex_wrapper', 'plane_cylinder', 'plane_cylinder_wrapper', 'plane_ellipsoid', 'plane_ellipsoid_wrapper', 'plane_sphere', 'plane_sphere_wrapper', 'primitive_narrowphase', 'sphere_box', 'sphere_box_wrapper', 'sphere_capsule', 'sphere_capsule_wrapper', 'sphere_cylinder', 'sphere_cylinder_wrapper', 'sphere_sphere', 'sphere_sphere_wrapper', 'upper_trid_index', 'vec5', 'wp', 'write_contact']
def _primitive_narrowphase(*args):
    ...
def primitive_narrowphase(*args, **kwargs):
    """
    Runs collision detection on primitive geom pairs discovered during broadphase.
    
    This function processes collision pairs involving primitive shapes that were
    identified during the broadphase stage. It computes detailed contact information
    such as distance, position, and frame, and populates the `d.contact` array.
    
    The primitive geom types: `PLANE`, `SPHERE`, `CAPSULE`, `CYLINDER`, and `BOX`.
    
    Additionally, collisions between planes and convex hulls.
    
    To improve performance, it dynamically builds and launches a kernel tailored to
    the specific primitive collision types present in the model, avoiding
    unnecessary checks for non-existent collision pairs.
    """
Geom: warp._src.codegen.Struct
MJ_MAXVAL: float = 10000000000.0
_PRIMITIVE_COLLISIONS: dict
_PRIMITIVE_COLLISION_FUNC: list = list()
_PRIMITIVE_COLLISION_TYPES: list = list()
box_box: warp._src.context.Function
box_box_wrapper: warp._src.context.Function
capsule_box: warp._src.context.Function
capsule_box_wrapper: warp._src.context.Function
capsule_capsule: warp._src.context.Function
capsule_capsule_wrapper: warp._src.context.Function
contact_params: warp._src.context.Function
geom_collision_pair: warp._src.context.Function
make_frame: warp._src.context.Function
plane_box: warp._src.context.Function
plane_box_wrapper: warp._src.context.Function
plane_capsule: warp._src.context.Function
plane_capsule_wrapper: warp._src.context.Function
plane_convex: warp._src.context.Function
plane_convex_wrapper: warp._src.context.Function
plane_cylinder: warp._src.context.Function
plane_cylinder_wrapper: warp._src.context.Function
plane_ellipsoid: warp._src.context.Function
plane_ellipsoid_wrapper: warp._src.context.Function
plane_sphere: warp._src.context.Function
plane_sphere_wrapper: warp._src.context.Function
sphere_box: warp._src.context.Function
sphere_box_wrapper: warp._src.context.Function
sphere_capsule: warp._src.context.Function
sphere_capsule_wrapper: warp._src.context.Function
sphere_cylinder: warp._src.context.Function
sphere_cylinder_wrapper: warp._src.context.Function
sphere_sphere: warp._src.context.Function
sphere_sphere_wrapper: warp._src.context.Function
upper_trid_index: warp._src.context.Function
write_contact: warp._src.context.Function
