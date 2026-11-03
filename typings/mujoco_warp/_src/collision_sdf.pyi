from __future__ import annotations
from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5f as vec5
from mujoco_warp._src.types import vec8f as vec8
from mujoco_warp._src.types import vec8i
from mujoco_warp._src.types import vec_pluginattr
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.codegen
import warp._src.context
__all__: list[str] = ['AABB', 'CollisionContext', 'Data', 'GeomType', 'MJ_MINVAL', 'MeshData', 'Model', 'OptimizationParams', 'VolumeData', 'box', 'box_project', 'capsule', 'clearance', 'compute_grad', 'contact_params', 'cylinder', 'ellipsoid', 'event_scope', 'find_oct', 'geom_collision_pair', 'get_sdf_params', 'grad_box', 'grad_capsule', 'grad_cylinder', 'grad_ellipsoid', 'grad_sphere', 'gradient_descent', 'gradient_step', 'halton', 'make_frame', 'radial_field', 'ray_mesh', 'sample_volume_grad', 'sample_volume_sdf', 'sdf', 'sdf_grad', 'sdf_narrowphase', 'sphere', 'transform_aabb', 'user_sdf', 'user_sdf_grad', 'vec5', 'vec8', 'vec8i', 'vec_pluginattr', 'wp', 'write_contact']
def sdf_narrowphase(*args, **kwargs):
    ...
AABB: warp._src.codegen.Struct
MJ_MINVAL: float = 1e-15
MeshData: warp._src.codegen.Struct
OptimizationParams: warp._src.codegen.Struct
VolumeData: warp._src.codegen.Struct
_sdf_narrowphase: warp._src.context.Kernel
box: warp._src.context.Function
box_project: warp._src.context.Function
capsule: warp._src.context.Function
clearance: warp._src.context.Function
compute_grad: warp._src.context.Function
contact_params: warp._src.context.Function
cylinder: warp._src.context.Function
ellipsoid: warp._src.context.Function
find_oct: warp._src.context.Function
geom_collision_pair: warp._src.context.Function
get_sdf_params: warp._src.context.Function
grad_box: warp._src.context.Function
grad_capsule: warp._src.context.Function
grad_cylinder: warp._src.context.Function
grad_ellipsoid: warp._src.context.Function
grad_sphere: warp._src.context.Function
gradient_descent: warp._src.context.Function
gradient_step: warp._src.context.Function
halton: warp._src.context.Function
make_frame: warp._src.context.Function
radial_field: warp._src.context.Function
ray_mesh: warp._src.context.Function
sample_volume_grad: warp._src.context.Function
sample_volume_sdf: warp._src.context.Function
sdf: warp._src.context.Function
sdf_grad: warp._src.context.Function
sphere: warp._src.context.Function
transform_aabb: warp._src.context.Function
user_sdf: warp._src.context.Function
user_sdf_grad: warp._src.context.Function
write_contact: warp._src.context.Function
