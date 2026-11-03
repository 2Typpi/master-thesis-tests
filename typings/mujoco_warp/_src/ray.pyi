from __future__ import annotations
import mujoco_warp._src.types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.types import vec6f as vec6
import typing
import warp as wp
import warp._src.context
import warp._src.types
__all__: list[str] = ['Data', 'GeomType', 'MJ_MAXVAL', 'MJ_MINVAL', 'Model', 'RAY_TOL_ABS', 'RAY_TOL_REL', 'RenderContext', 'ray', 'ray_box', 'ray_capsule', 'ray_cylinder', 'ray_ellipsoid', 'ray_flex_with_bvh', 'ray_flex_with_bvh_anyhit', 'ray_geom', 'ray_hfield', 'ray_mesh', 'ray_mesh_with_bvh', 'ray_mesh_with_bvh_anyhit', 'ray_plane', 'ray_sphere', 'rays', 'safe_div', 'vec6', 'wp']
def ray(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, pnt: wp.array2d[wp.vec3f], vec: wp.array2d[wp.vec3f], geomgroup: mujoco_warp._src.types.vec6f | None = None, flg_static: bool = True, bodyexclude: int = -1, rc: mujoco_warp._src.types.RenderContext | None = None) -> tuple[warp._src.types.array, warp._src.types.array, warp._src.types.array]:
    """
    Returns the distance at which rays intersect with primitive geoms.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      pnt: Ray origin points, shape (1, 1) or (nworld, 1).
      vec: Ray directions, shape (1, 1) or (nworld, 1).
      geomgroup: Group inclusion/exclusion mask.
      flg_static: If True, allows rays to intersect with static geoms.
      bodyexclude: Ignore geoms on specified body id (-1 to disable).
      rc: Optional Render context containing BVH information for BVH accelerated ray
        intersections.
    
    Returns:
      Distances from ray origins to geom surfaces, IDs of intersected geoms (-1 if none),
      and normals at intersection points.
    """
def rays(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, pnt: wp.array2d[wp.vec3f], vec: wp.array2d[wp.vec3f], geomgroup: mujoco_warp._src.types.vec6f, flg_static: bool, bodyexclude: wp.array[wp.int32], dist: wp.array2d[wp.float32], geomid: wp.array2d[wp.int32], normal: wp.array2d[wp.vec3f], rc: mujoco_warp._src.types.RenderContext | None = None):
    """
    Ray intersection for multiple worlds and multiple rays.
    
    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state and output arrays (device).
      pnt: Ray origin points, shape (1, nray) or (nworld, nray).
      vec: Ray directions, shape (1, nray) or (nworld, nray).
      geomgroup: Group inclusion/exclusion mask. Set all elements to -1 to ignore.
      flg_static: If True, allows rays to intersect with static geoms.
      bodyexclude: Per-ray body exclusion array of shape (nray,). Geoms on the
        specified body ids are ignored (-1 to disable for that ray).
      dist: Output array for distances from ray origins to geom surfaces, shape
        (nworld, nray). -1 indicates no intersection.
      geomid: Output array for IDs of intersected geoms, shape (nworld, nray). -1
        indicates no intersection.
      normal: Output array for normals at intersection points, shape (nworld, nray).
      rc: Optional Render context containing BVH information for BVH accelerated ray
        intersections.
    """
MJ_MAXVAL: float = 10000000000.0
MJ_MINVAL: float = 1e-15
RAY_TOL_ABS: float = 1e-09
RAY_TOL_REL: float = 1e-06
_orthogonal_basis: warp._src.context.Function
_ray: warp._src.context.Kernel
_ray_bvh: warp._src.context.Kernel
_ray_eliminate: warp._src.context.Function
_ray_geom_mesh: warp._src.context.Function
_ray_geom_mesh_bvh: warp._src.context.Function
_ray_map: warp._src.context.Function
_ray_quad: warp._src.context.Function
_ray_triangle: warp._src.context.Function
ray_box: warp._src.context.Function
ray_capsule: warp._src.context.Function
ray_cylinder: warp._src.context.Function
ray_ellipsoid: warp._src.context.Function
ray_flex_with_bvh: warp._src.context.Function
ray_flex_with_bvh_anyhit: warp._src.context.Function
ray_geom: warp._src.context.Function
ray_hfield: warp._src.context.Function
ray_mesh: warp._src.context.Function
ray_mesh_with_bvh: warp._src.context.Function
ray_mesh_with_bvh_anyhit: warp._src.context.Function
ray_plane: warp._src.context.Function
ray_sphere: warp._src.context.Function
safe_div: warp._src.context.Function
