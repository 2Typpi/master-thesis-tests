from __future__ import annotations
import mujoco as mujoco
from mujoco_warp._src import math as mjmath
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.warp_util import event_scope
import numpy as np
import warp as wp
import warp._src.context
__all__: list[str] = ['Data', 'GeomType', 'MJ_MAXVAL', 'Model', 'RenderContext', 'SPLAT_MIN_RESPONSE', 'accumulate_flex_vertex_normals', 'build_flex_bvh', 'build_hfield_bvh', 'build_mesh_bvh', 'build_scene_bvh', 'build_splat_bvh', 'compute_bvh_group_roots', 'compute_mesh_group_roots', 'event_scope', 'mjmath', 'mujoco', 'normalize_vertex_normals', 'np', 'refit_bvh', 'refit_flex_bvh', 'refit_scene_bvh', 'refit_splat_bvh', 'wp']
def _optimize_hfield_mesh(data: np.ndarray, nr: int, nc: int, sx: float, sy: float, sz_scale: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy meshing for heightfield optimization.
    
    Merges coplanar adjacent cells into larger rectangles to
    reduce triangle and vertex count.
    """
def build_flex_bvh(mjm: mujoco.MjModel, mjd: mujoco.MjData, nworld: int, flex_id: int, constructor: str = 'sah', leaf_size: int = 2) -> tuple[wp.Mesh, wp.array, wp.array, wp.array, int]:
    """
    Create a Warp mesh BVH for a single 2D or 3D flex.
    """
def build_hfield_bvh(mjm: mujoco.MjModel, hfieldid: int, constructor: str = 'sah', leaf_size: int = 2) -> tuple[wp.Mesh, wp.vec3]:
    """
    Create a Warp mesh BVH from heightfield data.
    """
def build_mesh_bvh(mjm: mujoco.MjModel, meshid: int, constructor: str = 'sah', leaf_size: int = 2) -> tuple[wp.Mesh, wp.vec3]:
    """
    Create a Warp mesh BVH from mesh data.
    """
def build_scene_bvh(mjm: mujoco.MjModel, mjd: mujoco.MjData, rc: RenderContext, nworld: int):
    """
    Build a global BVH for all geometries in all worlds.
    """
def build_splat_bvh(splat_position: np.ndarray, splat_rotation: np.ndarray, splat_scale: np.ndarray, splat_rgba: np.ndarray, splat_adr: np.ndarray, splat_group_id: np.ndarray, constructor: str = 'sah') -> tuple:
    """
    Creates a splat BVH for a set of splats.
    """
def refit_bvh(*args, **kwargs):
    """
    Refit the dynamic BVH structures in the render context.
    """
def refit_flex_bvh(m: Model, d: Data, rc: RenderContext):
    """
    Refit per-flex BVHs.
    """
def refit_scene_bvh(m: Model, d: Data, rc: RenderContext):
    ...
def refit_splat_bvh(rc: RenderContext):
    """
    Refits splat BVHs.
    """
MJ_MAXVAL: float = 10000000000.0
SPLAT_MIN_RESPONSE: float = 0.01
_build_flex_2d_elements: warp._src.context.Kernel
_build_flex_2d_sides: warp._src.context.Kernel
_build_flex_3d_shells: warp._src.context.Kernel
_compute_box_bounds: warp._src.context.Function
_compute_bvh_bounds: warp._src.context.Kernel
_compute_capsule_bounds: warp._src.context.Function
_compute_cylinder_bounds: warp._src.context.Function
_compute_ellipsoid_bounds: warp._src.context.Function
_compute_flex_bvh_bounds: warp._src.context.Kernel
_compute_plane_bounds: warp._src.context.Function
_compute_sphere_bounds: warp._src.context.Function
_compute_splat_bounds: warp._src.context.Kernel
_update_flex_2d_face_points: warp._src.context.Kernel
_update_flex_3d_face_points: warp._src.context.Kernel
accumulate_flex_vertex_normals: warp._src.context.Kernel
compute_bvh_group_roots: warp._src.context.Kernel
compute_mesh_group_roots: warp._src.context.Kernel
normalize_vertex_normals: warp._src.context.Kernel
