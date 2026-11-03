from __future__ import annotations
import dataclasses as dataclasses
from mujoco_warp._src import math
import mujoco_warp._src.types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.context
__all__: list[str] = ['DEFAULT_MAT_EMISSION', 'DEFAULT_MAT_SHININESS_EXPONENT', 'DEFAULT_MAT_SPECULAR', 'Data', 'GeomType', 'MAX_SHININESS', 'MJ_MAXVAL', 'Model', 'NO_LIGHT_AMBIENT_FALLBACK', 'ObjType', 'RAY_TOL_ABS', 'RAY_TOL_REL', 'RenderContext', 'SPLAT_MIN_RESPONSE', 'compute_ray', 'dataclasses', 'event_scope', 'math', 'pack_rgba_to_uint32', 'ray_box', 'ray_capsule', 'ray_cylinder', 'ray_ellipsoid', 'ray_flex_with_bvh', 'ray_flex_with_bvh_anyhit', 'ray_mesh_with_bvh', 'ray_mesh_with_bvh_anyhit', 'ray_plane', 'ray_sphere', 'ray_splat', 'render', 'sample_skybox', 'shade_splats', 'vertex_normal', 'wp']
def _build_megakernel(m: mujoco_warp._src.types.Model, rc: mujoco_warp._src.types.RenderContext):
    """
    Construct the specialised megakernel for this context.
    """
def _make_cast_ray(geom_ray_types: tuple[int], first_hit: bool = False) -> warp._src.context.Function:
    """
    Build a ray-cast func specialized to the geom types present in the scene.
    
    geom_ray_types is the set of GeomType int values that actually occur, so the
    per-type intersection branches for absent types are eliminated at compile time
    via wp.static, avoiding the register pressure of unreachable code paths.
    
    first_hit selects the variant (also resolved at compile time via wp.static):
      - False: full closest-hit cast. Returns the closest hit's full surface data.
      - True: any-hit cast (shadow rays). Uses the cheaper any-hit mesh/flex
        intersections and returns on the first hit within max_dist. The result is
        still the full tuple; callers test geom_id != -1 to detect a hit.
    """
def _make_compute_lighting(cast_ray_first_hit: warp._src.context.Function) -> warp._src.context.Function:
    """
    Build specialized compute_lighting.
    """
def _make_sample_texture(geom_ray_types: tuple[int]) -> warp._src.context.Function:
    """
    Build a texture-sampling func specialized to the geom types present in the scene.
    
    geom_ray_types is the set of GeomType int values that actually occur, so the
    per-type texture mapping branches for absent types are eliminated at compile time
    via wp.static, avoiding the register pressure of unreachable code paths.
    """
def render(*args, **kwargs):
    """
    Render the current frame.
    
    Outputs are stored in buffers within the render context.
    
    Args:
      m: The model on device.
      d: The data on device.
      rc: The render context on device.
    """
DEFAULT_MAT_EMISSION: float = 0.0
DEFAULT_MAT_SHININESS_EXPONENT: float = 64.0
DEFAULT_MAT_SPECULAR: float = 0.5
MAX_SHININESS: float = 128.0
MJ_MAXVAL: float = 10000000000.0
NO_LIGHT_AMBIENT_FALLBACK: float = 0.3
RAY_TOL_ABS: float = 1e-09
RAY_TOL_REL: float = 1e-06
SPLAT_MIN_RESPONSE: float = 0.01
_MAX_SPLAT_HITS: int = 32
_MIN_SPLAT_ALPHA: float = 0.00392156862745098
_MIN_SPLAT_TRANSMITTANCE: float = 0.005
_aa_resolve: warp._src.context.Kernel
_texture_box: warp._src.context.Function
_texture_capsule: warp._src.context.Function
_texture_cylinder: warp._src.context.Function
_texture_hfield: warp._src.context.Function
_texture_mesh: warp._src.context.Function
_texture_plane: warp._src.context.Function
_texture_sphere_ellipsoid: warp._src.context.Function
compute_ray: warp._src.context.Function
pack_rgba_to_uint32: warp._src.context.Function
ray_box: warp._src.context.Function
ray_capsule: warp._src.context.Function
ray_cylinder: warp._src.context.Function
ray_ellipsoid: warp._src.context.Function
ray_flex_with_bvh: warp._src.context.Function
ray_flex_with_bvh_anyhit: warp._src.context.Function
ray_mesh_with_bvh: warp._src.context.Function
ray_mesh_with_bvh_anyhit: warp._src.context.Function
ray_plane: warp._src.context.Function
ray_sphere: warp._src.context.Function
ray_splat: warp._src.context.Function
sample_skybox: warp._src.context.Function
shade_splats: warp._src.context.Function
vertex_normal: warp._src.context.Function
