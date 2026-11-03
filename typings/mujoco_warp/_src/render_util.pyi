from __future__ import annotations
import mujoco as mujoco
from mujoco_warp._src import bvh
import mujoco_warp._src.types
from mujoco_warp._src import types
from mujoco_warp._src.types import ProjectionType
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src import warp_util
import numpy
import numpy as np
import typing
import warp as wp
import warp._src.context
import warp._src.texture
__all__: list[str] = ['ProjectionType', 'RenderContext', 'bvh', 'compute_ray', 'create_render_context', 'create_warp_texture', 'extract_depth_kernel', 'get_depth', 'get_rgb', 'get_segmentation', 'mujoco', 'np', 'pack_rgba_to_uint32', 'types', 'unpack_rgb_kernel', 'warp_util', 'wp']
def create_render_context(mjm: mujoco._structs.MjModel, nworld: int = 1, cam_res: list[tuple[int, int]] | tuple[int, int] | None = None, render_rgb: list[bool] | bool | None = None, render_depth: list[bool] | bool | None = None, render_seg: list[bool] | bool | None = None, use_textures: bool = True, use_fast_math: bool = True, use_shadows: bool = False, use_ambient_lighting: bool = True, enabled_geom_groups: list[int] = [0, 1, 2], cam_active: list[bool] | list[str] | list[int] | None = None, background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), flex_render_smooth: bool = True, use_precomputed_rays: bool = True, render_skybox: bool = False, enable_backface_culling: bool = True, shadow_light_fraction: float = 0.3, samples_per_pixel: int = 1, enable_vertex_normals: bool = True, enable_specular: bool = True, enable_emission: bool = True, enable_per_light_ambient: bool = True, splat_position: numpy.ndarray | None = None, splat_rotation: numpy.ndarray | None = None, splat_scale: numpy.ndarray | None = None, splat_rgba: numpy.ndarray | None = None, splat_adr: numpy.ndarray | None = None, splat_group_id: numpy.ndarray | None = None) -> mujoco_warp._src.types.RenderContext:
    """
    Creates a render context on device.
    
    Args:
      mjm: The model containing kinematic and dynamic information on host.
      nworld: The number of worlds.
      cam_res: The width and height to render each camera image. If None, uses the
               MuJoCo model values.
      render_rgb: Whether to render RGB images. If None, uses the MuJoCo model values.
      render_depth: Whether to render depth images. If None, uses the MuJoCo model values.
      render_seg: Whether to render segmentation (per-pixel object ID/type pairs).
        If None, uses the MuJoCo model values.
      use_textures: Whether to use textures.
      use_fast_math: Whether to enable fast math for the render kernel.
      use_shadows: Whether to use shadows.
      use_ambient_lighting: Top-level ambient switch. When False, skips all
                            ambient contributions, including headlight ambient,
                            the no-light fallback, and per-light ambient.
      enabled_geom_groups: The geom groups to render.
      cam_active: List of booleans, camera names (str), or camera indices (int) indicating
                  which cameras to include in rendering. If None, all cameras are included.
                  An empty list includes no cameras.
      flex_render_smooth: Whether to render flex meshes smoothly.
      use_precomputed_rays: Use precomputed rays instead of computing during rendering.
                            When using domain randomization for camera intrinsics, set to False.
      render_skybox: Whether to shade missed rays with the MuJoCo skybox texture.
                     Requires the model to contain a texture with type `mjTEXTURE_SKYBOX`.
      shadow_light_fraction: Fraction of a light's direct contribution reaching an
        occluded point. 0 is a true shadow.
      samples_per_pixel: Sub-pixel samples per axis, averaged. Costs n*n renders.
      enable_vertex_normals: Shade meshes from their authored vertex normals,
        matching mjr_uploadMesh. When False, use the face normal.
      enable_backface_culling: Drop primitive-ray hits whose normal faces away from
                               the ray (ray origin inside the geom). Matches MuJoCo's
                               mesh-ray rule. Default True. Disable for a small
                               performance gain when no camera is ever inside a geom.
      background_color: The color to use for background pixels when no skybox is rendered.
      enable_specular: Evaluate specular highlights per light. When False the
                       half-vector normalize and shininess `pow` are dropped at
                       compile time. Disable for performance when no specular is present.
      enable_emission: Add `mat_emission * base_color` per shaded pixel. When
                       False the term is dropped at compile time. Disable for performance
                       when no emission is present.
      enable_per_light_ambient: When ambient lighting is enabled, sum each
                                light's `ambient` color into shaded pixels
                                even outside its cone or in shadow. When False
                                the per-light ambient pass is removed at compile
                                time. Disable for performance when model lights
                                do not use ambient colors.
      splat_position: Splat centers in world coordinates (nsplat, 3).
      splat_rotation: Splat rotations as (w, x, y, z) (nsplat, 4).
      splat_scale: Splat scales as standard deviation in each dimension (nsplat, 3).
      splat_rgba: Splat color and opacity (nsplat, 4).
      splat_adr: Offset of each splat in the splat attribute arrays,
                 if None then all splats are in one group.
      splat_group_id: Splat id for each world (nworld,). If None then all worlds
                      use the first splat group.
    
    Returns:
      The render context containing rendering fields and output arrays on device.
    """
def create_warp_texture(mjm: mujoco._structs.MjModel, tex_id: int) -> warp._src.texture.Texture2D:
    """
    Create a Warp texture from a MuJoCo model texture data.
    """
def get_depth(rc: mujoco_warp._src.types.RenderContext, camera_index: int, depth_scale: float, depth_out: wp.array3d[wp.float32]):
    """
    Get the depth data output from the render context buffers for a given camera index.
    
    Args:
      rc: The render context on device.
      camera_index: The index of the camera to get the depth data for.
      depth_scale: The scale factor to apply to the depth data.
      depth_out: The output array to store the scaled and clamped depth data in
        with shape (nworld, height, width).
    """
def get_rgb(rc: mujoco_warp._src.types.RenderContext, camera_index: int, rgb_out: wp.array3d[wp.vec3f]):
    """
    Get the RGB data output from the render context buffers for a given camera index.
    
    Args:
      rc: The render context on device.
      camera_index: The index of the camera to get the RGB data for.
      rgb_out: The output array to store the RGB data in, with shape (nworld, height, width).
    """
def get_segmentation(rc: mujoco_warp._src.types.RenderContext, camera_index: int, seg_out: wp.array3d[wp.vec2i]):
    """
    Get the segmentation data from the render context buffers for a given camera index.
    
    Each pixel stores MuJoCo-style `(object_id, object_type)` data. Background
    pixels are `(-1, -1)`. Regular geometry hits are `(geom_id, mjOBJ_GEOM)`.
    Flex hits are `(flex_id, mjOBJ_FLEX)`.
    
    Args:
      rc: The render context on device.
      camera_index: The index of the camera to get the segmentation data for.
      seg_out: The output array to store segmentation data in, with shape
        `(nworld, height, width)` and dtype `wp.vec2i`.
    """
_build_rays: warp._src.context.Kernel
_convert_texture_data: warp._src.context.Kernel
_extract_seg_kernel: warp._src.context.Kernel
compute_ray: warp._src.context.Function
extract_depth_kernel: warp._src.context.Kernel
pack_rgba_to_uint32: warp._src.context.Function
unpack_rgb_kernel: warp._src.context.Kernel
