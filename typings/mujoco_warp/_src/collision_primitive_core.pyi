from __future__ import annotations
import warp as wp
import warp._src.context
__all__: list[str] = ['MJ_MAXVAL', 'MJ_MINVAL', 'box_box', 'box_triangle', 'capsule_box', 'capsule_capsule', 'capsule_triangle', 'closest_segment_point', 'closest_segment_to_segment_points', 'cylinder_triangle', 'mat23f', 'mat43f', 'mat83f', 'normalize_with_norm', 'plane_box', 'plane_capsule', 'plane_cylinder', 'plane_ellipsoid', 'plane_sphere', 'safe_div', 'sphere_box', 'sphere_capsule', 'sphere_cylinder', 'sphere_sphere', 'sphere_triangle', 'vec8f', 'wp']
class vec8f(warp._src.types.vector.<locals>.vec_t):
    pass
class mat23f(warp._src.types.matrix.<locals>.mat_t):
    pass
class mat43f(warp._src.types.matrix.<locals>.mat_t):
    pass
class mat83f(warp._src.types.matrix.<locals>.mat_t):
    pass
MJ_MAXVAL: float = 10000000000.0
MJ_MINVAL: float = 1e-15
_compute_rotmore: warp._src.context.Function
_tri_area_sign: warp._src.context.Function
_tri_point_segment: warp._src.context.Function
box_box: warp._src.context.Function
box_triangle: warp._src.context.Function
capsule_box: warp._src.context.Function
capsule_capsule: warp._src.context.Function
capsule_triangle: warp._src.context.Function
closest_segment_point: warp._src.context.Function
closest_segment_to_segment_points: warp._src.context.Function
cylinder_triangle: warp._src.context.Function
normalize_with_norm: warp._src.context.Function
plane_box: warp._src.context.Function
plane_capsule: warp._src.context.Function
plane_cylinder: warp._src.context.Function
plane_ellipsoid: warp._src.context.Function
plane_sphere: warp._src.context.Function
safe_div: warp._src.context.Function
sphere_box: warp._src.context.Function
sphere_capsule: warp._src.context.Function
sphere_cylinder: warp._src.context.Function
sphere_sphere: warp._src.context.Function
sphere_triangle: warp._src.context.Function
