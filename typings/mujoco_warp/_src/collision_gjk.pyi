from __future__ import annotations
import math as math
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import mat43f as mat43
from mujoco_warp._src.types import mat63f as mat63
import warp as wp
import warp._src.codegen
import warp._src.context
import warp._src.types
__all__: list[str] = ['EDGE_TOL', 'FACE_TOL', 'FLOAT_MAX', 'FLOAT_MIN', 'GJKResult', 'Geom', 'GeomType', 'INTERSECT_TOL', 'MAXVAL', 'MAXVAL2', 'MINVAL', 'MINVAL2', 'MIN_DIST2', 'MIN_DIST3', 'MIN_DIST4', 'MIN_EPATOL', 'OverflowType', 'Polytope', 'SupportPoint', 'ccd', 'epa_phase', 'gjk', 'gjk_phase', 'mat43', 'mat63', 'math', 'multicontact', 'support', 'wp']
EDGE_TOL: float = 0.08878294703611658
FACE_TOL: float = 0.9960509968448313
FLOAT_MAX: float = 1e+30
FLOAT_MIN: float = -1e+30
GJKResult: warp._src.codegen.Struct
Geom: warp._src.codegen.Struct
INTERSECT_TOL: float = 3e-07
MAXVAL: float = 1000000000000000.0
MAXVAL2: float = 1e+30
MINVAL: float = 1e-15
MINVAL2: float = 1e-30
MIN_DIST2: float = 1e-10
MIN_DIST3: float = 1e-10
MIN_DIST4: float = 1e-17
MIN_EPATOL: float = 1e-07
Polytope: warp._src.codegen.Struct
SupportPoint: warp._src.codegen.Struct
_CYLINDER_COS_16: tuple = (1.0, 0.9238795325112867, 0.7071067811865476, 0.38268343236508984, 6.123233995736766e-17, -0.3826834323650897, -0.7071067811865475, -0.9238795325112867, -1.0, -0.9238795325112868, -0.7071067811865477, -0.38268343236509034, -1.8369701987210297e-16, 0.38268343236509, 0.7071067811865474, 0.9238795325112865)
_CYLINDER_SIN_16: tuple = (0.0, 0.3826834323650898, 0.7071067811865475, 0.9238795325112867, 1.0, 0.9238795325112867, 0.7071067811865476, 0.3826834323650899, 1.2246467991473532e-16, -0.38268343236508967, -0.7071067811865475, -0.9238795325112865, -1.0, -0.9238795325112866, -0.7071067811865477, -0.3826834323650904)
_FACE_DELETED_BIT: warp._src.types.uint32
_FACE_INVALID_BIT: warp._src.types.uint32
_FACE_INVALID_OR_DELETED_MASK: warp._src.types.uint32
_S1D: warp._src.context.Function
_S2D: warp._src.context.Function
_S3D: warp._src.context.Function
_add_edge: warp._src.context.Function
_aligned_face_edge: warp._src.context.Function
_aligned_faces: warp._src.context.Function
_area4: warp._src.context.Function
_attach_face: warp._src.context.Function
_box_edge_normals: warp._src.context.Function
_box_face: warp._src.context.Function
_box_normals: warp._src.context.Function
_box_normals2: warp._src.context.Function
_cylinder_edge_normals: warp._src.context.Function
_cylinder_face: warp._src.context.Function
_cylinder_normals: warp._src.context.Function
_delete_face: warp._src.context.Function
_det3: warp._src.context.Function
_discrete_geoms: warp._src.context.Function
_epa: warp._src.context.Function
_epa_support: warp._src.context.Function
_epa_witness: warp._src.context.Function
_feature_dim: warp._src.context.Function
_get_edge: warp._src.context.Function
_get_face_verts: warp._src.context.Function
_gjk_support: warp._src.context.Function
_halfspace: warp._src.context.Function
_inflate: warp._src.context.Function
_intersect1: warp._src.context.Function
_intersect2: warp._src.context.Function
_invalidate_face: warp._src.context.Function
_is_face_deleted: warp._src.context.Function
_is_invalid_face: warp._src.context.Function
_linear_combine: warp._src.context.Function
_mesh_edge_normals: warp._src.context.Function
_mesh_face: warp._src.context.Function
_mesh_normals: warp._src.context.Function
_plane_intersect: warp._src.context.Function
_plane_normal: warp._src.context.Function
_polygon_clip: warp._src.context.Function
_polygon_quad: warp._src.context.Function
_polytope2: warp._src.context.Function
_polytope3: warp._src.context.Function
_polytope4: warp._src.context.Function
_project_origin_line: warp._src.context.Function
_project_origin_plane: warp._src.context.Function
_ray_triangle: warp._src.context.Function
_replace_simplex3: warp._src.context.Function
_rotmat: warp._src.context.Function
_same_side: warp._src.context.Function
_same_sign: warp._src.context.Function
_set_edge: warp._src.context.Function
_subdistance: warp._src.context.Function
_test_tetra: warp._src.context.Function
_tri_affine_coord: warp._src.context.Function
_tri_point_intersect: warp._src.context.Function
_witness_on_face: warp._src.context.Function
ccd: warp._src.context.Function
epa_phase: warp._src.context.Function
gjk: warp._src.context.Function
gjk_phase: warp._src.context.Function
multicontact: warp._src.context.Function
support: warp._src.context.Function
