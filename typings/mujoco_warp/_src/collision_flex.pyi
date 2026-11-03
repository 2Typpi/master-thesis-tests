"""
Flex collision detection (geom vs flex triangles).
"""
from __future__ import annotations
import dataclasses as dataclasses
import math as math
from mujoco_warp._src import collision_primitive_core
import mujoco_warp._src.types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import mat63f as mat63
from mujoco_warp._src.types import vec5f as vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import typing
import warp as wp
import warp._src.codegen
import warp._src.context
import warp._src.types
__all__: list[str] = ['Data', 'ENABLE_SAT_PREFILTER', 'FlexWorkspace', 'Geom', 'GeomType', 'MJ_MAXCONPAIR', 'MJ_MAXVAL', 'MJ_MAX_EPAFACES', 'MJ_MAX_EPAHORIZON', 'MJ_MINMU', 'MJ_MINVAL', 'Model', 'OverflowType', 'cache_kernel', 'ccd', 'collision_primitive_core', 'dataclasses', 'event_scope', 'flex_broadphase_aabb', 'flex_collision', 'make_frame', 'mat63', 'math', 'sap_binary_search', 'sap_range', 'vec5', 'wp']
class FlexWorkspace:
    """
    GPU workspace buffers for flex collision detection, EPA CCD, and contact filtering.
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('dist', 'pos', 'nrm', 'geom', 'flex', 'elem', 'vert', 'worldid', 'ncand', 'filter_key', 'filter_val', 'cand_active', 'epa_vert', 'epa_vert_index', 'epa_face', 'epa_pr', 'epa_norm2', 'epa_horizon', 'cand_active_sorted', 'flex_group_temp', 'flex_group_ids', 'flex_group_start_indices', 'flex_fps_min_dist', 'flex_num_groups', 'fps_scratch_dist', 'fps_scratch_cidx', 'fps_scratch_count', 'fps_selected_cidx', 'fps_selected_pos', 'fps_groups_active', 'fps_condition', 'fps_iter', 'nccd')
    cand_active_sorted = None
    flex_fps_min_dist = None
    flex_group_ids = None
    flex_group_start_indices = None
    flex_group_temp = None
    flex_num_groups = None
    fps_condition = None
    fps_groups_active = None
    fps_iter = None
    fps_scratch_cidx = None
    fps_scratch_count = None
    fps_scratch_dist = None
    fps_selected_cidx = None
    fps_selected_pos = None
    nccd = None
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, dist: warp._src.types.array, pos: warp._src.types.array, nrm: warp._src.types.array, geom: warp._src.types.array, flex: warp._src.types.array, elem: warp._src.types.array, vert: warp._src.types.array, worldid: warp._src.types.array, ncand: warp._src.types.array, filter_key: warp._src.types.array, filter_val: warp._src.types.array, cand_active: warp._src.types.array, epa_vert: warp._src.types.array, epa_vert_index: warp._src.types.array, epa_face: warp._src.types.array, epa_pr: warp._src.types.array, epa_norm2: warp._src.types.array, epa_horizon: warp._src.types.array, cand_active_sorted: warp._src.types.array | None = None, flex_group_temp: warp._src.types.array | None = None, flex_group_ids: warp._src.types.array | None = None, flex_group_start_indices: warp._src.types.array | None = None, flex_fps_min_dist: warp._src.types.array | None = None, flex_num_groups: warp._src.types.array | None = None, fps_scratch_dist: warp._src.types.array | None = None, fps_scratch_cidx: warp._src.types.array | None = None, fps_scratch_count: warp._src.types.array | None = None, fps_selected_cidx: warp._src.types.array | None = None, fps_selected_pos: warp._src.types.array | None = None, fps_groups_active: warp._src.types.array | None = None, fps_condition: warp._src.types.array | None = None, fps_iter: warp._src.types.array | None = None, nccd: warp._src.types.array | None = None) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
def __annotate__(format):
    ...
def _allocate_flex_workspace(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data) -> FlexWorkspace:
    ...
def _detect_1d_geom_candidates(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ws: FlexWorkspace):
    """
    Detect candidates between 1D flex rope vertices and geoms.
    """
def _detect_elem_geom_candidates(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ws: FlexWorkspace):
    """
    Detect candidates between 2D/3D flex elements and geoms.
    """
def _detect_plane_flex_candidates(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ws: FlexWorkspace):
    """
    Detect candidates between plane geoms and flex vertices.
    """
def _filter_and_write_contacts(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ws: FlexWorkspace, enable_fps: bool = False):
    """
    Optionally applies FPS filtering and writes contacts to d.contact.
    """
def _flex_geom_collision(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ws: FlexWorkspace):
    """
    Detect and write collisions between rigid geoms and flex objects.
    """
def _flex_geom_vertex_narrowphase_detect(*args):
    ...
def _flex_narrowphase(*args):
    ...
def _flex_narrowphase_elem_detect(*args):
    ...
def _flex_plane_narrowphase(*args):
    ...
def _flex_sap_collision(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx, ws: FlexWorkspace, is_self: bool, sap_data: tuple[warp._src.types.array, warp._src.types.array, warp._src.types.array, warp._src.types.array] | None = None, enable_sat: bool | None = None):
    """
    Detect and write flex self or flex-flex collision contacts (broadphase and narrowphase).
    """
def _flex_sap_sweep(*args):
    ...
def _populate_group_starts(*args):
    ...
def _run_filter_flex_fps(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ws: FlexWorkspace, nmax_groups: int):
    """
    Applies Far Point Sampling to limit contact points per group to MJ_MAXCONPAIR in parallel.
    """
def _run_flex_narrowphase(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx, ws: FlexWorkspace):
    """
    Executes narrowphase collision detection for element pairs.
    """
def _run_flex_sap_sort(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data) -> tuple[warp._src.types.array, warp._src.types.array, warp._src.types.array, warp._src.types.array]:
    """
    Performs SAP projection, segmented sorting, and cumulative range scan for flex elements.
    """
def _write_filtered_contacts(*args):
    ...
def flex_broadphase_aabb(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Precompute dynamic flex object bounding boxes.
    """
def flex_collision(*args, **kwargs):
    """
    Runs collision detection for all flex collisions.
    """
ENABLE_SAT_PREFILTER: bool = True
Geom: warp._src.codegen.Struct
MJ_MAXCONPAIR: int = 50
MJ_MAXVAL: float = 10000000000.0
MJ_MAX_EPAFACES: int = 5
MJ_MAX_EPAHORIZON: int = 24
MJ_MINMU: float = 1e-05
MJ_MINVAL: float = 1e-15
_FPS_BLOCK_SIZE: int = 64
__conditional_annotations__: set = {0, 1}
_axis_separated: warp._src.context.Function
_collide_geom_triangle_detect: warp._src.context.Function
_collide_mesh_convex: warp._src.context.Function
_compute_filter_key: warp._src.context.Kernel
_elem_active: warp._src.context.Function
_exclude_self_collision: warp._src.context.Function
_find_group_starts: warp._src.context.Kernel
_flex_broadphase_bounds: warp._src.context.Kernel
_flex_element_aabb_filter: warp._src.context.Function
_flex_sap_project: warp._src.context.Kernel
_get_element_vertices: warp._src.context.Function
_init_cand_active: warp._src.context.Kernel
_inside_triangle: warp._src.context.Function
_mix_flex_contact_params: warp._src.context.Function
_parallel_fps_check_condition: warp._src.context.Kernel
_parallel_fps_find_seed: warp._src.context.Kernel
_parallel_fps_init_condition: warp._src.context.Kernel
_parallel_fps_init_dist_and_find_max: warp._src.context.Kernel
_parallel_fps_resolve_max: warp._src.context.Kernel
_parallel_fps_resolve_seed: warp._src.context.Kernel
_parallel_fps_step_condition: warp._src.context.Kernel
_parallel_fps_update_and_find_max: warp._src.context.Kernel
_point_segment_axis: warp._src.context.Function
_populate_active_sorted: warp._src.context.Kernel
_tie_break_fps: warp._src.context.Function
_triangle_sat_separated: warp._src.context.Function
_write_candidate: warp._src.context.Function
ccd: warp._src.context.Function
make_frame: warp._src.context.Function
sap_binary_search: warp._src.context.Function
sap_range: warp._src.context.Kernel
