from __future__ import annotations
from mujoco_warp._src import types
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.context
__all__: list[str] = ['ConstraintType', 'EqType', 'ObjType', 'OverflowType', 'cache_kernel', 'compute_island_mapping', 'direct_dsu', 'event_scope', 'island', 'types', 'update_active_dofs', 'wp']
def _compact_dofs_builder(*args):
    ...
def compute_island_mapping(*args, **kwargs):
    """
    Compute DOF/constraint island mappings after island discovery.
    
    Populates d.dof_island, d.efc.island, d.island_idofadr, d.island_dofadr,
    d.island_nv, d.island_nefc, d.island_ne, d.island_nf, d.island_iefcadr,
    d.nidof, d.map_dof2idof, d.map_idof2dof, d.dof_islandid, d.map_efc2iefc,
    d.map_iefc2efc, d.efc_islandid.
    
    Args:
      m: Model.
      d: Data.
    """
def direct_dsu(*args, **kwargs):
    """
    Discover islands with EFC-parallel atomic minimum-root hooks.
    
    `island_parent` is the (nworld, ntree) disjoint-set workspace, owned by the caller.
    """
def island(*args, **kwargs):
    """
    Discover constraint islands.
    """
def update_active_dofs(*args, **kwargs):
    """
    Rebuild the compaction maps (dof_cdof / cdof_dof) from tree_awake.
    """
_DSU_MIN_CHUNK: int = 256
_DSU_TARGET_BLOCKS: int = 2048
_compress_roots: warp._src.context.Kernel
_compute_efc_tree: warp._src.context.Kernel
_dsu_activate: warp._src.context.Function
_dsu_find: warp._src.context.Function
_dsu_union: warp._src.context.Function
_init_dof_arrays: warp._src.context.Kernel
_init_efc_arrays: warp._src.context.Kernel
_init_island_arrays: warp._src.context.Kernel
_is_repeated_fixed_support_efc: warp._src.context.Function
_island_count_constraints: warp._src.context.Kernel
_island_count_dofs: warp._src.context.Kernel
_island_dsu: warp._src.context.Kernel
_island_map_constraints: warp._src.context.Kernel
_island_map_dofs: warp._src.context.Kernel
_island_scan_sizes: warp._src.context.Kernel
_label_roots: warp._src.context.Kernel
_propagate_labels: warp._src.context.Kernel
_reset_compact_maps: warp._src.context.Kernel
_reset_dsu: warp._src.context.Kernel
_zero_island_counts: warp._src.context.Kernel
