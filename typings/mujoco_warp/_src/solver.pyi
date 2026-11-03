from __future__ import annotations
import dataclasses as dataclasses
from math import ceil
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_augmented_factorize_solve_newton_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_newton_func
from mujoco_warp._src import island
from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import support
import mujoco_warp._src.types
from mujoco_warp._src import types
from mujoco_warp._src.types import InverseContext
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SolverContext
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import typing
from typing import Any
import warp as wp
import warp._src.context
__all__: list[str] = ['Any', 'InverseContext', 'OverflowType', 'SolverContext', 'cache_kernel', 'ceil', 'create_blocked_cholesky_augmented_factorize_solve_newton_func', 'create_blocked_cholesky_factorize_solve_func', 'create_blocked_cholesky_solve_newton_func', 'create_inverse_context', 'dataclasses', 'event_scope', 'init_context', 'island', 'math', 'smooth', 'smooth_solve_compact', 'solve', 'solve_compact', 'solve_search_sums', 'support', 'types', 'wp']
def _JTDACJ_sparse(*args):
    ...
def _cholesky_factorize_solve(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext, skip_unchanged: bool = False, skip_noflip: bool = False):
    """
    Cholesky factorize ctx.h and form the Newton search direction.
    
    If skip_unchanged is True (blocked path only), worlds where no constraints
    changed reuse the cached factorization in hfactor instead of refactorizing.
    """
def _cholesky_factorize_solve_blocked(*args):
    ...
def _compact_gather(*args, **kwargs):
    ...
def _compact_scatter(*args, **kwargs):
    ...
def _create_solver_context(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data) -> mujoco_warp._src.types.SolverContext:
    """
    Create a SolverContext with allocated workspace arrays.
    
    Args:
      m: Model.
      d: Data.
    
    Returns:
      SolverContext with allocated arrays.
    """
def _jtdaj_groups_per_world(nworld: int, njmax: int) -> int:
    ...
def _linesearch(*args, **kwargs):
    """
    Linesearch for constraint solver.
    
    When state changes are tracked, worlds with ctx.search_unchanged reuse last
    iteration's mv/jv, so a fresh search requires clearing the flag (see the
    invalidation in _solve and the writer in _update_gradient_zero_grad_dot).
    
    Args:
      m: Model
      d: Data
      ctx: SolverContext
    """
def _linesearch_iterative(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext, fuse_jv: bool):
    """
    Iterative linesearch with parallel reductions over efc rows and dofs.
    
    Args:
      m: Model.
      d: Data.
      ctx: SolverContext.
      fuse_jv: Whether jv is computed in-kernel (True) or pre-computed (False).
    """
def _linesearch_iterative_kernel(*args):
    """
    Factory for iterative linesearch kernel.
    
    Args:
      ls_iterations: Max linesearch iterations (compile-time constant for loop optimization).
      cone_type: Friction cone type (PYRAMIDAL or ELLIPTIC) for compile-time optimization.
      fuse_jv: Whether to compute jv = J @ search in-kernel (efficient for small nv).
      is_sparse: Use sparse matrix representation for constraint Jacobian.
      incremental: Use incremental linesearch updates.
      warn_overflow: Overflow warning bitmask.
    """
def _linesearch_jv_fused_kernel(*args):
    ...
def _mul_m_compact_aware(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext | mujoco_warp._src.types.InverseContext, res, vec, skip):
    """
    M @ vec: full-coordinate sparse walk under compact, support.mul_m natively.
    """
def _solve(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext, compact: bool = False):
    """
    Finds forces that satisfy constraints.
    """
def _solve_beta_finalize_tiled(*args):
    ...
def _solve_done(*args):
    ...
def _solve_init_dof(*args):
    ...
def _solve_init_jaref_kernel(*args):
    ...
def _solver_iteration(*args, **kwargs):
    ...
def _sparse_compact(ctx: mujoco_warp._src.types.SolverContext | mujoco_warp._src.types.InverseContext) -> bool:
    """
    Whether this solve is compact over a sparse full model (full J structures exist).
    """
def _update_constraint(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext | mujoco_warp._src.types.InverseContext, track_changes: bool = False, stable_fast: bool = False):
    """
    Update constraint arrays after each solve iteration.
    """
def _update_constraint_efc(*args):
    ...
def _update_constraint_init_qfrc_constraint_dense(*args):
    ...
def _update_constraint_init_qfrc_constraint_sparse(*args):
    ...
def _update_gradient(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext, compact: bool = False):
    ...
def _update_gradient_JTDAJ_dense_tiled(*args):
    ...
def _update_gradient_JTDAJ_dense_tiled_compact(*args):
    """
    Compact-path variant of _update_gradient_JTDAJ_dense_tiled.
    
    Takes M_in as a dense 3D array (nworld, nv_pad, nv_pad) -- the compacted active-DOF
    inertia block cM -- instead of a 2D CSR array.  Cholesky reads fill_mode="upper";
    cM is full-symmetric so the tile_load covers both triangles and the upper triangle
    of the result is correct.
    """
def _update_gradient_cholesky(*args):
    ...
def _update_gradient_cholesky_blocked(*args):
    ...
def _update_gradient_cholesky_blocked_skip_unchanged(*args):
    """
    Blocked Cholesky that skips factorization when no constraints changed.
    """
def _update_gradient_grad(*args):
    ...
def _update_gradient_h_incremental_sparse(*args):
    ...
def _update_gradient_incremental(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext, stable_fast: bool = False):
    """
    Incremental gradient update: update H for changed constraints + re-factorize.
    
    Skips the full J^T*D*J rebuild by applying only the delta from constraints
    that changed QUADRATIC state, then re-factorizes and solves.
    """
def _update_gradient_init_h_sparse(*args):
    ...
def _update_gradient_zero_grad_dot(*args):
    ...
def _use_incremental(m: mujoco_warp._src.types.Model) -> bool:
    """
    Whether constraint state changes are tracked for incremental H updates.
    """
def create_inverse_context(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data) -> mujoco_warp._src.types.InverseContext:
    """
    Create an InverseContext with allocated workspace arrays.
    
    Args:
      m: Model.
      d: Data.
    
    Returns:
      InverseContext with allocated arrays.
    """
def init_context(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, ctx: mujoco_warp._src.types.SolverContext | mujoco_warp._src.types.InverseContext, grad: bool = True, compact: bool = False):
    ...
def smooth_solve_compact(*args, **kwargs):
    """
    Compacted equivalent of solve_m: qacc_smooth[active] = cM^-1 qfrc_smooth[active].
    
    Inactive DOFs are frozen (qacc_smooth set to 0). Reads the sparse Model inertia.
    """
def solve(*args, **kwargs):
    ...
def solve_compact(*args, **kwargs):
    """
    Run the dense Newton constraint solver in compacted DOF space.
    
    Gathers the active-DOF inertia, constraint Jacobian, and smooth/warmstart vectors
    into nvmax_pad-sized dense workspaces, runs the stock dense Newton solver on a
    shallow-replaced (m, d) at nvmax_pad, then scatters qacc/qfrc_constraint back.
    Inactive DOFs are frozen to 0. On the incremental Newton path the solver
    kernels read the sparse M and J directly through the compaction maps.
    """
_ALPHA_NOISE_EPS: float = 9.5367432e-07
_BLOCK_CHOLESKY_DIM: int = 32
_JTDAJ_OVERSUBSCRIBE_WAVES: int = 6
_JTDAJ_THREADS_PER_GROUP: int = 32
_active_check: warp._src.context.Function
_compute_efc_eval_pt_3alphas_elliptic: warp._src.context.Function
_compute_efc_eval_pt_3alphas_pyramidal: warp._src.context.Function
_compute_efc_eval_pt_alpha_zero_elliptic: warp._src.context.Function
_compute_efc_eval_pt_alpha_zero_pyramidal: warp._src.context.Function
_compute_efc_eval_pt_elliptic: warp._src.context.Function
_compute_efc_eval_pt_pyramidal: warp._src.context.Function
_elliptic_hessian_entry_from_projections: warp._src.context.Function
_eval_constraint: warp._src.context.Function
_eval_cost: warp._src.context.Function
_eval_elliptic_alpha_zero: warp._src.context.Function
_eval_elliptic_middle: warp._src.context.Function
_eval_elliptic_quadratic_cone_gap: warp._src.context.Function
_eval_elliptic_quadratic_shifted: warp._src.context.Function
_eval_elliptic_reference: warp._src.context.Function
_eval_elliptic_shifted: warp._src.context.Function
_eval_frictionloss_cost: warp._src.context.Function
_eval_frictionloss_pt: warp._src.context.Function
_eval_frictionloss_pt_3alphas: warp._src.context.Function
_eval_frictionloss_pt_one: warp._src.context.Function
_eval_pt: warp._src.context.Function
_eval_pt_3alphas: warp._src.context.Function
_eval_pt_direct: warp._src.context.Function
_eval_pt_direct_3alphas: warp._src.context.Function
_eval_pt_direct_alpha_zero: warp._src.context.Function
_eval_pt_direct_cost_alpha_zero: warp._src.context.Function
_eval_pt_direct_shifted: warp._src.context.Function
_eval_pt_direct_shifted_3alphas: warp._src.context.Function
_gather_J_dense: warp._src.context.Kernel
_gather_J_sparse: warp._src.context.Kernel
_gather_M_sparse: warp._src.context.Kernel
_gather_dof_vecs_compact: warp._src.context.Kernel
_gather_rhs_compact: warp._src.context.Kernel
_in_bracket: warp._src.context.Function
_init_compact_inertia: warp._src.context.Kernel
_linesearch_zero_jv: warp._src.context.Kernel
_mul_m_sparse_compact: warp._src.context.Kernel
_padding_h: warp._src.context.Kernel
_qfrc_constraint_from_grad: warp._src.context.Kernel
_rescale: warp._src.context.Function
_scatter_dof_vecs: warp._src.context.Kernel
_scatter_solution: warp._src.context.Kernel
_shift_cost: warp._src.context.Function
_solve_init_efc: warp._src.context.Kernel
_solve_init_search_cg_tiled: warp._src.context.Kernel
_solve_search_update_cg_tiled: warp._src.context.Kernel
_state_check: warp._src.context.Function
_update_gradient_JTCJ_dense: warp._src.context.Kernel
_update_gradient_grad_tiled: warp._src.context.Kernel
_update_gradient_h_incremental: warp._src.context.Kernel
_zero_change_counters: warp._src.context.Kernel
_zero_qfrc_constraint_sparse: warp._src.context.Kernel
solve_search_sums: warp._src.context.Function
