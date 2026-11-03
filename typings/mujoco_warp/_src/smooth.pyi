from __future__ import annotations
from mujoco_warp._src import math
from mujoco_warp._src import support
import mujoco_warp._src.types
from mujoco_warp._src.types import CamLightType
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import TileSet
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import WrapType
from mujoco_warp._src.types import vec10f as vec10
from mujoco_warp._src.types import vec11f as vec11
from mujoco_warp._src.types import vec5f as vec5
from mujoco_warp._src import util_misc
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.context
__all__: list[str] = ['CamLightType', 'ConeType', 'ConstraintType', 'Data', 'DisableBit', 'EqType', 'JointType', 'MJ_MAXVAL', 'MJ_MINVAL', 'Model', 'ObjType', 'Q_LD_BLOCK_COMPACT', 'Q_LD_BLOCK_SPARSE', 'TileSet', 'TrnType', 'WrapType', 'cache_kernel', 'camlight', 'com_pos', 'com_vel', 'crb', 'event_scope', 'factor_m', 'factor_solve_i', 'factor_solve_lu', 'flex', 'kinematics', 'math', 'rne', 'rne_postconstraint', 'solve_LD', 'solve_m', 'subtree_vel', 'support', 'tendon', 'tendon_armature', 'tendon_bias', 'transform_force', 'transmission', 'util_misc', 'vec10', 'vec11', 'vec5', 'wp']
def _factor_blocks(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, M: wp.array2d[wp.float32], L: wp.array2d[wp.float32], D: wp.array2d[wp.float32]):
    ...
def _factor_i_sparse(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, M: wp.array2d[wp.float32], L: wp.array2d[wp.float32], D: wp.array2d[wp.float32]):
    """
    Sparse L'*D*L factorization of inertia-like matrix M, assumed spd.
    """
def _factor_solve_blocks(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, M: wp.array2d[wp.float32], L: wp.array2d[wp.float32], D: wp.array2d[wp.float32], x: wp.array2d[wp.float32], y: wp.array2d[wp.float32]):
    ...
def _factor_solve_lu_sparse_fused(*args):
    """
    Fused sparse LU factorization and solve in a single kernel.
    """
def _rne_cacc_forward(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, flg_acc: bool = False):
    ...
def _rne_cacc_world(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    ...
def _rne_cfrc(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, flg_cfrc_ext: bool = False):
    ...
def _rne_cfrc_backward(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    ...
def _small_cholesky_factorize_block(*args):
    ...
def _small_cholesky_factorize_solve_block(*args):
    ...
def _small_cholesky_solve_block(*args):
    ...
def _solve_LD_sparse(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, L: wp.array2d[wp.float32], D: wp.array2d[wp.float32], x: wp.array2d[wp.float32], y: wp.array2d[wp.float32]):
    """
    Computes sparse backsubstitution: x = inv(L'*D*L)*y.
    """
def _solve_LD_sparse_fused(*args):
    """
    Fused sparse backsubstitution: UP + diag + DOWN in one kernel.
    """
def _solve_blocks(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, L: wp.array2d[wp.float32], D: wp.array2d[wp.float32], x: wp.array2d[wp.float32], y: wp.array2d[wp.float32]):
    ...
def _tile_cholesky_factorize_block(*args):
    ...
def _tile_cholesky_factorize_solve_block(*args):
    ...
def _tile_cholesky_solve_block(*args):
    ...
def camlight(*args, **kwargs):
    """
    Computes camera and light positions and orientations.
    
    Updates the global positions and orientations for all cameras and lights in the model,
    including special handling for tracking and target modes.
    """
def com_pos(*args, **kwargs):
    """
    Computes subtree center of mass positions.
    
    Transforms inertia and motion to global frame centered at subtree CoM. Accumulates the
    mass-weighted positions up the kinematic tree, divides by total mass, and computes composite
    inertias and motion degrees of freedom in the subtree CoM frame.
    """
def com_vel(*args, **kwargs):
    """
    Computes the spatial velocities (cvel) and the derivative `cdof_dot` for all bodies.
    
    Propagates velocities down the kinematic tree, updating the spatial velocity and
    derivative for each body.
    """
def crb(*args, **kwargs):
    """
    Computes composite rigid body inertias for each body and the joint-space inertia matrix.
    
    Accumulates composite rigid body inertias up the kinematic tree and computes the
    joint-space inertia matrix in either sparse or dense format, depending on model options.
    """
def factor_m(*args, **kwargs):
    """
    Factorization of inertia-like matrix M, assumed spd.
    
    Compact blocks use reciprocal diagonals, full small blocks use scalar Cholesky, larger dense
    blocks use tile Cholesky, and oversized blocks use sparse LDL.
    """
def factor_solve_i(m, d, M, L, D, x, y):
    """
    Factorizes and solves the inertia-like linear system.
    
    Compact blocks use reciprocal diagonals, full small blocks use scalar Cholesky, dense blocks use
    tile Cholesky, and sparse blocks use LDL. Factorizes M and solves for x.
    
    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      M: The inertia-like matrix to factorize (CSR, length nC).
      L: Output factor: packed dense region followed by the nC LDL region (sized like d.qLD).
      D: Output reciprocal diagonal for compact and sparse blocks.
      x: Output array for the solution.
      y: Input right-hand side array.
    """
def factor_solve_lu(*args, **kwargs):
    """
    Factorize and solve non-symmetric implicit system: qacc = A \\\\ qfrc.
    
    qLU is overwritten in-place with the LU factors, then used to solve for qacc.
    
    Args:
      m: The model containing D-structure sparsity information.
      d: The data object.
      qLU: array containing the system matrix, overwritten with LU factors.
      qacc: output array for the solution.
      qfrc: input right-hand side.
    """
def flex(*args, **kwargs):
    ...
def kinematics(*args, **kwargs):
    """
    Computes forward kinematics for all bodies, sites, geoms, and flexible elements.
    
    This function updates the global positions and orientations of all bodies, as well as the
    derived positions and orientations of geoms, sites, and flexible elements, based on the
    current joint positions and any attached mocap bodies.
    """
def rne(*args, **kwargs):
    """
    Computes inverse dynamics using the recursive Newton-Euler algorithm.
    
    Computes the bias forces (`qfrc_bias`) and internal forces (`cfrc_int`) for the current state,
    including the effects of gravity and optionally joint accelerations.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      flg_acc: If True, includes joint accelerations in the computation.
    """
def rne_postconstraint(*args, **kwargs):
    """
    Computes the recursive Newton-Euler algorithm after constraints are applied.
    
    Computes `cacc`, `cfrc_ext`, and `cfrc_int`, including the effects of applied forces, equality
    constraints, contacts, and spatial tendons.
    """
def solve_LD(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data, L: wp.array2d[wp.float32], D: wp.array2d[wp.float32], x: wp.array2d[wp.float32], y: wp.array2d[wp.float32]):
    """
    Computes backsubstitution for the inertia factorization.
    
    Compact blocks use reciprocal diagonals, full small blocks use scalar Cholesky, dense blocks use
    tile Cholesky, and sparse blocks use the LDL region.
    
    Args:
      m: The model containing factorization and sparsity information.
      d: The data object containing workspace and factorization results.
      L: The factor: packed dense region followed by the nC LDL region.
      D: Reciprocal diagonal for compact and sparse blocks.
      x: Output array for the solution.
      y: Input right-hand side array.
    """
def solve_m(*args, **kwargs):
    """
    Computes backsubstitution: x = qLD * y.
    
    Args:
      m: The model containing inertia and factorization information.
      d: The data object containing factorization results.
      x: Output array for the solution.
      y: Input right-hand side array.
    """
def subtree_vel(m: mujoco_warp._src.types.Model, d: mujoco_warp._src.types.Data):
    """
    Computes subtree linear velocity and angular momentum.
    
    Computes the linear momentum and angular momentum for each subtree, accumulating
    contributions up the kinematic tree.
    """
def tendon(*args, **kwargs):
    """
    Computes tendon lengths and moments.
    
    Updates the tendon length and moment arrays for all tendons in the model, including joint,
    site, and geom tendons.
    """
def tendon_armature(*args, **kwargs):
    """
    Add tendon armature to M.
    """
def tendon_bias(*args, **kwargs):
    """
    Add bias force due to tendon armature.
    
    Args:
      m: The model containing kinematic and dynamic information.
      d: The data object containing the current state and output arrays.
      qfrc: Force.
    """
def transmission(*args, **kwargs):
    """
    Computes actuator/transmission lengths and moments.
    
    Updates the actuator length and moments for all actuators in the model, including joint
    and tendon transmissions.
    """
MJ_MAXVAL: float = 10000000000.0
MJ_MINVAL: float = 1e-15
Q_LD_BLOCK_COMPACT: int = -2
Q_LD_BLOCK_SPARSE: int = -1
_M: warp._src.context.Kernel
_accumulate_jac_chain: warp._src.context.Function
_accumulate_jac_dot_chain: warp._src.context.Function
_angular_momentum: warp._src.context.Kernel
_cacc_branch: warp._src.context.Kernel
_cacc_world: warp._src.context.Kernel
_cam_local_to_global: warp._src.context.Kernel
_cdof: warp._src.context.Kernel
_cfrc: warp._src.context.Kernel
_cfrc_backward: warp._src.context.Kernel
_cfrc_ext: warp._src.context.Kernel
_cfrc_ext_contact: warp._src.context.Kernel
_cfrc_ext_equality: warp._src.context.Kernel
_cfrc_ext_spatial_tendon: warp._src.context.Kernel
_cfrc_ext_tendon_actuator: warp._src.context.Kernel
_cfrc_ext_tendon_constraint: warp._src.context.Kernel
_cinert: warp._src.context.Kernel
_compute_body_inertial_frames: warp._src.context.Kernel
_compute_body_matrices: warp._src.context.Kernel
_comvel_branch: warp._src.context.Kernel
_comvel_root: warp._src.context.Kernel
_count_equality_constraints: warp._src.context.Kernel
_crb_accumulate: warp._src.context.Kernel
_flex_edges: warp._src.context.Kernel
_flex_face_kinematics: warp._src.context.Kernel
_flex_nodes: warp._src.context.Kernel
_flex_vertices: warp._src.context.Kernel
_geom_local_to_global: warp._src.context.Kernel
_joint_tendon: warp._src.context.Kernel
_kinematics_branch: warp._src.context.Kernel
_light_local_to_global: warp._src.context.Kernel
_linear_momentum: warp._src.context.Kernel
_qLD_acc: warp._src.context.Kernel
_qLDiag_div: warp._src.context.Kernel
_qfrc_bias: warp._src.context.Kernel
_site_local_to_global: warp._src.context.Kernel
_small_cholesky_solve: warp._src.context.Function
_spatial_geom_tendon: warp._src.context.Kernel
_spatial_site_tendon: warp._src.context.Kernel
_spatial_tendon_wrap: warp._src.context.Kernel
_subtree_com_acc: warp._src.context.Kernel
_subtree_com_init: warp._src.context.Kernel
_subtree_div: warp._src.context.Kernel
_subtree_vel_forward: warp._src.context.Kernel
_tendon_armature: warp._src.context.Kernel
_tendon_bias_coef: warp._src.context.Kernel
_tendon_bias_qfrc: warp._src.context.Kernel
_tendon_dot: warp._src.context.Kernel
_transmission: warp._src.context.Kernel
_transmission_body_moment: warp._src.context.Kernel
_transmission_body_moment_scale: warp._src.context.Kernel
transform_force: warp._src.context.Function
