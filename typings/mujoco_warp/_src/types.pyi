from __future__ import annotations
import builtins
import dataclasses as dataclasses
import enum as enum
import mujoco as mujoco
import numpy as np
import typing
import warp as wp
import warp._src.context
import warp._src.texture
import warp._src.types
__all__: list[str] = ['BiasType', 'BlockDim', 'BroadphaseFilter', 'BroadphaseType', 'Callback', 'CamLightType', 'CollisionType', 'ConeType', 'Constraint', 'ConstraintState', 'ConstraintType', 'Contact', 'ContactType', 'CtrlChart', 'CtrlInput', 'Data', 'DataType', 'DisableBit', 'DynType', 'EnableBit', 'EqType', 'GainType', 'GeomType', 'IntegratorType', 'InverseContext', 'JointType', 'MJ_MAXCONPAIR', 'MJ_MAXIMP', 'MJ_MAXVAL', 'MJ_MAX_EPAFACES', 'MJ_MAX_EPAHORIZON', 'MJ_MINAWAKE', 'MJ_MINIMP', 'MJ_MINMU', 'MJ_MINVAL', 'M_BLOCK_DENSE_MAX', 'M_BLOCK_SCALAR_MAX', 'Model', 'ObjType', 'Option', 'OverflowType', 'ProjectionType', 'Q_LD_BLOCK_COMPACT', 'Q_LD_BLOCK_SPARSE', 'RenderContext', 'SensorType', 'SleepPolicy', 'SleepState', 'SolverContext', 'SolverType', 'Stage', 'State', 'Statistic', 'TILE_SIZE_JTDAJ_DENSE', 'TILE_SIZE_JTDAJ_SPARSE', 'TileSet', 'TrnType', 'WrapType', 'array', 'dataclasses', 'enum', 'mat23', 'mat23f', 'mat43', 'mat43f', 'mat63', 'mat63f', 'mat66', 'mat66f', 'mujoco', 'np', 'vec10', 'vec10f', 'vec11', 'vec11f', 'vec128', 'vec16', 'vec16f', 'vec16i', 'vec5', 'vec5f', 'vec6', 'vec6f', 'vec6i', 'vec8', 'vec8f', 'vec8i', 'vec_pluginattr', 'wp']
class BlockDim:
    """
    Block dimension 'block_dim' settings for wp.launch_tiled.
    
    TODO(team): experimental and may be removed
    
    Attributes:
      segmented_sort: segmented sort block dimension (collision_driver)
      convex_ccd: convex CCD kernel block dimension (collision_convex)
      actuator_velocity: actuator velocity block dimension (forward)
      island_dsu: island discovery DSU block dimension (island)
      ray: ray block dimension (ray)
      contact_sort: contact sort block dimension (sensor)
      energy_vel_kinetic: energy velocity kinetic block dimension (sensor)
      cholesky_factorize: block-dense Cholesky factorize block dimension (smooth)
      cholesky_factorize_solve: block-dense Cholesky factorize+solve block dimension (smooth)
      cholesky_solve: Cholesky solve block dimension (smooth)
      small_cholesky: scalar small-block Cholesky block dimension (smooth)
      solve_LD_sparse_fused: solve LD sparse fused block dimension (smooth)
      update_gradient_cholesky: update gradient Cholesky block dimension (solver)
      update_gradient_cholesky_blocked: update gradient Cholesky blocked block dimension (solver)
      update_gradient_JTDAJ_sparse: update gradient JTDAJ sparse block dimension (solver)
      update_gradient_JTDAJ_dense: update gradient JTDAJ dense block dimension (solver)
      linesearch_iterative: linesearch iterative block dimension (solver)
      update_gradient_grad: update gradient grad block dimension (solver)
      solve_beta_accumulate: solve beta accumulate block dimension (solver)
      solve_search_update_cg: solve search update CG block dimension (solver)
      solve_init_search_cg: solve init search CG block dimension (solver)
      contact_jac_tiled: contact Jacobian tiled block dimension (solver)
      qderiv_actuator_dense: qderiv actuator dense block dimension (derivative)
      render: render block dimension (render)
    """
    __annotations_cache__: typing.ClassVar[dict] = {'segmented_sort': int, 'convex_ccd': int, 'actuator_velocity': int, 'island_dsu': int, 'ray': int, 'contact_sort': int, 'energy_vel_kinetic': int, 'cholesky_factorize': int, 'cholesky_factorize_solve': int, 'cholesky_solve': int, 'small_cholesky': int, 'solve_LD_sparse_fused': int, 'update_gradient_cholesky': int, 'update_gradient_cholesky_blocked': int, 'update_gradient_JTDAJ_sparse': int, 'update_gradient_JTDAJ_dense': int, 'linesearch_iterative': int, 'update_gradient_grad': int, 'solve_beta_accumulate': int, 'solve_search_update_cg': int, 'solve_init_search_cg': int, 'contact_jac_tiled': int, 'qderiv_actuator_dense': int, 'render': int}
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('segmented_sort', 'convex_ccd', 'actuator_velocity', 'island_dsu', 'ray', 'contact_sort', 'energy_vel_kinetic', 'cholesky_factorize', 'cholesky_factorize_solve', 'cholesky_solve', 'small_cholesky', 'solve_LD_sparse_fused', 'update_gradient_cholesky', 'update_gradient_cholesky_blocked', 'update_gradient_JTDAJ_sparse', 'update_gradient_JTDAJ_dense', 'linesearch_iterative', 'update_gradient_grad', 'solve_beta_accumulate', 'solve_search_update_cg', 'solve_init_search_cg', 'contact_jac_tiled', 'qderiv_actuator_dense', 'render')
    actuator_velocity: typing.ClassVar[int] = 32
    cholesky_factorize: typing.ClassVar[int] = 32
    cholesky_factorize_solve: typing.ClassVar[int] = 32
    cholesky_solve: typing.ClassVar[int] = 64
    contact_jac_tiled: typing.ClassVar[int] = 32
    contact_sort: typing.ClassVar[int] = 64
    convex_ccd: typing.ClassVar[int] = 64
    energy_vel_kinetic: typing.ClassVar[int] = 32
    island_dsu: typing.ClassVar[int] = 32
    linesearch_iterative: typing.ClassVar[int] = 32
    qderiv_actuator_dense: typing.ClassVar[int] = 32
    ray: typing.ClassVar[int] = 64
    render: typing.ClassVar[int] = 64
    segmented_sort: typing.ClassVar[int] = 128
    small_cholesky: typing.ClassVar[int] = 64
    solve_LD_sparse_fused: typing.ClassVar[int] = 128
    solve_beta_accumulate: typing.ClassVar[int] = 256
    solve_init_search_cg: typing.ClassVar[int] = 256
    solve_search_update_cg: typing.ClassVar[int] = 256
    update_gradient_JTDAJ_dense: typing.ClassVar[int] = 128
    update_gradient_JTDAJ_sparse: typing.ClassVar[int] = 128
    update_gradient_cholesky: typing.ClassVar[int] = 64
    update_gradient_cholesky_blocked: typing.ClassVar[int] = 32
    update_gradient_grad: typing.ClassVar[int] = 256
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, segmented_sort: int = 128, convex_ccd: int = 64, actuator_velocity: int = 32, island_dsu: int = 32, ray: int = 64, contact_sort: int = 64, energy_vel_kinetic: int = 32, cholesky_factorize: int = 32, cholesky_factorize_solve: int = 32, cholesky_solve: int = 64, small_cholesky: int = 64, solve_LD_sparse_fused: int = 128, update_gradient_cholesky: int = 64, update_gradient_cholesky_blocked: int = 32, update_gradient_JTDAJ_sparse: int = 128, update_gradient_JTDAJ_dense: int = 128, linesearch_iterative: int = 32, update_gradient_grad: int = 256, solve_beta_accumulate: int = 256, solve_search_update_cg: int = 256, solve_init_search_cg: int = 256, contact_jac_tiled: int = 32, qderiv_actuator_dense: int = 32, render: int = 64) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class BroadphaseType(enum.IntEnum):
    """
    Type of broadphase algorithm.
    
    Attributes:
       NXN: Broad phase checking all pairs
       SAP_TILE: Sweep and prune broad phase using tile sort
       SAP_SEGMENTED: Sweep and prune broad phase using segment sort
    """
    NXN: typing.ClassVar[BroadphaseType]
    SAP_SEGMENTED: typing.ClassVar[BroadphaseType]
    SAP_TILE: typing.ClassVar[BroadphaseType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['NXN', 'SAP_TILE', 'SAP_SEGMENTED']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class BroadphaseFilter(enum.IntFlag):
    """
    Bitmask specifying which collision functions to run during broadphase.
    
    Attributes:
      PLANE: collision between bounding sphere and plane
      SPHERE: collision between bounding spheres
      AABB: collision between axis-aligned bounding boxes
      OBB: collision between oriented bounding boxes
    """
    AABB: typing.ClassVar[BroadphaseFilter]
    OBB: typing.ClassVar[BroadphaseFilter]
    PLANE: typing.ClassVar[BroadphaseFilter]
    SPHERE: typing.ClassVar[BroadphaseFilter]
    _all_bits_: typing.ClassVar[int] = 15
    _boundary_: typing.ClassVar[enum.FlagBoundary]
    _flag_mask_: typing.ClassVar[int] = 15
    _hashable_values_: typing.ClassVar[list] = [1, 2, 4, 8]
    _inverted_ = None
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['PLANE', 'SPHERE', 'AABB', 'OBB']
    _singles_mask_: typing.ClassVar[int] = 15
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
    def _value_repr_(self):
        """
        Return repr(self).
        """
class OverflowType(enum.IntFlag):
    """
    Bitmask for physics and collision overflows.
    
    Attributes:
      NONE: no overflow
      NEFC: nefc > njmax
      NJMAX_NNZ: njmax_nnz overflow
      BROADPHASE: broadphase overflow / flex broadphase overflow
      NARROWPHASE: narrowphase overflow / flex narrowphase / contact overflow
      CCD: CCD overflow / flex CCD overflow
      HFIELD: height field collision overflow
      CONTACT_MATCH: contact match sensor overflow
      NVMAX: nvmax overflow (islands)
      EPA_HORIZON: EPA horizon buffer overflow
      ITERATIONS: solver iteration limit reached
      LS_ITERATIONS: linesearch iteration limit reached
      TACTILE: tactile sensor collision pair overflow
      ALL: all overflows
    """
    ALL: typing.ClassVar[OverflowType]
    BROADPHASE: typing.ClassVar[OverflowType]
    CCD: typing.ClassVar[OverflowType]
    CONTACT_MATCH: typing.ClassVar[OverflowType]
    EPA_HORIZON: typing.ClassVar[OverflowType]
    HFIELD: typing.ClassVar[OverflowType]
    ITERATIONS: typing.ClassVar[OverflowType]
    LS_ITERATIONS: typing.ClassVar[OverflowType]
    NARROWPHASE: typing.ClassVar[OverflowType]
    NEFC: typing.ClassVar[OverflowType]
    NJMAX_NNZ: typing.ClassVar[OverflowType]
    NONE: typing.ClassVar[OverflowType]
    NVMAX: typing.ClassVar[OverflowType]
    TACTILE: typing.ClassVar[OverflowType]
    _all_bits_: typing.ClassVar[int] = 4095
    _boundary_: typing.ClassVar[enum.FlagBoundary]
    _flag_mask_: typing.ClassVar[int] = 4095
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4095]
    _inverted_ = None
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['NEFC', 'NJMAX_NNZ', 'BROADPHASE', 'NARROWPHASE', 'CCD', 'HFIELD', 'CONTACT_MATCH', 'NVMAX', 'EPA_HORIZON', 'ITERATIONS', 'LS_ITERATIONS', 'TACTILE']
    _singles_mask_: typing.ClassVar[int] = 4095
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
    def _value_repr_(self):
        """
        Return repr(self).
        """
class CamLightType(enum.IntEnum):
    """
    Type of camera light.
    
    Attributes:
      FIXED: pos and rot fixed in body
      TRACK: pos tracks body, rot fixed in global
      TRACKCOM: pos tracks subtree com, rot fixed in body
      TARGETBODY: pos fixed in body, rot tracks target body
      TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
    """
    FIXED: typing.ClassVar[CamLightType]
    TARGETBODY: typing.ClassVar[CamLightType]
    TARGETBODYCOM: typing.ClassVar[CamLightType]
    TRACK: typing.ClassVar[CamLightType]
    TRACKCOM: typing.ClassVar[CamLightType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3, 4]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['FIXED', 'TRACK', 'TRACKCOM', 'TARGETBODY', 'TARGETBODYCOM']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class ProjectionType(enum.IntEnum):
    """
    Type of camera projection.
    
    Attributes:
      PERSPECTIVE: perspective projection
      ORTHOGRAPHIC: orthographic projection
    """
    ORTHOGRAPHIC: typing.ClassVar[ProjectionType]
    PERSPECTIVE: typing.ClassVar[ProjectionType]
    _hashable_values_: typing.ClassVar[list] = [0, 1]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['PERSPECTIVE', 'ORTHOGRAPHIC']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class Stage(enum.IntEnum):
    """
    Computation stage.
    
    Attributes:
      POS: position-dependent
      VEL: velocity-dependent
      ACC: acceleration/force-dependent
    """
    ACC: typing.ClassVar[Stage]
    POS: typing.ClassVar[Stage]
    VEL: typing.ClassVar[Stage]
    _hashable_values_: typing.ClassVar[list] = [1, 2, 3]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['POS', 'VEL', 'ACC']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class DataType(enum.IntFlag):
    """
    Sensor data types.
    
    Attributes:
      REAL: real values, no constraints
      POSITIVE: positive values, 0 or negative: inactive
    """
    POSITIVE: typing.ClassVar[DataType]
    REAL: typing.ClassVar[DataType]
    _all_bits_: typing.ClassVar[int] = 1
    _boundary_: typing.ClassVar[enum.FlagBoundary]
    _flag_mask_: typing.ClassVar[int] = 1
    _hashable_values_: typing.ClassVar[list] = [0, 1]
    _inverted_ = None
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['POSITIVE']
    _singles_mask_: typing.ClassVar[int] = 1
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
    def _value_repr_(self):
        """
        Return repr(self).
        """
class DisableBit(enum.IntFlag):
    """
    Disable default feature bitflags.
    
    Attributes:
      CONSTRAINT:   entire constraint solver
      EQUALITY:     equality constraints
      FRICTIONLOSS: joint and tendon frictionloss constraints
      LIMIT:        joint and tendon limit constraints
      CONTACT:      contact constraints
      SPRING:       passive spring forces
      DAMPER:       passive damper forces
      GRAVITY:      gravitational forces
      CLAMPCTRL:    clamp control to specified range
      WARMSTART:    warmstart constraint solver
      FILTERPARENT: disable collisions between parent and child bodies
      ACTUATION:    apply actuation forces
      REFSAFE:      integrator safety: make ref[0]>=2*timestep
      SENSOR:       sensors
      EULERDAMP:    implicit damping for Euler integration
      NATIVECCD:    native convex collision detection (ignored in MJWarp)
      ISLAND:       constraint islands
      MULTICCD:     disable multiple contacts with CCD
    """
    ACTUATION: typing.ClassVar[DisableBit]
    CLAMPCTRL: typing.ClassVar[DisableBit]
    CONSTRAINT: typing.ClassVar[DisableBit]
    CONTACT: typing.ClassVar[DisableBit]
    DAMPER: typing.ClassVar[DisableBit]
    EQUALITY: typing.ClassVar[DisableBit]
    EULERDAMP: typing.ClassVar[DisableBit]
    FILTERPARENT: typing.ClassVar[DisableBit]
    FRICTIONLOSS: typing.ClassVar[DisableBit]
    GRAVITY: typing.ClassVar[DisableBit]
    ISLAND: typing.ClassVar[DisableBit]
    LIMIT: typing.ClassVar[DisableBit]
    MULTICCD: typing.ClassVar[DisableBit]
    NATIVECCD: typing.ClassVar[DisableBit]
    REFSAFE: typing.ClassVar[DisableBit]
    SENSOR: typing.ClassVar[DisableBit]
    SPRING: typing.ClassVar[DisableBit]
    WARMSTART: typing.ClassVar[DisableBit]
    _all_bits_: typing.ClassVar[int] = 1048575
    _boundary_: typing.ClassVar[enum.FlagBoundary]
    _flag_mask_: typing.ClassVar[int] = 966655
    _hashable_values_: typing.ClassVar[list] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768, 131072, 262144, 524288]
    _inverted_ = None
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['CONSTRAINT', 'EQUALITY', 'FRICTIONLOSS', 'LIMIT', 'CONTACT', 'SPRING', 'DAMPER', 'GRAVITY', 'CLAMPCTRL', 'WARMSTART', 'FILTERPARENT', 'ACTUATION', 'REFSAFE', 'SENSOR', 'EULERDAMP', 'NATIVECCD', 'ISLAND', 'MULTICCD']
    _singles_mask_: typing.ClassVar[int] = 966655
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
    def _value_repr_(self):
        """
        Return repr(self).
        """
class EnableBit(enum.IntFlag):
    """
    Enable optional feature bitflags.
    
    Attributes:
      ENERGY: energy computation
      INVDISCRETE: discrete-time inverse dynamics
      SLEEP: sleeping
    """
    ENERGY: typing.ClassVar[EnableBit]
    INVDISCRETE: typing.ClassVar[EnableBit]
    SLEEP: typing.ClassVar[EnableBit]
    _all_bits_: typing.ClassVar[int] = 31
    _boundary_: typing.ClassVar[enum.FlagBoundary]
    _flag_mask_: typing.ClassVar[int] = 26
    _hashable_values_: typing.ClassVar[list] = [2, 8, 16]
    _inverted_ = None
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['ENERGY', 'INVDISCRETE', 'SLEEP']
    _singles_mask_: typing.ClassVar[int] = 26
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
    def _value_repr_(self):
        """
        Return repr(self).
        """
class SleepPolicy(enum.IntEnum):
    """
    Per-tree sleep policy.
    
    Attributes:
      AUTO: compiler chooses sleep policy
      AUTO_NEVER: compiler sleep policy: never
      AUTO_ALLOWED: compiler sleep policy: allowed
    """
    AUTO: typing.ClassVar[SleepPolicy]
    AUTO_ALLOWED: typing.ClassVar[SleepPolicy]
    AUTO_NEVER: typing.ClassVar[SleepPolicy]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['AUTO', 'AUTO_NEVER', 'AUTO_ALLOWED']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class SleepState(enum.IntEnum):
    """
    Sleep state for bodies.
    
    Attributes:
      STATIC: body is static (world body or mocap)
      ASLEEP: body is asleep
      AWAKE: body is awake
    """
    ASLEEP: typing.ClassVar[SleepState]
    AWAKE: typing.ClassVar[SleepState]
    STATIC: typing.ClassVar[SleepState]
    _hashable_values_: typing.ClassVar[list] = [-1, 0, 1]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['STATIC', 'ASLEEP', 'AWAKE']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class TrnType(enum.IntEnum):
    """
    Type of actuator transmission.
    
    Attributes:
      JOINT: force on joint
      JOINTINPARENT: force on joint, expressed in parent frame
      SLIDERCRANK: force via slider-crank linkage
      TENDON: force on tendon
      BODY: adhesion force on body's geoms
      SITE: force on site
    """
    BODY: typing.ClassVar[TrnType]
    JOINT: typing.ClassVar[TrnType]
    JOINTINPARENT: typing.ClassVar[TrnType]
    SITE: typing.ClassVar[TrnType]
    SLIDERCRANK: typing.ClassVar[TrnType]
    TENDON: typing.ClassVar[TrnType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3, 5, 4]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['JOINT', 'JOINTINPARENT', 'SLIDERCRANK', 'TENDON', 'BODY', 'SITE']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class DynType(enum.IntEnum):
    """
    Type of actuator dynamics.
    
    Attributes:
      NONE: no internal dynamics; ctrl specifies force
      INTEGRATOR: integrator: da/dt = u
      FILTER: linear filter: da/dt = (u-a) / tau
      FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
      MUSCLE: piece-wise linear filter with two time constants
      USER: user-defined dynamics via act_dyn_callback
      DCMOTOR: DC motor dynamics
    """
    DCMOTOR: typing.ClassVar[DynType]
    FILTER: typing.ClassVar[DynType]
    FILTEREXACT: typing.ClassVar[DynType]
    INTEGRATOR: typing.ClassVar[DynType]
    MUSCLE: typing.ClassVar[DynType]
    NONE: typing.ClassVar[DynType]
    USER: typing.ClassVar[DynType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3, 4, 7, 5]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['NONE', 'INTEGRATOR', 'FILTER', 'FILTEREXACT', 'MUSCLE', 'USER', 'DCMOTOR']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class GainType(enum.IntEnum):
    """
    Type of actuator gain.
    
    Attributes:
      FIXED: fixed gain
      AFFINE: const + kp*length + kv*velocity
      MUSCLE: muscle FLV curve computed by muscle_gain
      USER: user-defined gain via act_gain_callback
      DCMOTOR: DC motor gain
    """
    AFFINE: typing.ClassVar[GainType]
    DCMOTOR: typing.ClassVar[GainType]
    FIXED: typing.ClassVar[GainType]
    MUSCLE: typing.ClassVar[GainType]
    SO3: typing.ClassVar[GainType]
    USER: typing.ClassVar[GainType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 6, 3, 4]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['FIXED', 'AFFINE', 'MUSCLE', 'USER', 'DCMOTOR', 'SO3']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class BiasType(enum.IntEnum):
    """
    Type of actuator bias.
    
    Attributes:
      NONE: no bias
      AFFINE: const + kp*length + kv*velocity
      MUSCLE: muscle passive force computed by muscle_bias
      USER: user-defined bias via act_bias_callback
      DCMOTOR: DC motor back-EMF bias
    """
    AFFINE: typing.ClassVar[BiasType]
    DCMOTOR: typing.ClassVar[BiasType]
    MUSCLE: typing.ClassVar[BiasType]
    NONE: typing.ClassVar[BiasType]
    USER: typing.ClassVar[BiasType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 5, 3]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['NONE', 'AFFINE', 'MUSCLE', 'USER', 'DCMOTOR']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class CtrlInput(enum.IntFlag):
    """
    Control input signature bitflags for actuators.
    
    Attributes:
      POS: position setpoint input
      VEL: velocity setpoint input
      FF: feedforward input
      VOLTAGE: raw terminal voltage input
      NONE: explicitly no inputs (purely passive)
    """
    FF: typing.ClassVar[CtrlInput]
    NONE: typing.ClassVar[CtrlInput]
    POS: typing.ClassVar[CtrlInput]
    VEL: typing.ClassVar[CtrlInput]
    VOLTAGE: typing.ClassVar[CtrlInput]
    _all_bits_: typing.ClassVar[int] = 31
    _boundary_: typing.ClassVar[enum.FlagBoundary]
    _flag_mask_: typing.ClassVar[int] = 31
    _hashable_values_: typing.ClassVar[list] = [1, 2, 4, 8, 16]
    _inverted_ = None
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['POS', 'VEL', 'FF', 'VOLTAGE', 'NONE']
    _singles_mask_: typing.ClassVar[int] = 31
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
    def _value_repr_(self):
        """
        Return repr(self).
        """
class CtrlChart(enum.IntEnum):
    """
    Orientation chart type for SO(3) actuators.
    
    Attributes:
      EXPMAP: exponential-map orientation target (3 controls)
      QUAT: quaternion orientation target (4 controls)
    """
    EXPMAP: typing.ClassVar[CtrlChart]
    QUAT: typing.ClassVar[CtrlChart]
    _hashable_values_: typing.ClassVar[list] = [1, 2]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['EXPMAP', 'QUAT']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class JointType(enum.IntEnum):
    """
    Type of degree of freedom.
    
    Attributes:
      FREE:  global position and orientation (quat)       (7,)
      BALL:  orientation (quat) relative to parent        (4,)
      SLIDE: sliding distance along body-fixed axis       (1,)
      HINGE: rotation angle (rad) around body-fixed axis  (1,)
    """
    BALL: typing.ClassVar[JointType]
    FREE: typing.ClassVar[JointType]
    HINGE: typing.ClassVar[JointType]
    SLIDE: typing.ClassVar[JointType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['FREE', 'BALL', 'SLIDE', 'HINGE']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class ConeType(enum.IntEnum):
    """
    Type of friction cone.
    
    Attributes:
      PYRAMIDAL: pyramidal
      ELLIPTIC: elliptic
    """
    ELLIPTIC: typing.ClassVar[ConeType]
    PYRAMIDAL: typing.ClassVar[ConeType]
    _hashable_values_: typing.ClassVar[list] = [0, 1]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['PYRAMIDAL', 'ELLIPTIC']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class IntegratorType(enum.IntEnum):
    """
    Integrator mode.
    
    Attributes:
      EULER: semi-implicit Euler
      RK4: 4th-order Runge Kutta
      IMPLICITFAST: implicit in velocity, no rne derivative
      IMPLICIT: implicit in velocity, with rne derivative
    """
    EULER: typing.ClassVar[IntegratorType]
    IMPLICIT: typing.ClassVar[IntegratorType]
    IMPLICITFAST: typing.ClassVar[IntegratorType]
    RK4: typing.ClassVar[IntegratorType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 3, 2]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['EULER', 'RK4', 'IMPLICITFAST', 'IMPLICIT']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class GeomType(enum.IntEnum):
    """
    Type of geometry.
    
    Attributes:
      PLANE: plane
      HFIELD: heightfield
      SPHERE: sphere
      CAPSULE: capsule
      ELLIPSOID: ellipsoid
      CYLINDER: cylinder
      BOX: box
      MESH: mesh
      SDF: sdf
      FLEX: flex
    """
    BOX: typing.ClassVar[GeomType]
    CAPSULE: typing.ClassVar[GeomType]
    CYLINDER: typing.ClassVar[GeomType]
    ELLIPSOID: typing.ClassVar[GeomType]
    FLEX: typing.ClassVar[GeomType]
    HFIELD: typing.ClassVar[GeomType]
    MESH: typing.ClassVar[GeomType]
    PLANE: typing.ClassVar[GeomType]
    SDF: typing.ClassVar[GeomType]
    SPHERE: typing.ClassVar[GeomType]
    TRIANGLE: typing.ClassVar[GeomType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 105, 999]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['PLANE', 'HFIELD', 'SPHERE', 'CAPSULE', 'ELLIPSOID', 'CYLINDER', 'BOX', 'MESH', 'SDF', 'FLEX', 'TRIANGLE']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class CollisionType(enum.IntEnum):
    """
    Type of narrowphase collision.
    
    Attributes:
      PRIMITIVE: primitive collision
      CONVEX: convex collision (CCD)
      SDF: sdf collision
    """
    CONVEX: typing.ClassVar[CollisionType]
    PRIMITIVE: typing.ClassVar[CollisionType]
    SDF: typing.ClassVar[CollisionType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['PRIMITIVE', 'CONVEX', 'SDF']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class SolverType(enum.IntEnum):
    """
    Constraint solver algorithm.
    
    Attributes:
      CG: Conjugate gradient (primal)
      NEWTON: Newton (primal)
    """
    CG: typing.ClassVar[SolverType]
    NEWTON: typing.ClassVar[SolverType]
    _hashable_values_: typing.ClassVar[list] = [1, 2]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['CG', 'NEWTON']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class ConstraintState(enum.IntEnum):
    """
    State of constraint.
    
    Attributes:
      SATISFIED: constraint satisfied, zero cost (limit, contact)
      QUADRATIC: quadratic cost (equality, friction, limit, contact)
      LINEARNEG: linear cost, negative side (friction)
      LINEARPOS: linear cost, positive side (friction)
      CONE: square distance to cone cost (elliptic contact)
    """
    CONE: typing.ClassVar[ConstraintState]
    LINEARNEG: typing.ClassVar[ConstraintState]
    LINEARPOS: typing.ClassVar[ConstraintState]
    QUADRATIC: typing.ClassVar[ConstraintState]
    SATISFIED: typing.ClassVar[ConstraintState]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3, 4]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['SATISFIED', 'QUADRATIC', 'LINEARNEG', 'LINEARPOS', 'CONE']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class ConstraintType(enum.IntEnum):
    """
    Type of constraint.
    
    Attributes:
      EQUALITY: equality constraint
      FRICTION_DOF: dof friction
      FRICTION_TENDON: tendon friction
      LIMIT_JOINT: joint limit
      LIMIT_TENDON: tendon limit
      CONTACT_FRICTIONLESS: frictionless contact
      CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
      CONTACT_ELLIPTIC: frictional contact, elliptic friction cone
    """
    CONTACT_ELLIPTIC: typing.ClassVar[ConstraintType]
    CONTACT_FRICTIONLESS: typing.ClassVar[ConstraintType]
    CONTACT_PYRAMIDAL: typing.ClassVar[ConstraintType]
    EQUALITY: typing.ClassVar[ConstraintType]
    FRICTION_DOF: typing.ClassVar[ConstraintType]
    FRICTION_TENDON: typing.ClassVar[ConstraintType]
    LIMIT_JOINT: typing.ClassVar[ConstraintType]
    LIMIT_TENDON: typing.ClassVar[ConstraintType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3, 4, 5, 6, 7]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['EQUALITY', 'FRICTION_DOF', 'FRICTION_TENDON', 'LIMIT_JOINT', 'LIMIT_TENDON', 'CONTACT_FRICTIONLESS', 'CONTACT_PYRAMIDAL', 'CONTACT_ELLIPTIC']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class SensorType(enum.IntEnum):
    """
    Type of sensor.
    
    Attributes:
      MAGNETOMETER: magnetometer
      CAMPROJECTION: camera projection
      RANGEFINDER: scalar distance to nearest geom or site along z-axis
      JOINTPOS: joint position
      TENDONPOS: scalar tendon position
      ACTUATORPOS: actuator position
      BALLQUAT: ball joint orientation
      JOINTLIMITPOS: joint limit distance-margin
      TENDONLIMITPOS: tendon limit distance-margin
      FRAMEPOS: frame position
      FRAMEXAXIS: frame x-axis
      FRAMEYAXIS: frame y-axis
      FRAMEZAXIS: frame z-axis
      FRAMEQUAT: frame orientation, represented as quaternion
      SUBTREECOM: subtree center of mass
      GEOMDIST: signed distance between two geoms
      GEOMNORMAL: normal direction between two geoms
      GEOMFROMTO: segment between two geoms
      INSIDESITE: 1 if object is inside site, 0 otherwise
      E_POTENTIAL: potential energy
      E_KINETIC: kinetic energy
      CLOCK: simulation time
      VELOCIMETER: 3D linear velocity, in local frame
      GYRO: 3D angular velocity, in local frame
      JOINTVEL: joint velocity
      TENDONVEL: scalar tendon velocity
      ACTUATORVEL: actuator velocity
      BALLANGVEL: ball joint angular velocity
      JOINTLIMITVEL: joint limit velocity
      TENDONLIMITVEL: tendon limit velocity
      FRAMELINVEL: 3D linear velocity
      FRAMEANGVEL: 3D angular velocity
      SUBTREELINVEL: subtree linear velocity
      SUBTREEANGMOM: subtree angular momentum
      TOUCH: scalar contact normal forces summed over sensor zone
      CONTACT: contacts which occurred during the simulation
      ACCELEROMETER: accelerometer
      FORCE: force
      TORQUE: torque
      ACTUATORFRC: scalar actuator force, measured at the joint
      TENDONACTFRC: scalar actuator force, measured at the tendon
      JOINTACTFRC: scalar actuator force, measured at the joint
      JOINTLIMITFRC: joint limit force
      TENDONLIMITFRC: tendon limit force
      FRAMELINACC: 3D linear acceleration
      FRAMEANGACC: 3D angular acceleration
      TACTILE: tactile sensor
      USER: user-defined sensor via sensor_callback
    """
    ACCELEROMETER: typing.ClassVar[SensorType]
    ACTUATORFRC: typing.ClassVar[SensorType]
    ACTUATORPOS: typing.ClassVar[SensorType]
    ACTUATORVEL: typing.ClassVar[SensorType]
    BALLANGVEL: typing.ClassVar[SensorType]
    BALLQUAT: typing.ClassVar[SensorType]
    CAMPROJECTION: typing.ClassVar[SensorType]
    CLOCK: typing.ClassVar[SensorType]
    CONTACT: typing.ClassVar[SensorType]
    E_KINETIC: typing.ClassVar[SensorType]
    E_POTENTIAL: typing.ClassVar[SensorType]
    FORCE: typing.ClassVar[SensorType]
    FRAMEANGACC: typing.ClassVar[SensorType]
    FRAMEANGVEL: typing.ClassVar[SensorType]
    FRAMELINACC: typing.ClassVar[SensorType]
    FRAMELINVEL: typing.ClassVar[SensorType]
    FRAMEPOS: typing.ClassVar[SensorType]
    FRAMEQUAT: typing.ClassVar[SensorType]
    FRAMEXAXIS: typing.ClassVar[SensorType]
    FRAMEYAXIS: typing.ClassVar[SensorType]
    FRAMEZAXIS: typing.ClassVar[SensorType]
    GEOMDIST: typing.ClassVar[SensorType]
    GEOMFROMTO: typing.ClassVar[SensorType]
    GEOMNORMAL: typing.ClassVar[SensorType]
    GYRO: typing.ClassVar[SensorType]
    INSIDESITE: typing.ClassVar[SensorType]
    JOINTACTFRC: typing.ClassVar[SensorType]
    JOINTLIMITFRC: typing.ClassVar[SensorType]
    JOINTLIMITPOS: typing.ClassVar[SensorType]
    JOINTLIMITVEL: typing.ClassVar[SensorType]
    JOINTPOS: typing.ClassVar[SensorType]
    JOINTVEL: typing.ClassVar[SensorType]
    MAGNETOMETER: typing.ClassVar[SensorType]
    RANGEFINDER: typing.ClassVar[SensorType]
    SUBTREEANGMOM: typing.ClassVar[SensorType]
    SUBTREECOM: typing.ClassVar[SensorType]
    SUBTREELINVEL: typing.ClassVar[SensorType]
    TACTILE: typing.ClassVar[SensorType]
    TENDONACTFRC: typing.ClassVar[SensorType]
    TENDONLIMITFRC: typing.ClassVar[SensorType]
    TENDONLIMITPOS: typing.ClassVar[SensorType]
    TENDONLIMITVEL: typing.ClassVar[SensorType]
    TENDONPOS: typing.ClassVar[SensorType]
    TENDONVEL: typing.ClassVar[SensorType]
    TORQUE: typing.ClassVar[SensorType]
    TOUCH: typing.ClassVar[SensorType]
    USER: typing.ClassVar[SensorType]
    VELOCIMETER: typing.ClassVar[SensorType]
    _hashable_values_: typing.ClassVar[list] = [6, 8, 7, 9, 11, 13, 18, 20, 23, 26, 28, 29, 30, 27, 35, 39, 40, 41, 38, 43, 44, 45, 2, 3, 10, 12, 14, 19, 21, 24, 31, 32, 36, 37, 0, 42, 1, 4, 5, 15, 17, 16, 22, 25, 33, 34, 46, 48]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['MAGNETOMETER', 'CAMPROJECTION', 'RANGEFINDER', 'JOINTPOS', 'TENDONPOS', 'ACTUATORPOS', 'BALLQUAT', 'JOINTLIMITPOS', 'TENDONLIMITPOS', 'FRAMEPOS', 'FRAMEXAXIS', 'FRAMEYAXIS', 'FRAMEZAXIS', 'FRAMEQUAT', 'SUBTREECOM', 'GEOMDIST', 'GEOMNORMAL', 'GEOMFROMTO', 'INSIDESITE', 'E_POTENTIAL', 'E_KINETIC', 'CLOCK', 'VELOCIMETER', 'GYRO', 'JOINTVEL', 'TENDONVEL', 'ACTUATORVEL', 'BALLANGVEL', 'JOINTLIMITVEL', 'TENDONLIMITVEL', 'FRAMELINVEL', 'FRAMEANGVEL', 'SUBTREELINVEL', 'SUBTREEANGMOM', 'TOUCH', 'CONTACT', 'ACCELEROMETER', 'FORCE', 'TORQUE', 'ACTUATORFRC', 'TENDONACTFRC', 'JOINTACTFRC', 'JOINTLIMITFRC', 'TENDONLIMITFRC', 'FRAMELINACC', 'FRAMEANGACC', 'TACTILE', 'USER']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class ObjType(enum.IntEnum):
    """
    Type of object.
    
    Attributes:
      UNKNOWN: unknown object type
      BODY: body
      XBODY: body, used to access regular frame instead of i-frame
      GEOM: geom
      FLEX: flex
      SITE: site
      CAMERA: camera
    """
    BODY: typing.ClassVar[ObjType]
    CAMERA: typing.ClassVar[ObjType]
    FLEX: typing.ClassVar[ObjType]
    GEOM: typing.ClassVar[ObjType]
    SITE: typing.ClassVar[ObjType]
    UNKNOWN: typing.ClassVar[ObjType]
    XBODY: typing.ClassVar[ObjType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 5, 9, 6, 7]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['UNKNOWN', 'BODY', 'XBODY', 'GEOM', 'FLEX', 'SITE', 'CAMERA']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class EqType(enum.IntEnum):
    """
    Type of equality constraint.
    
    Attributes:
      CONNECT: connect two bodies at a point (ball joint)
      JOINT: couple the values of two scalar joints with cubic
      WELD: fix relative position and orientation of two bodies
      TENDON: couple the lengths of two tendons with cubic
      FLEX: couple the edge lengths of a flex
      FLEXSTRAIN: strain constraint for interpolated flex
    """
    CONNECT: typing.ClassVar[EqType]
    FLEX: typing.ClassVar[EqType]
    FLEXSTRAIN: typing.ClassVar[EqType]
    JOINT: typing.ClassVar[EqType]
    TENDON: typing.ClassVar[EqType]
    WELD: typing.ClassVar[EqType]
    _hashable_values_: typing.ClassVar[list] = [0, 1, 2, 3, 4, 6]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['CONNECT', 'WELD', 'JOINT', 'TENDON', 'FLEX', 'FLEXSTRAIN']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class WrapType(enum.IntEnum):
    """
    Type of tendon wrapping object.
    
    Attributes:
      JOINT: constant moment arm
      PULLEY: pulley used to split tendon
      SITE: pass through site
      SPHERE: wrap around sphere
      CYLINDER: wrap around (infinite) cylinder
    """
    CYLINDER: typing.ClassVar[WrapType]
    JOINT: typing.ClassVar[WrapType]
    PULLEY: typing.ClassVar[WrapType]
    SITE: typing.ClassVar[WrapType]
    SPHERE: typing.ClassVar[WrapType]
    _hashable_values_: typing.ClassVar[list] = [1, 2, 3, 4, 5]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['JOINT', 'PULLEY', 'SITE', 'SPHERE', 'CYLINDER']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class State(enum.IntEnum):
    """
    State component elements as integer bitflags.
    
    Includes several convenient combinations of these flags.
    
    Attributes:
      TIME: time
      QPOS: position
      QVEL: velocity
      ACT: actuator activation
      HISTORY: delay/interval history buffers
      WARMSTART: acceleration used for warmstart
      CTRL: control
      QFRC_APPLIED: applied generalized force
      XFRC_APPLIED: applied Cartesian force/torque
      EQ_ACTIVE: enable/disable constraints
      MOCAP_POS: positions of mocap bodies
      MOCAP_QUAT: orientations of mocap bodies
      NSTATE: number of state elements
      PHYSICS: TIME | QPOS | QVEL | ACT | HISTORY
      FULLPHYSICS: TIME | PHYSICS | PLUGIN
      USER: CTRL | QFRC_APPLIED | XFRC_APPLIED | EQ_ACTIVE | MOCAP_POS | MOCAP_QUAT | USERDATA
      INTEGRATION: FULLPHYSICS | USER | WARMSTART
    """
    ACT: typing.ClassVar[State]
    CTRL: typing.ClassVar[State]
    EQ_ACTIVE: typing.ClassVar[State]
    FULLPHYSICS: typing.ClassVar[State]
    HISTORY: typing.ClassVar[State]
    INTEGRATION: typing.ClassVar[State]
    MOCAP_POS: typing.ClassVar[State]
    MOCAP_QUAT: typing.ClassVar[State]
    NSTATE: typing.ClassVar[State]
    PHYSICS: typing.ClassVar[State]
    QFRC_APPLIED: typing.ClassVar[State]
    QPOS: typing.ClassVar[State]
    QVEL: typing.ClassVar[State]
    TIME: typing.ClassVar[State]
    USER: typing.ClassVar[State]
    USERDATA: typing.ClassVar[State]
    WARMSTART: typing.ClassVar[State]
    XFRC_APPLIED: typing.ClassVar[State]
    _hashable_values_: typing.ClassVar[list] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 14, 30, 8223, 8128, 16383, 4096]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['TIME', 'QPOS', 'QVEL', 'ACT', 'HISTORY', 'WARMSTART', 'CTRL', 'QFRC_APPLIED', 'XFRC_APPLIED', 'EQ_ACTIVE', 'MOCAP_POS', 'MOCAP_QUAT', 'NSTATE', 'PHYSICS', 'FULLPHYSICS', 'USER', 'INTEGRATION', 'USERDATA']
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class vec5f(warp._src.types.vector.<locals>.vec_t):
    pass
class vec6f(warp._src.types.vector.<locals>.vec_t):
    pass
class vec6i(warp._src.types.vector.<locals>.vec_t):
    pass
class vec8f(warp._src.types.vector.<locals>.vec_t):
    pass
class vec8i(warp._src.types.vector.<locals>.vec_t):
    pass
class vec16f(warp._src.types.vector.<locals>.vec_t):
    pass
class vec16i(warp._src.types.vector.<locals>.vec_t):
    pass
class vec10f(warp._src.types.vector.<locals>.vec_t):
    pass
class vec11f(warp._src.types.vector.<locals>.vec_t):
    pass
class vec_pluginattr(warp._src.types.vector.<locals>.vec_t):
    pass
class mat23f(warp._src.types.matrix.<locals>.mat_t):
    pass
class mat43f(warp._src.types.matrix.<locals>.mat_t):
    pass
class mat63f(warp._src.types.matrix.<locals>.mat_t):
    pass
class mat66f(warp._src.types.matrix.<locals>.mat_t):
    pass
class Option:
    """
    Physics options.
    
    Attributes:
      timestep: simulation timestep
      tolerance: main solver tolerance
      ls_tolerance: CG/Newton linesearch tolerance
      ccd_tolerance: convex collision detection tolerance
      sleep_tolerance: sleep velocity tolerance
      gravity: gravitational acceleration
      wind: wind (for lift, drag, and viscosity)
      magnetic: global magnetic flux
      density: density of medium
      viscosity: viscosity of medium
      integrator: integration mode (IntegratorType)
      cone: type of friction cone (ConeType)
      solver: solver algorithm (SolverType)
      iterations: number of main solver iterations
      ls_iterations: maximum number of CG/Newton linesearch iterations
      ccd_iterations: number of iterations in convex collision detection
      disableflags: bit flags for disabling standard features
      enableflags: bit flags for enabling optional features
      sdf_initpoints: number of starting points for gradient descent
      sdf_iterations: max number of iterations for gradient descent
    
    warp only fields:
      impratio_invsqrt: ratio of friction-to-normal contact impedance (stored as inverse square root)
      broadphase: broadphase type (BroadphaseType)
      broadphase_filter: broadphase filter bitflag (BroadphaseFilter)
      graph_conditional: flag to use cuda graph conditional
      run_collision_detection: if False, skips collision detection and allows user-populated
        contacts during the physics step (as opposed to DisableBit.CONTACT which explicitly
        zeros out the contacts at each step)
      run_rne_postconstraint: if True, evaluates rne_postconstraint after the solver step even if
        sensors are disabled or no sensors are present
      contact_sensor_maxmatch: max number of contacts considered by contact sensor matching criteria
                               contacts matched after this value is exceded will be ignored
      warn_overflow: overflow warning bitmask (OverflowType)
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('timestep', 'tolerance', 'ls_tolerance', 'ccd_tolerance', 'sleep_tolerance', 'gravity', 'wind', 'magnetic', 'density', 'viscosity', 'integrator', 'cone', 'solver', 'iterations', 'ls_iterations', 'ccd_iterations', 'disableflags', 'enableflags', 'sdf_initpoints', 'sdf_iterations', 'impratio_invsqrt', 'broadphase', 'broadphase_filter', 'graph_conditional', 'run_collision_detection', 'run_rne_postconstraint', 'contact_sensor_maxmatch', 'warn_overflow')
    ls_parallel: bool
    ls_parallel_min_step: float
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, timestep: array(ndim=1, dtype=float32), tolerance: array(ndim=1, dtype=float32), ls_tolerance: array(ndim=1, dtype=float32), ccd_tolerance: array(ndim=1, dtype=float32), sleep_tolerance: array(ndim=1, dtype=float32), gravity: array(ndim=1, dtype=vec3f), wind: array(ndim=1, dtype=vec3f), magnetic: array(ndim=1, dtype=vec3f), density: array(ndim=1, dtype=float32), viscosity: array(ndim=1, dtype=float32), integrator: int, cone: int, solver: int, iterations: int, ls_iterations: int, ccd_iterations: int, disableflags: int, enableflags: int, sdf_initpoints: int, sdf_iterations: int, impratio_invsqrt: array(ndim=1, dtype=float32), broadphase: BroadphaseType, broadphase_filter: BroadphaseFilter, graph_conditional: bool, run_collision_detection: bool, run_rne_postconstraint: bool, contact_sensor_maxmatch: int, warn_overflow: int = ...) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
    @property
    def warn_overflow(self) -> int:
        ...
    @warn_overflow.setter
    def warn_overflow(self, value: bool | int):
        ...
class Statistic:
    """
    Model statistics (in qpos0).
    
    Attributes:
      meaninertia: mean diagonal inertia (per-world)
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('meaninertia')
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, meaninertia: array(ndim=1, dtype=float32)) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class TileSet:
    """
    Tiling configuration for decomposable block diagonal matrix.
    
    For non-square, non-block-diagonal tiles, use two tilesets.
    
    Attributes:
      adr: address of each tile in the set
      size: size of all the tiles in this set
      elemid: flat CSR gather indices for tiled blocks; empty selects the native-layout scalar path
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __match_args__: typing.ClassVar[tuple] = ('adr', 'size', 'elemid')
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self, adr: wp.array[wp.int32], size: int, elemid: wp.array[wp.int32] = ...) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class Callback:
    """
    Callbacks for custom physics behavior.
    
    Attributes:
      passive: custom passive forces, writes to ``Data.qfrc_passive``
      control: custom control laws, writes to ``Data.ctrl``
      act_dyn: custom actuator dynamics, writes to ``Data.act_dot``
      act_gain: custom actuator gains, writes to ``Data.actuator_force``
      act_bias: custom actuator biases, writes to ``Data.actuator_force``
      sensor: custom sensors, writes to ``Data.sensordata``
      contactfilter: custom contact filtering, writes to ``Data.contact``
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('passive', 'control', 'act_dyn', 'act_gain', 'act_bias', 'sensor', 'contactfilter')
    act_bias = None
    act_dyn = None
    act_gain = None
    contactfilter = None
    control = None
    passive = None
    sensor = None
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, passive: typing.Callable | None = None, control: typing.Callable | None = None, act_dyn: typing.Callable | None = None, act_gain: typing.Callable | None = None, act_bias: typing.Callable | None = None, sensor: typing.Callable | None = None, contactfilter: typing.Callable | None = None) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class Model:
    """
    Model definition and parameters.
    
    Attributes:
      nq: number of generalized coordinates
      nv: number of degrees of freedom
      nu: number of controls/inputs
      nactuator: number of actuators
      na: number of activation states
      nbody: number of bodies
      noct: number of total octree cells in all meshes
      njnt: number of joints
      ntree: number of kinematic trees
      nM: number of non-zeros in sparse inertia matrix
      nC: number of non-zeros in sparse body-dof matrix
      nD: number of non-zeros in sparse derivative matrix
      ngeom: number of geoms
      nsite: number of sites
      ncam: number of cameras
      nlight: number of lights
      nflex: number of flexes
      nflexnode: number of nodes in all flexes
      nflexvert: number of vertices in all flexes
      nflexedge: number of edges in all flexes
      nflexelem: number of elements in all flexes
      nflexelemdata: number of element vertex ids in all flexes
      nflexstiffness: number of stiffness parameters in all flexes
      nflexbending: number of bending parameters in all flexes
      nflexelemedge: number of element edge ids in all flexes
      nflexshelldata: number of shell fragment vertex ids in all flexes
      nJfe: number of non-zeros in sparse flexedge Jacobian
      nmesh: number of meshes
      nmeshvert: number of vertices for all meshes
      nmeshnormal: number of normals in all meshes
      nmeshface: number of faces for all meshes
      nmeshgraph: number of ints in mesh auxiliary data
      nmeshpoly: number of polygons in all meshes
      nmeshpolyvert: number of vertices in all polygons
      nmeshpolymap: number of polygons in vertex map
      nhfield: number of heightfields
      nhfielddata: size of elevation data
      nmat: number of materials
      npair: number of predefined geom pairs
      nexclude: number of excluded geom pairs
      neq: number of equality constraints
      ntendon: number of tendons
      nJten: number of non-zeros in sparse tendon Jacobian
      nwrap: number of wrap objects in all tendon paths
      nsensor: number of sensors
      nkey: number of keyframes
      nmocap: number of mocap bodies
      nplugin: number of plugin instances
      nJmom: number of non-zeros in actuator_moment
      npolygonmax: maximum number of verts per polygon
      nmeshdegmax: maximum number of polygons per vert
      nuserdata: number of custom user parameters
      nsensordata: number of elements in sensor data vector
      nhistory: number of history buffer entries
      opt: physics options
      stat: model statistics
      qpos0: qpos values at default pose                       (*, nq)
      qpos_spring: reference pose for springs                  (*, nq)
      body_parentid: id of body's parent                       (nbody,)
      body_rootid: id of root above body                       (nbody,)
      body_weldid: id of body that this body is welded to      (nbody,)
      body_mocapid: id of mocap data; -1: none                 (nbody,)
      body_jntnum: number of joints for this body              (nbody,)
      body_jntadr: start addr of joints; -1: no joints         (nbody,)
      body_dofnum: number of motion degrees of freedom         (nbody,)
      body_dofadr: start addr of dofs; -1: no dofs             (nbody,)
      body_treeid: id of body's tree; -1: static               (nbody,)
      body_geomnum: number of geoms                            (nbody,)
      body_geomadr: start addr of geoms; -1: no geoms          (nbody,)
      body_simple: body simple type                            (nbody,)
      body_pos: position offset rel. to parent body            (*, nbody, 3)
      body_quat: orientation offset rel. to parent body        (*, nbody, 4)
      body_ipos: local position of center of mass              (*, nbody, 3)
      body_iquat: local orientation of inertia ellipsoid       (*, nbody, 4)
      body_mass: mass                                          (*, nbody,)
      body_subtreemass: mass of subtree starting at this body  (*, nbody,)
      body_inertia: diagonal inertia in ipos/iquat frame       (*, nbody, 3)
      body_invweight0: mean inv inert in qpos0 (trn, rot)      (*, nbody, 2)
      body_gravcomp: antigravity force, units of body weight   (*, nbody)
      body_contype: OR over all geom contypes                  (nbody,)
      body_conaffinity: OR over all geom conaffinities         (nbody,)
      oct_child: octree children                               (noct, 8)
      oct_aabb: octree axis-aligned bounding boxes             (noct, 2, 3)
      oct_coeff: octree interpolation coefficients             (noct, 8)
      jnt_type: type of joint (JointType)                      (njnt,)
      jnt_qposadr: start addr in 'qpos' for joint's data       (njnt,)
      jnt_dofadr: start addr in 'qvel' for joint's data        (njnt,)
      jnt_bodyid: id of joint's body                           (njnt,)
      jnt_limited: does joint have limits                      (njnt,)
      jnt_actfrclimited: does joint have actuator force limits (njnt,)
      jnt_actgravcomp: is gravcomp force applied via actuators (njnt,)
      jnt_solref: constraint solver reference: limit           (*, njnt, mjNREF)
      jnt_solimp: constraint solver impedance: limit           (*, njnt, mjNIMP)
      jnt_pos: local anchor position                           (*, njnt, 3)
      jnt_axis: local joint axis                               (*, njnt, 3)
      jnt_stiffness: stiffness coefficient                     (*, njnt)
      jnt_stiffnesspoly: high-order stiffness coefficients     (*, njnt, 2)
      jnt_range: joint limits                                  (*, njnt, 2)
      jnt_actfrcrange: range of total actuator force           (*, njnt, 2)
      jnt_margin: min distance for limit detection             (*, njnt)
      dof_bodyid: id of dof's body                             (nv,)
      dof_jntid: id of dof's joint                             (nv,)
      dof_parentid: id of dof's parent; -1: none               (nv,)
      dof_treeid: id of dof's tree                             (nv,)
      dof_Madr: dof address in M-diagonal                      (nv,)
      dof_solref: constraint solver reference: frictionloss    (*, nv, NREF)
      dof_solimp: constraint solver impedance: frictionloss    (*, nv, NIMP)
      dof_frictionloss: dof friction loss                      (*, nv)
      dof_armature: dof armature inertia/mass                  (*, nv)
      dof_damping: damping coefficient                         (*, nv)
      dof_dampingpoly: high-order damping coefficients         (*, nv, 2)
      dof_invweight0: diag. inverse inertia in qpos0           (*, nv)
      dof_length: dof length for weighting velocity norm       (nv,)
      tree_bodynum: number of bodies in tree (incl. root)      (ntree,)
      tree_dofadr: start address of tree's dofs                (ntree,)
      tree_dofnum: number of dofs in tree                      (ntree,)
      tree_sleep_policy: tree sleep policy (SleepPolicy)       (ntree,)
      geom_type: geometric type (GeomType)                     (ngeom,)
      geom_contype: geom contact type                          (ngeom,)
      geom_conaffinity: geom contact affinity                  (ngeom,)
      geom_condim: contact dimensionality (1, 3, 4, 6)         (ngeom,)
      geom_bodyid: id of geom's body                           (ngeom,)
      geom_dataid: id of geom's mesh/hfield; -1: none          (*, ngeom)
      geom_matid: material id for rendering                    (*, ngeom,)
      geom_group: geom group inclusion/exclusion mask          (ngeom,)
      geom_priority: geom contact priority                     (ngeom,)
      geom_solmix: mixing coef for solref/imp in geom pair     (*, ngeom,)
      geom_solref: constraint solver reference: contact        (*, ngeom, mjNREF)
      geom_solimp: constraint solver impedance: contact        (*, ngeom, mjNIMP)
      geom_size: geom-specific size parameters                 (*, ngeom, 3)
      geom_aabb: bounding box, (center, size)                  (*, ngeom, 2, 3)
      geom_rbound: radius of bounding sphere                   (*, ngeom,)
      geom_pos: local position offset rel. to body             (*, ngeom, 3)
      geom_quat: local orientation offset rel. to body         (*, ngeom, 4)
      geom_friction: friction for (slide, spin, roll)          (*, ngeom, 3)
      geom_margin: detect contact if dist<margin               (*, ngeom,)
      geom_gap: additional contact detection buffer            (*, ngeom,)
      geom_surfacevel: surface velocity in local frame: lin,ang(*, ngeom, 6)
      geom_adhesion: adhesive force of contacts                (*, ngeom,)
      geom_fluid: fluid interaction parameters                 (ngeom, mjNFLUID)
      geom_rgba: rgba when material is omitted                 (*, ngeom, 4)
      site_type: geom type for rendering (GeomType)            (nsite,)
      site_bodyid: id of site's body                           (nsite,)
      site_size: geom size for rendering                       (nsite, 3)
      site_pos: local position offset rel. to body             (*, nsite, 3)
      site_quat: local orientation offset rel. to body         (*, nsite, 4)
      cam_mode: camera tracking mode (CamLightType)            (ncam,)
      cam_bodyid: id of camera's body                          (ncam,)
      cam_targetbodyid: id of targeted body; -1: none          (ncam,)
      cam_pos: position rel. to body frame                     (*, ncam, 3)
      cam_quat: orientation rel. to body frame                 (*, ncam, 4)
      cam_poscom0: global position rel. to sub-com in qpos0    (*, ncam, 3)
      cam_pos0: global position rel. to body in qpos0          (*, ncam, 3)
      cam_mat0: global orientation in qpos0                    (*, ncam, 3, 3)
      cam_projection: projection type (ProjectionType)         (ncam,)
      cam_fovy: y field-of-view (ortho ? len : deg)            (*, ncam)
      cam_resolution: resolution: pixels [width, height]       (ncam, 2)
      cam_sensorsize: sensor size: length [width, height]      (ncam, 2)
      cam_intrinsic: [focal length; principal point]           (*, ncam, 4)
      light_mode: light tracking mode (CamLightType)           (nlight,)
      light_bodyid: id of light's body                         (nlight,)
      light_targetbodyid: id of targeted body; -1: none        (nlight,)
      light_type: spot, directional, etc. (mjtLightType)       (*, nlight)
      light_castshadow: does light cast shadows                (*, nlight)
      light_active: is light active                            (*, nlight)
      light_pos: position rel. to body frame                   (*, nlight, 3)
      light_dir: direction rel. to body frame                  (*, nlight, 3)
      light_poscom0: global position rel. to sub-com in qpos0  (*, nlight, 3)
      light_pos0: global position rel. to body in qpos0        (*, nlight, 3)
      light_dir0: global direction in qpos0                    (*, nlight, 3)
      light_attenuation: OpenGL constant/linear/quadratic      (*, nlight, 3)
      light_cutoff: spotlight half-cone angle in degrees       (*, nlight)
      light_exponent: spotlight angular falloff exponent       (*, nlight)
      light_ambient: ambient RGB                               (*, nlight, 3)
      light_diffuse: diffuse RGB                               (*, nlight, 3)
      light_specular: specular RGB                             (*, nlight, 3)
      flex_contype: flex contact type                          (nflex,)
      flex_conaffinity: flex contact affinity                  (nflex,)
      flex_condim: contact dimensionality (1, 3, 4, 6)         (nflex,)
      flex_priority: geom contact priority                     (nflex,)
      flex_solmix: mixing coef for solref/imp in geom pair     (nflex,)
      flex_solref: constraint solver reference: contact        (nflex, mjNREF)
      flex_solimp: constraint solver impedance: contact        (nflex, mjNIMP)
      flex_friction: friction for (slide, spin, roll)          (nflex, 3)
      flex_margin: detect contact if dist<margin               (nflex,)
      flex_gap: include in solver if dist<margin-gap           (nflex,)
      flex_selfcollide: self-collision mode                    (nflex,)
      flex_activelayers: active element layers                 (nflex,)
      flex_dim: 1: lines, 2: triangles, 3: tetrahedra          (nflex,)
      flex_interp: interpolation order (0: vertex, 1+: nodes)  (nflex,)
      flex_cellnum: cell count per dimension                   (nflex, 3)
      flex_nodeadr: first node address                         (nflex,)
      flex_nodenum: number of nodes                            (nflex,)
      flex_vertadr: first vertex address                       (nflex,)
      flex_vertnum: number of vertices                         (nflex,)
      flex_edgeadr: first edge address                         (nflex,)
      flex_edgenum: number of edges                            (nflex,)
      flex_elemadr: first element address                      (nflex,)
      flex_elemnum: number of elements                         (nflex,)
      flex_elemdataadr: first element vertex id address        (nflex,)
      flex_stiffnessadr: stiffness matrix address              (nflex,)
      flex_elemedgeadr: first element edge id address          (nflex,)
      flex_bendingadr: first bending data address              (nflex,)
      flex_shellnum: number of shells                          (nflex,)
      flex_shelldataadr: first shell data address              (nflex,)
      flex_nodebodyid: node body ids                           (nflexnode,)
      flex_vertbodyid: vertex body ids                         (nflexvert,)
      flex_edge: edge vertex ids (2 per edge)                  (nflexedge, 2)
      flex_edgeflap: adjacent vertex ids (dim=2 only)          (nflexedge, 2)
      flex_elem: element vertex ids (dim+1 per elem)           (nflexelemdata,)
      flex_elemedge: element edge ids                          (nflexelemedge,)
      flex_elemlayer: element distance from surface            (nflexelem,)
      flex_shell: shell fragment vertex ids (dim per frag)     (nflexshelldata,)
      flex_vert: vertex local positions                        (nflexvert, 3)
      flex_vert0: reference vertex positions in qpos0          (nflexvert, 3)
      flex_node: node local positions                          (nflexnode, 3)
      flex_node0: reference node positions in qpos0            (nflexnode, 3)
      flexedge_length0: edge lengths in qpos0                  (nflexedge,)
      flexedge_invweight0: inv. inertia for the edge           (nflexedge,)
      flex_radius: radius around primitive element             (nflex,)
      flex_stiffness: finite element stiffness matrix          (nflexstiffness,)
      flex_bending: bending stiffness                          (nflexbending,)
      flex_damping: Rayleigh's damping coefficient             (nflex,)
      flex_edgeequality: edge equality type (0:none,1:edge,2:vert,3:strain) (nflex,)
      flex_centered: flex vertices are centered at body origin (nflex,)
      flexedge_J_rownnz: number of nonzeros in Jacobian row    (nflexedge,)
      flexedge_J_rowadr: row start address in colind array     (nflexedge,)
      flexedge_J_colind: column indices in sparse Jacobian     (nJfe,)
      mesh_vertadr: first vertex address                       (nmesh,)
      mesh_vertnum: number of vertices                         (nmesh,)
      mesh_faceadr: first face address                         (nmesh,)
      mesh_octadr: octree address for each mesh                (nmesh,)
      mesh_normaladr: first normal address                     (nmesh,)
      mesh_normalnum: number of normals                        (nmesh,)
      mesh_graphadr: graph data address; -1: no graph          (nmesh,)
      mesh_vert: vertex positions for all meshes               (nmeshvert, 3)
      mesh_normal: normals for all meshes                      (nmeshnormal, 3)
      mesh_face: face indices for all meshes                   (nface, 3)
      mesh_graph: convex graph data                            (nmeshgraph,)
      mesh_pos: translation applied to asset vertices          (nmesh, 3)
      mesh_quat: rotation applied to asset vertices            (nmesh, 4)
      mesh_polynum: number of polygons per mesh                (nmesh,)
      mesh_polyadr: first polygon address per mesh             (nmesh,)
      mesh_polynormal: all polygon normals                     (nmeshpoly, 3)
      mesh_polyvertadr: polygon vertex start address           (nmeshpoly,)
      mesh_polyvertnum: number of vertices per polygon         (nmeshpoly,)
      mesh_polyvert: all polygon vertices                      (nmeshpolyvert,)
      mesh_polymapadr: first polygon address per vertex        (nmeshvert,)
      mesh_polymapnum: number of polygons per vertex           (nmeshvert,)
      mesh_polymap: vertex to polygon map                      (nmeshpolymap,)
      hfield_size: (x, y, z_top, z_bottom)                     (nhfield, 4)
      hfield_nrow: number of rows in grid                      (nhfield,)
      hfield_ncol: number of columns in grid                   (nhfield,)
      hfield_adr: start address in hfield_data                 (nhfield,)
      hfield_data: elevation data                              (nhfielddata,)
      mat_texid: texture id for rendering                      (*, nmat, mjNTEXROLE)
      mat_texuniform: texture uniform flag (spatial scaling)   (*, nmat)
      mat_texrepeat: texture repeat for rendering              (*, nmat, 2)
      mat_emission: emission scalar (self-illumination)        (*, nmat)
      mat_specular: specular reflection scalar                 (*, nmat)
      mat_shininess: shininess in [0, 1], mapped to GL [0, 128](*, nmat)
      mat_rgba: rgba                                           (*, nmat, 4)
      pair_dim: contact dimensionality                         (npair,)
      pair_geom1: id of geom1                                  (npair,)
      pair_geom2: id of geom2                                  (npair,)
      pair_solref: solver reference: contact normal            (*, npair, mjNREF)
      pair_solreffriction: solver reference: contact friction  (*, npair, mjNREF)
      pair_solimp: solver impedance: contact                   (*, npair, mjNIMP)
      pair_margin: detect contact if dist<margin               (*, npair,)
      pair_gap: additional contact detection buffer            (*, npair,)
      pair_adhesion: adhesive force of contacts                (*, npair,)
      pair_friction: tangent1, 2, spin, roll1, 2               (*, npair, 5)
      exclude_signature: body1 << 16 + body2                   (nexclude,)
      eq_type: constraint type (EqType)                        (neq,)
      eq_obj1id: id of object 1                                (neq,)
      eq_obj2id: id of object 2                                (neq,)
      eq_objtype: type of both objects (ObjType)               (neq,)
      eq_active0: initial enable/disable constraint state      (neq,)
      eq_solref: constraint solver reference                   (*, neq, mjNREF)
      eq_solimp: constraint solver impedance                   (*, neq, mjNIMP)
      eq_data: numeric data for constraint                     (*, neq, mjNEQDATA)
      tendon_adr: address of first object in tendon's path     (ntendon,)
      tendon_num: number of objects in tendon's path           (ntendon,)
      ten_J_rownnz: number of non-zeros in each tendon row     (ntendon,)
      ten_J_rowadr: row start address for sparse ten_J         (ntendon,)
      ten_J_colind: column indices in sparse ten_J             (nJten,)
      tendon_limited: does tendon have length limits           (ntendon,)
      tendon_actfrclimited: does ten have actuator force limit (ntendon,)
      tendon_solref_lim: constraint solver reference: limit    (*, ntendon, mjNREF)
      tendon_solimp_lim: constraint solver impedance: limit    (*, ntendon, mjNIMP)
      tendon_solref_fri: constraint solver reference: friction (*, ntendon, mjNREF)
      tendon_solimp_fri: constraint solver impedance: friction (*, ntendon, mjNIMP)
      tendon_range: tendon length limits                       (*, ntendon, 2)
      tendon_actfrcrange: range of total actuator force        (*, ntendon, 2)
      tendon_margin: min distance for limit detection          (*, ntendon)
      tendon_stiffness: stiffness coefficient                  (*, ntendon)
      tendon_stiffnesspoly: high-order stiffness coefficients  (*, ntendon, 2)
      tendon_damping: damping coefficient                      (*, ntendon)
      tendon_dampingpoly: high-order damping coefficients      (*, ntendon, 2)
      tendon_armature: inertia associated with tendon velocity (*, ntendon)
      tendon_frictionloss: loss due to friction                (*, ntendon)
      tendon_lengthspring: spring resting length range         (*, ntendon, 2)
      tendon_length0: tendon length in qpos0                   (*, ntendon)
      tendon_invweight0: inv. weight in qpos0                  (*, ntendon)
      wrap_type: wrap object type (WrapType)                   (nwrap,)
      wrap_objid: object id: geom, site, joint                 (nwrap,)
      wrap_prm: divisor, joint coef, or site id                (nwrap,)
      actuator_trntype: transmission type (TrnType)            (nactuator,)
      actuator_dyntype: dynamics type (DynType)                (nactuator,)
      actuator_gaintype: gain type (GainType)                  (nactuator,)
      actuator_biastype: bias type (BiasType)                  (nactuator,)
      actuator_ctrladr: first control address; -1: none        (nactuator,)
      actuator_ctrlnum: number of control variables            (nactuator,)
      actuator_ctrlspec: input specification bitmask           (nactuator,)
      actuator_actadr: first activation address; -1: stateless (nactuator,)
      actuator_actnum: number of activation variables          (nactuator,)
      actuator_trnid: transmission id: joint, tendon, site     (nactuator, 2)
      actuator_cranklength: crank length for slider-crank      (*, nactuator)
      actuator_dynprm: dynamics parameters                     (*, nactuator, mjNDYN)
      actuator_gainprm: gain parameters                        (*, nactuator, mjNGAIN)
      actuator_biasprm: bias parameters                        (*, nactuator, mjNBIAS)
      actuator_actlimited: is activation limited               (nactuator,)
      actuator_actrange: range of activations                  (*, nactuator, 2)
      actuator_actearly: step activation before force          (nactuator,)
      actuator_history: history buffer sizes                   (*, nactuator, 2)
      actuator_historyadr: history buffer address              (*, nactuator)
      actuator_delay: delay in seconds                         (*, nactuator)
      actuator_forcelimited: is force limited                  (nactuator,)
      actuator_forcerange: range of forces                     (*, nactuator, 2)
      actuator_ctrllimited: is control limited                 (nu,)
      actuator_ctrlrange: range of controls                    (*, nu, 2)
      actuator_gear: scale length and transmitted force        (*, nactuator, 6)
      actuator_acc0: acceleration from unit force in qpos0     (*, nactuator)
      actuator_lengthrange: feasible actuator length range     (*, nu, 2)
      sensor_type: sensor type (SensorType)                    (nsensor,)
      sensor_datatype: numeric data type (DataType)            (nsensor,)
      sensor_objtype: type of sensorized object (ObjType)      (nsensor,)
      sensor_objid: id of sensorized object                    (nsensor,)
      sensor_reftype: type of reference frame (ObjType)        (nsensor,)
      sensor_refid: id of reference frame; -1: global frame    (nsensor,)
      sensor_intprm: sensor parameters                         (nsensor, mjNSENS)
      sensor_dim: number of scalar outputs                     (nsensor,)
      sensor_adr: address in sensor array                      (nsensor,)
      sensor_cutoff: cutoff for real and positive; 0: ignore   (nsensor,)
      sensor_history: history buffer sizes                     (*, nsensor, 2)
      sensor_historyadr: history buffer address                (*, nsensor)
      sensor_delay: delay in seconds                           (*, nsensor)
      sensor_interval: sensor interval and phase               (*, nsensor, 2)
      plugin: globally registered plugin slot number           (nplugin,)
      plugin_attr: config attributes of geom plugin            (nplugin, _NPLUGINATTR)
      key_time: keyframe time                                  (nkey,)
      key_qpos: keyframe qpos                                  (nkey, nq)
      key_qvel: keyframe qvel                                  (nkey, nv)
      key_act: keyframe act                                    (nkey, na)
      key_mpos: keyframe mocap pos                             (nkey, nmocap, 3)
      key_mquat: keyframe mocap quat                           (nkey, nmocap, 4)
      key_ctrl: keyframe ctrl                                  (nkey, nu)
      M_rownnz: number of non-zeros in each row of M           (nv,)
      M_rowadr: index of each row in M                         (nv,)
      M_colind: column indices of non-zeros in M               (nC,)
      mapM2M: index mapping from M (legacy) to M (CSR)         (nC)
      D_rownnz: non-zeros per row in D-structure               (nv,)
      D_rowadr: row start addresses in D-structure             (nv,)
      D_diag: diagonal element index within each row           (nv,)
      D_colind: column indices in D-structure                  (nD,)
      mapM2D: index mapping from M to D                        (nD,)
      mapD2M: index mapping from D to M                        (nC,)
    
    warp only fields:
      callback: custom physics callbacks
      nbranch: number of branches (leaf-to-root paths)
      nv_pad: number of degrees of freedom + padding
      nacttrnbody: number of actuators with body transmission
      nsensorcollision: number of unique collisions for
                        geom distance sensors
      nsensortaxel: number of taxels in all tactile sensors
      ntactileweld: number of unique weld bodies with tactile sensors
      nsensorcontact: number of contact sensors
      nrangefinder: number of rangefinder sensors
      nmaxcondim: maximum condim across geoms, pairs, and flexes
      nmaxpyramid: maximum number of pyramid directions
      nflexintcell: total interp cells (non-strain) for passive forces
      is_sparse: constraint Jacobian/Hessian layout (sparse vs dense). Does not affect M, whose
        factorization is a per-block decision -- see M_tiles and m_block_layout
      qLD_block_total: packed length of the dense region per world (also the offset of the LDL region)
      qLD_block_adr: packed factor offset; Q_LD_BLOCK_* sentinel otherwise (nv,)
      flg_adhesion: flag indicating if model has passive adhesion
      has_fluid: True if wind, density, or viscosity are non-zero at put_model time
      flg_surfacevel: whether model has non-zero surfacevel
      has_sdf_geom: whether the model contains SDF geoms
      has_flex_selfcollide: whether any flex has self-collision enabled
      has_ellipsoid_geom: whether the model contains ellipsoid geoms
      has_plane_geom: whether the model contains plane geoms
      has_1d_flex: whether the model contains 1D flexes
      has_2d_flex: whether the model contains 2D flexes
      has_3d_flex: whether the model contains 3D flexes
      max_flex_dim: maximum flex dimension in the model
      block_dim: block dim options
      body_tree: list of body ids by tree level
      body_branches: flattened body ids for all branches
      body_branch_start: start index in body_branches for each branch   (nbranch + 1,)
      mocap_bodyid: id of body for mocap                       (nmocap,)
      body_fluid_ellipsoid: does body use ellipsoid fluid      (nbody,)
      body_is_free: is body a standalone free body             (nbody,)
      body_fluid_ellipsoid_adr: body ids with ellipsoid fluid  (nbody_fluid_ellipsoid,)
      body_fluid_box_adr: body ids with box fluid              (nbody_fluid_box,)
      body_freeadr: body ids of free bodies                    (nbodyfree,)
      jnt_limited_slide_hinge_adr: limited/slide/hinge jntadr
      jnt_limited_ball_adr: limited/ball jntadr
      body_isdofancestor: precomputed mask of which DOFs affect each body
      dof_tri_row: dof upper triangle row (used in solver)
      dof_tri_col: dof upper triangle col (used in solver)
      nxn_geom_pair: collision pair geom ids [-2, ngeom-1]
      nxn_geom_pair_filtered: valid collision pair geom ids
                              [-1, ngeom - 1]
      nxn_pairid: contact pair id, -1 if not predefined,
                    -2 if skipped
                  collision id, else -1
      nxn_pairid_filtered: active subset of nxn_pairid
      geom_pair_type_count: count of max number of each
                            potential collision
      geom_plugin_index: geom index in plugin array            (ngeom,)
      eq_connect_adr: eq_* addresses of type `CONNECT`
      eq_wld_adr: eq_* addresses of type `WELD`
      eq_jnt_adr: eq_* addresses of type `JOINT`
      eq_ten_adr: eq_* addresses of type `TENDON`
      eq_flex_adr: eq * addresses of type `FLEX
      eq_flexstrain_adr: eq_* addresses of type `FLEXSTRAIN`
      tendon_jnt_adr: joint tendon address
      tendon_site_pair_adr: site pair tendon address
      tendon_geom_adr: geom tendon address
      tendon_limited_adr: addresses for limited tendons
      max_ten_J_rownnz: maximum number of non-zeros in a tendon row
      ten_wrapadr_site: wrap object starting address for sites
      ten_wrapnum_site: number of site wrap objects per tendon
      wrap_jnt_adr: addresses for joint tendon wrap object
      wrap_site_adr: addresses for site tendon wrap object
      wrap_site_pair_adr: first address for site wrap pair
      wrap_geom_adr: addresses for geom tendon wrap object
      wrap_pulley_scale: pulley scaling                        (nwrap,)
      actuator_trntype_body_adr: addresses for actuators
                                 with body transmission
      sensor_pos_adr: addresses for position sensors
      sensor_limitpos_adr: address for limit position sensors
      sensor_vel_adr: addresses for velocity sensors
                      (excluding limit velocity sensors)
      sensor_limitvel_adr: address for limit velocity sensors
      sensor_acc_adr: addresses for acceleration sensors
      sensor_rangefinder_adr: addresses for rangefinder sensors
      rangefinder_sensor_adr: map sensor id to rangefinder id
                      (excluding touch sensors)
                      (excluding limit force sensors)
      sensor_collision_start_adr: address for sensor's first
                                  item in collision
      collision_sensor_adr: map sensor id to collision id      (nsensor,)
      sensor_touch_adr: addresses for touch sensors
      sensor_limitfrc_adr: address for limit force sensors
      sensor_e_potential: evaluate energy_pos
      sensor_e_kinetic: evaluate energy_vel
      sensor_tendonactfrc_adr: address for tendonactfrc sensor
      sensor_subtree_vel: evaluate subtree_vel
      sensor_contact_adr: addresses for contact sensors        (nsensorcontact,)
      sensor_adr_to_contact_adr: map sensor adr to contact adr (nsensor,)
      sensor_rne_postconstraint: evaluate rne_postconstraint
      sensor_rangefinder_bodyid: bodyid for rangefinder        (nrangefinder,)
      weld_tactile_id: weld body to tactile weld index         (nbody,)
      taxel_vertadr: tactile sensor vertex address             (nsensortaxel,)
      taxel_sensorid: address for tactile sensors
      M_tiles: scalar and tiled block-factorization groups
      qLD_updates: sparse factor updates grouped by tree level
      qLD_all_updates: tuple of all levels concatenated
      qLD_level_offsets: tuple of start offsets for each level
      M_fullm_i: sparse mass matrix addressing
      M_fullm_j: sparse mass matrix addressing
      M_elemid: (row, col) -> CSR madr addresses; -1 if not a chain ancestor
      M_hinit_i: row index of each CSR M entry; for densifying M into the dense Newton H (nC,)
      M_fullm_upper_i: upper-triangle row indices for solver h seeding
      M_fullm_upper_j: upper-triangle column indices for solver h seeding
      M_fullm_upper_elemid: source elemid into M_fullm_i/M_fullm_j
      qD_fullm_i: D-structure row indices for RNE derivatives
      qD_fullm_j: D-structure column indices for RNE derivatives
      M_mulm_rowadr: sparse matmul row pointers
      M_mulm_col: sparse matmul column indices
      M_mulm_madr: sparse matmul matrix addresses
      flex_elemflexid: maps each element index directly to its flexid         (nflexelem,)
      flex_shellflexid: maps each shell index directly to its flexid          (nflexshelldata,)
      flex_vertflexid: maps each vertex index directly to its flexid          (nflexvert,)
      flex_shelladr: maps each flex to its start shell index                  (nflex,)
      flex_faceadr: maps each flex to its start face index                    (nflex,)
      flex_cell_map: precomputed flex cell mapping (nflexintcell,)
      flexstrain_J_rownnz: number of nonzeros in flex strain Jacobian row     (neq_flexstrain,)
      flexstrain_J_rowadr: row start address in colind array (neq_flexstrain,)
      flexstrain_J_colind: column indices in sparse flex strain Jacobian      (nJfs,)
      neq_flexstrain: number of flex strain equality constraints
      nJfs: number of non-zeros in sparse flex strain Jacobian
      nflexbend_interp: number of interpolated bending edges
      flex_bend_interp_map: mapping of interpolated bending edges to flex and (nflexbend_interp, 2)
                            local edge indices
      nflexface: number of interpolated flex shell faces
      flex_face_map: mapping of face index to flex and local element face indices
      flex_face: global node indices of each face                             (nflexface, 9)
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('nq', 'nv', 'nu', 'nactuator', 'na', 'nbody', 'noct', 'njnt', 'ntree', 'nM', 'nC', 'nD', 'ngeom', 'nsite', 'ncam', 'nlight', 'nflex', 'nflexnode', 'nflexvert', 'nflexedge', 'nflexelem', 'nflexelemdata', 'nflexstiffness', 'nflexbending', 'nflexelemedge', 'nflexshelldata', 'nJfe', 'nmesh', 'nmeshvert', 'nmeshnormal', 'nmeshface', 'nmeshgraph', 'nmeshpoly', 'nmeshpolyvert', 'nmeshpolymap', 'nhfield', 'nhfielddata', 'nmat', 'npair', 'nexclude', 'neq', 'ntendon', 'nJten', 'nwrap', 'nsensor', 'nkey', 'nmocap', 'nplugin', 'nJmom', 'npolygonmax', 'nmeshdegmax', 'nuserdata', 'nsensordata', 'nhistory', 'opt', 'stat', 'qpos0', 'qpos_spring', 'body_parentid', 'body_rootid', 'body_weldid', 'body_mocapid', 'body_jntnum', 'body_jntadr', 'body_dofnum', 'body_dofadr', 'body_treeid', 'body_geomnum', 'body_geomadr', 'body_simple', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass', 'body_subtreemass', 'body_inertia', 'body_invweight0', 'body_gravcomp', 'body_contype', 'body_conaffinity', 'oct_child', 'oct_aabb', 'oct_coeff', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_bodyid', 'jnt_limited', 'jnt_actfrclimited', 'jnt_actgravcomp', 'jnt_solref', 'jnt_solimp', 'jnt_pos', 'jnt_axis', 'jnt_stiffness', 'jnt_stiffnesspoly', 'jnt_range', 'jnt_actfrcrange', 'jnt_margin', 'dof_bodyid', 'dof_jntid', 'dof_parentid', 'dof_treeid', 'dof_Madr', 'dof_solref', 'dof_solimp', 'dof_frictionloss', 'dof_armature', 'dof_damping', 'dof_dampingpoly', 'dof_invweight0', 'dof_length', 'tree_bodynum', 'tree_dofadr', 'tree_dofnum', 'tree_sleep_policy', 'geom_type', 'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_bodyid', 'geom_dataid', 'geom_matid', 'geom_group', 'geom_priority', 'geom_solmix', 'geom_solref', 'geom_solimp', 'geom_size', 'geom_aabb', 'geom_rbound', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_margin', 'geom_gap', 'geom_surfacevel', 'geom_adhesion', 'geom_fluid', 'geom_rgba', 'site_type', 'site_bodyid', 'site_size', 'site_pos', 'site_quat', 'cam_mode', 'cam_bodyid', 'cam_targetbodyid', 'cam_pos', 'cam_quat', 'cam_poscom0', 'cam_pos0', 'cam_mat0', 'cam_projection', 'cam_fovy', 'cam_resolution', 'cam_sensorsize', 'cam_intrinsic', 'light_mode', 'light_bodyid', 'light_targetbodyid', 'light_type', 'light_castshadow', 'light_active', 'light_pos', 'light_dir', 'light_poscom0', 'light_pos0', 'light_dir0', 'light_attenuation', 'light_cutoff', 'light_exponent', 'light_ambient', 'light_diffuse', 'light_specular', 'flex_contype', 'flex_conaffinity', 'flex_condim', 'flex_priority', 'flex_solmix', 'flex_solref', 'flex_solimp', 'flex_friction', 'flex_margin', 'flex_gap', 'flex_selfcollide', 'flex_activelayers', 'flex_dim', 'flex_interp', 'flex_cellnum', 'flex_nodeadr', 'flex_nodenum', 'flex_vertadr', 'flex_vertnum', 'flex_edgeadr', 'flex_edgenum', 'flex_elemadr', 'flex_elemnum', 'flex_elemdataadr', 'flex_stiffnessadr', 'flex_elemedgeadr', 'flex_bendingadr', 'flex_shellnum', 'flex_shelldataadr', 'flex_nodebodyid', 'flex_vertbodyid', 'flex_edge', 'flex_edgeflap', 'flex_elem', 'flex_elemedge', 'flex_elemlayer', 'flex_shell', 'flex_vert', 'flex_vert0', 'flex_node', 'flex_node0', 'flexedge_length0', 'flexedge_invweight0', 'flex_radius', 'flex_stiffness', 'flex_bending', 'flex_damping', 'flex_edgeequality', 'flex_centered', 'flexedge_J_rownnz', 'flexedge_J_rowadr', 'flexedge_J_colind', 'mesh_vertadr', 'mesh_vertnum', 'mesh_faceadr', 'mesh_octadr', 'mesh_normaladr', 'mesh_normalnum', 'mesh_graphadr', 'mesh_vert', 'mesh_normal', 'mesh_face', 'mesh_graph', 'mesh_pos', 'mesh_quat', 'mesh_polynum', 'mesh_polyadr', 'mesh_polynormal', 'mesh_polyvertadr', 'mesh_polyvertnum', 'mesh_polyvert', 'mesh_polymapadr', 'mesh_polymapnum', 'mesh_polymap', 'hfield_size', 'hfield_nrow', 'hfield_ncol', 'hfield_adr', 'hfield_data', 'mat_texid', 'mat_texuniform', 'mat_texrepeat', 'mat_emission', 'mat_specular', 'mat_shininess', 'mat_rgba', 'pair_dim', 'pair_geom1', 'pair_geom2', 'pair_solref', 'pair_solreffriction', 'pair_solimp', 'pair_margin', 'pair_gap', 'pair_adhesion', 'pair_friction', 'exclude_signature', 'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_objtype', 'eq_active0', 'eq_solref', 'eq_solimp', 'eq_data', 'tendon_adr', 'tendon_num', 'ten_J_rownnz', 'ten_J_rowadr', 'ten_J_colind', 'tendon_limited', 'tendon_actfrclimited', 'tendon_solref_lim', 'tendon_solimp_lim', 'tendon_solref_fri', 'tendon_solimp_fri', 'tendon_range', 'tendon_actfrcrange', 'tendon_margin', 'tendon_stiffness', 'tendon_stiffnesspoly', 'tendon_damping', 'tendon_dampingpoly', 'tendon_armature', 'tendon_frictionloss', 'tendon_lengthspring', 'tendon_length0', 'tendon_invweight0', 'wrap_type', 'wrap_objid', 'wrap_prm', 'actuator_trntype', 'actuator_dyntype', 'actuator_gaintype', 'actuator_biastype', 'actuator_ctrladr', 'actuator_ctrlnum', 'actuator_ctrlspec', 'actuator_actadr', 'actuator_actnum', 'actuator_trnid', 'actuator_cranklength', 'actuator_dynprm', 'actuator_gainprm', 'actuator_biasprm', 'actuator_actlimited', 'actuator_actrange', 'actuator_actearly', 'actuator_history', 'actuator_historyadr', 'actuator_delay', 'actuator_forcelimited', 'actuator_forcerange', 'actuator_ctrllimited', 'actuator_ctrlrange', 'actuator_gear', 'actuator_acc0', 'actuator_lengthrange', 'sensor_type', 'sensor_datatype', 'sensor_objtype', 'sensor_objid', 'sensor_reftype', 'sensor_refid', 'sensor_intprm', 'sensor_dim', 'sensor_adr', 'sensor_cutoff', 'sensor_history', 'sensor_historyadr', 'sensor_delay', 'sensor_interval', 'plugin', 'plugin_attr', 'key_time', 'key_qpos', 'key_qvel', 'key_act', 'key_mpos', 'key_mquat', 'key_ctrl', 'M_rownnz', 'M_rowadr', 'M_colind', 'mapM2M', 'D_rownnz', 'D_rowadr', 'D_diag', 'D_colind', 'mapM2D', 'mapD2M', 'callback', 'nbranch', 'nv_pad', 'nacttrnbody', 'nsensorcollision', 'nsensortaxel', 'ntactileweld', 'nsensorcontact', 'nrangefinder', 'nmaxcondim', 'nmaxpyramid', 'nflexintcell', 'is_sparse', 'qLD_block_total', 'qLD_block_adr', 'flg_adhesion', 'has_fluid', 'flg_surfacevel', 'has_sdf_geom', 'has_flex_selfcollide', 'has_ellipsoid_geom', 'has_plane_geom', 'has_1d_flex', 'has_2d_flex', 'has_3d_flex', 'max_flex_dim', 'block_dim', 'body_tree', 'body_branches', 'body_branch_start', 'mocap_bodyid', 'body_fluid_ellipsoid', 'body_is_free', 'body_fluid_ellipsoid_adr', 'body_fluid_box_adr', 'body_freeadr', 'jnt_limited_slide_hinge_adr', 'jnt_limited_ball_adr', 'body_isdofancestor', 'dof_tri_row', 'dof_tri_col', 'nxn_geom_pair', 'nxn_geom_pair_filtered', 'nxn_pairid', 'nxn_pairid_filtered', 'geom_pair_type_count', 'geom_plugin_index', 'eq_connect_adr', 'eq_wld_adr', 'eq_jnt_adr', 'eq_ten_adr', 'eq_flex_adr', 'eq_flexstrain_adr', 'tendon_jnt_adr', 'tendon_site_pair_adr', 'tendon_geom_adr', 'tendon_limited_adr', 'max_ten_J_rownnz', 'ten_wrapadr_site', 'ten_wrapnum_site', 'wrap_jnt_adr', 'wrap_site_adr', 'wrap_site_pair_adr', 'wrap_geom_adr', 'wrap_pulley_scale', 'actuator_trntype_body_adr', 'sensor_pos_adr', 'sensor_limitpos_adr', 'sensor_vel_adr', 'sensor_limitvel_adr', 'sensor_acc_adr', 'sensor_rangefinder_adr', 'rangefinder_sensor_adr', 'sensor_collision_start_adr', 'collision_sensor_adr', 'sensor_touch_adr', 'sensor_limitfrc_adr', 'sensor_e_potential', 'sensor_e_kinetic', 'sensor_tendonactfrc_adr', 'sensor_subtree_vel', 'sensor_contact_adr', 'sensor_adr_to_contact_adr', 'sensor_rne_postconstraint', 'sensor_rangefinder_bodyid', 'weld_tactile_id', 'taxel_vertadr', 'taxel_sensorid', 'M_tiles', 'qLD_updates', 'qLD_all_updates', 'qLD_level_offsets', 'M_fullm_i', 'M_fullm_j', 'M_elemid', 'M_hinit_i', 'M_fullm_upper_i', 'M_fullm_upper_j', 'M_fullm_upper_elemid', 'qD_fullm_i', 'qD_fullm_j', 'M_mulm_rowadr', 'M_mulm_col', 'M_mulm_madr', 'flex_elemflexid', 'flex_shellflexid', 'flex_vertflexid', 'flex_shelladr', 'flex_faceadr', 'flex_cell_map', 'flexstrain_J_rownnz', 'flexstrain_J_rowadr', 'flexstrain_J_colind', 'neq_flexstrain', 'nJfs', 'nflexbend_interp', 'flex_bend_interp_map', 'nflexface', 'flex_face_map', 'flex_face')
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, nq: int, nv: int, nu: int, nactuator: int, na: int, nbody: int, noct: int, njnt: int, ntree: int, nM: int, nC: int, nD: int, ngeom: int, nsite: int, ncam: int, nlight: int, nflex: int, nflexnode: int, nflexvert: int, nflexedge: int, nflexelem: int, nflexelemdata: int, nflexstiffness: int, nflexbending: int, nflexelemedge: int, nflexshelldata: int, nJfe: int, nmesh: int, nmeshvert: int, nmeshnormal: int, nmeshface: int, nmeshgraph: int, nmeshpoly: int, nmeshpolyvert: int, nmeshpolymap: int, nhfield: int, nhfielddata: int, nmat: int, npair: int, nexclude: int, neq: int, ntendon: int, nJten: int, nwrap: int, nsensor: int, nkey: int, nmocap: int, nplugin: int, nJmom: int, npolygonmax: int, nmeshdegmax: int, nuserdata: int, nsensordata: int, nhistory: int, opt: Option, stat: Statistic, qpos0: array(ndim=2, dtype=float32), qpos_spring: array(ndim=2, dtype=float32), body_parentid: array(ndim=1, dtype=int32), body_rootid: array(ndim=1, dtype=int32), body_weldid: array(ndim=1, dtype=int32), body_mocapid: array(ndim=1, dtype=int32), body_jntnum: array(ndim=1, dtype=int32), body_jntadr: array(ndim=1, dtype=int32), body_dofnum: array(ndim=1, dtype=int32), body_dofadr: array(ndim=1, dtype=int32), body_treeid: array(ndim=1, dtype=int32), body_geomnum: array(ndim=1, dtype=int32), body_geomadr: array(ndim=1, dtype=int32), body_simple: array(ndim=1, dtype=int32), body_pos: array(ndim=2, dtype=vec3f), body_quat: array(ndim=2, dtype=quatf), body_ipos: array(ndim=2, dtype=vec3f), body_iquat: array(ndim=2, dtype=quatf), body_mass: array(ndim=2, dtype=float32), body_subtreemass: array(ndim=2, dtype=float32), body_inertia: array(ndim=2, dtype=vec3f), body_invweight0: array(ndim=2, dtype=vec2f), body_gravcomp: array(ndim=2, dtype=float32), body_contype: array(ndim=1, dtype=int32), body_conaffinity: array(ndim=1, dtype=int32), oct_child: array(ndim=1, dtype=vector(length=8, dtype=int32)), oct_aabb: array(ndim=2, dtype=vec3f), oct_coeff: array(ndim=1, dtype=vector(length=8, dtype=float32)), jnt_type: array(ndim=1, dtype=int32), jnt_qposadr: array(ndim=1, dtype=int32), jnt_dofadr: array(ndim=1, dtype=int32), jnt_bodyid: array(ndim=1, dtype=int32), jnt_limited: array(ndim=1, dtype=int32), jnt_actfrclimited: array(ndim=1, dtype=warp._src.types.bool), jnt_actgravcomp: array(ndim=1, dtype=int32), jnt_solref: array(ndim=2, dtype=vec2f), jnt_solimp: array(ndim=2, dtype=vector(length=5, dtype=float32)), jnt_pos: array(ndim=2, dtype=vec3f), jnt_axis: array(ndim=2, dtype=vec3f), jnt_stiffness: array(ndim=2, dtype=float32), jnt_stiffnesspoly: array(ndim=2, dtype=vec2f), jnt_range: array(ndim=2, dtype=vec2f), jnt_actfrcrange: array(ndim=2, dtype=vec2f), jnt_margin: array(ndim=2, dtype=float32), dof_bodyid: array(ndim=1, dtype=int32), dof_jntid: array(ndim=1, dtype=int32), dof_parentid: array(ndim=1, dtype=int32), dof_treeid: array(ndim=1, dtype=int32), dof_Madr: array(ndim=1, dtype=int32), dof_solref: array(ndim=2, dtype=vec2f), dof_solimp: array(ndim=2, dtype=vector(length=5, dtype=float32)), dof_frictionloss: array(ndim=2, dtype=float32), dof_armature: array(ndim=2, dtype=float32), dof_damping: array(ndim=2, dtype=float32), dof_dampingpoly: array(ndim=2, dtype=vec2f), dof_invweight0: array(ndim=2, dtype=float32), dof_length: array(ndim=1, dtype=float32), tree_bodynum: array(ndim=1, dtype=int32), tree_dofadr: array(ndim=1, dtype=int32), tree_dofnum: array(ndim=1, dtype=int32), tree_sleep_policy: array(ndim=1, dtype=int32), geom_type: array(ndim=1, dtype=int32), geom_contype: array(ndim=1, dtype=int32), geom_conaffinity: array(ndim=1, dtype=int32), geom_condim: array(ndim=1, dtype=int32), geom_bodyid: array(ndim=1, dtype=int32), geom_dataid: array(ndim=2, dtype=int32), geom_matid: array(ndim=2, dtype=int32), geom_group: array(ndim=1, dtype=int32), geom_priority: array(ndim=1, dtype=int32), geom_solmix: array(ndim=2, dtype=float32), geom_solref: array(ndim=2, dtype=vec2f), geom_solimp: array(ndim=2, dtype=vector(length=5, dtype=float32)), geom_size: array(ndim=2, dtype=vec3f), geom_aabb: array(ndim=3, dtype=vec3f), geom_rbound: array(ndim=2, dtype=float32), geom_pos: array(ndim=2, dtype=vec3f), geom_quat: array(ndim=2, dtype=quatf), geom_friction: array(ndim=2, dtype=vec3f), geom_margin: array(ndim=2, dtype=float32), geom_gap: array(ndim=2, dtype=float32), geom_surfacevel: array(ndim=2, dtype=vector(length=6, dtype=float32)), geom_adhesion: array(ndim=2, dtype=float32), geom_fluid: array(ndim=2, dtype=float32), geom_rgba: array(ndim=2, dtype=vec4f), site_type: array(ndim=1, dtype=int32), site_bodyid: array(ndim=1, dtype=int32), site_size: array(ndim=1, dtype=vec3f), site_pos: array(ndim=2, dtype=vec3f), site_quat: array(ndim=2, dtype=quatf), cam_mode: array(ndim=1, dtype=int32), cam_bodyid: array(ndim=1, dtype=int32), cam_targetbodyid: array(ndim=1, dtype=int32), cam_pos: array(ndim=2, dtype=vec3f), cam_quat: array(ndim=2, dtype=quatf), cam_poscom0: array(ndim=2, dtype=vec3f), cam_pos0: array(ndim=2, dtype=vec3f), cam_mat0: array(ndim=2, dtype=mat33f), cam_projection: array(ndim=1, dtype=int32), cam_fovy: array(ndim=2, dtype=float32), cam_resolution: array(ndim=1, dtype=vec2i), cam_sensorsize: array(ndim=1, dtype=vec2f), cam_intrinsic: array(ndim=2, dtype=vec4f), light_mode: array(ndim=1, dtype=int32), light_bodyid: array(ndim=1, dtype=int32), light_targetbodyid: array(ndim=1, dtype=int32), light_type: array(ndim=2, dtype=int32), light_castshadow: array(ndim=2, dtype=warp._src.types.bool), light_active: array(ndim=2, dtype=warp._src.types.bool), light_pos: array(ndim=2, dtype=vec3f), light_dir: array(ndim=2, dtype=vec3f), light_poscom0: array(ndim=2, dtype=vec3f), light_pos0: array(ndim=2, dtype=vec3f), light_dir0: array(ndim=2, dtype=vec3f), light_attenuation: array(ndim=2, dtype=vec3f), light_cutoff: array(ndim=2, dtype=float32), light_exponent: array(ndim=2, dtype=float32), light_ambient: array(ndim=2, dtype=vec3f), light_diffuse: array(ndim=2, dtype=vec3f), light_specular: array(ndim=2, dtype=vec3f), flex_contype: array(ndim=1, dtype=int32), flex_conaffinity: array(ndim=1, dtype=int32), flex_condim: array(ndim=1, dtype=int32), flex_priority: array(ndim=1, dtype=int32), flex_solmix: array(ndim=1, dtype=float32), flex_solref: array(ndim=1, dtype=vec2f), flex_solimp: array(ndim=1, dtype=vector(length=5, dtype=float32)), flex_friction: array(ndim=1, dtype=vec3f), flex_margin: array(ndim=1, dtype=float32), flex_gap: array(ndim=1, dtype=float32), flex_selfcollide: array(ndim=1, dtype=int32), flex_activelayers: array(ndim=1, dtype=int32), flex_dim: array(ndim=1, dtype=int32), flex_interp: array(ndim=1, dtype=int32), flex_cellnum: array(ndim=1, dtype=vec3i), flex_nodeadr: array(ndim=1, dtype=int32), flex_nodenum: array(ndim=1, dtype=int32), flex_vertadr: array(ndim=1, dtype=int32), flex_vertnum: array(ndim=1, dtype=int32), flex_edgeadr: array(ndim=1, dtype=int32), flex_edgenum: array(ndim=1, dtype=int32), flex_elemadr: array(ndim=1, dtype=int32), flex_elemnum: array(ndim=1, dtype=int32), flex_elemdataadr: array(ndim=1, dtype=int32), flex_stiffnessadr: array(ndim=1, dtype=int32), flex_elemedgeadr: array(ndim=1, dtype=int32), flex_bendingadr: array(ndim=1, dtype=int32), flex_shellnum: array(ndim=1, dtype=int32), flex_shelldataadr: array(ndim=1, dtype=int32), flex_nodebodyid: array(ndim=1, dtype=int32), flex_vertbodyid: array(ndim=1, dtype=int32), flex_edge: array(ndim=1, dtype=vec2i), flex_edgeflap: array(ndim=1, dtype=vec2i), flex_elem: array(ndim=1, dtype=int32), flex_elemedge: array(ndim=1, dtype=int32), flex_elemlayer: array(ndim=1, dtype=int32), flex_shell: array(ndim=1, dtype=int32), flex_vert: array(ndim=1, dtype=vec3f), flex_vert0: array(ndim=1, dtype=vec3f), flex_node: array(ndim=1, dtype=vec3f), flex_node0: array(ndim=1, dtype=vec3f), flexedge_length0: array(ndim=1, dtype=float32), flexedge_invweight0: array(ndim=1, dtype=float32), flex_radius: array(ndim=1, dtype=float32), flex_stiffness: array(ndim=1, dtype=float32), flex_bending: array(ndim=1, dtype=float32), flex_damping: array(ndim=1, dtype=float32), flex_edgeequality: array(ndim=1, dtype=int32), flex_centered: array(ndim=1, dtype=warp._src.types.bool), flexedge_J_rownnz: array(ndim=1, dtype=int32), flexedge_J_rowadr: array(ndim=1, dtype=int32), flexedge_J_colind: array(ndim=1, dtype=int32), mesh_vertadr: array(ndim=1, dtype=int32), mesh_vertnum: array(ndim=1, dtype=int32), mesh_faceadr: array(ndim=1, dtype=int32), mesh_octadr: array(ndim=1, dtype=int32), mesh_normaladr: array(ndim=1, dtype=int32), mesh_normalnum: array(ndim=1, dtype=int32), mesh_graphadr: array(ndim=1, dtype=int32), mesh_vert: array(ndim=1, dtype=vec3f), mesh_normal: array(ndim=1, dtype=vec3f), mesh_face: array(ndim=1, dtype=vec3i), mesh_graph: array(ndim=1, dtype=int32), mesh_pos: array(ndim=1, dtype=vec3f), mesh_quat: array(ndim=1, dtype=quatf), mesh_polynum: array(ndim=1, dtype=int32), mesh_polyadr: array(ndim=1, dtype=int32), mesh_polynormal: array(ndim=1, dtype=vec3f), mesh_polyvertadr: array(ndim=1, dtype=int32), mesh_polyvertnum: array(ndim=1, dtype=int32), mesh_polyvert: array(ndim=1, dtype=int32), mesh_polymapadr: array(ndim=1, dtype=int32), mesh_polymapnum: array(ndim=1, dtype=int32), mesh_polymap: array(ndim=1, dtype=int32), hfield_size: array(ndim=1, dtype=vec4f), hfield_nrow: array(ndim=1, dtype=int32), hfield_ncol: array(ndim=1, dtype=int32), hfield_adr: array(ndim=1, dtype=int32), hfield_data: array(ndim=1, dtype=float32), mat_texid: array(ndim=3, dtype=int32), mat_texuniform: array(ndim=2, dtype=warp._src.types.bool), mat_texrepeat: array(ndim=2, dtype=vec2f), mat_emission: array(ndim=2, dtype=float32), mat_specular: array(ndim=2, dtype=float32), mat_shininess: array(ndim=2, dtype=float32), mat_rgba: array(ndim=2, dtype=vec4f), pair_dim: array(ndim=1, dtype=int32), pair_geom1: array(ndim=1, dtype=int32), pair_geom2: array(ndim=1, dtype=int32), pair_solref: array(ndim=2, dtype=vec2f), pair_solreffriction: array(ndim=2, dtype=vec2f), pair_solimp: array(ndim=2, dtype=vector(length=5, dtype=float32)), pair_margin: array(ndim=2, dtype=float32), pair_gap: array(ndim=2, dtype=float32), pair_adhesion: array(ndim=2, dtype=float32), pair_friction: array(ndim=2, dtype=vector(length=5, dtype=float32)), exclude_signature: array(ndim=1, dtype=int32), eq_type: array(ndim=1, dtype=int32), eq_obj1id: array(ndim=1, dtype=int32), eq_obj2id: array(ndim=1, dtype=int32), eq_objtype: array(ndim=1, dtype=int32), eq_active0: array(ndim=1, dtype=warp._src.types.bool), eq_solref: array(ndim=2, dtype=vec2f), eq_solimp: array(ndim=2, dtype=vector(length=5, dtype=float32)), eq_data: array(ndim=2, dtype=vector(length=11, dtype=float32)), tendon_adr: array(ndim=1, dtype=int32), tendon_num: array(ndim=1, dtype=int32), ten_J_rownnz: array(ndim=1, dtype=int32), ten_J_rowadr: array(ndim=1, dtype=int32), ten_J_colind: array(ndim=1, dtype=int32), tendon_limited: array(ndim=1, dtype=int32), tendon_actfrclimited: array(ndim=1, dtype=warp._src.types.bool), tendon_solref_lim: array(ndim=2, dtype=vec2f), tendon_solimp_lim: array(ndim=2, dtype=vector(length=5, dtype=float32)), tendon_solref_fri: array(ndim=2, dtype=vec2f), tendon_solimp_fri: array(ndim=2, dtype=vector(length=5, dtype=float32)), tendon_range: array(ndim=2, dtype=vec2f), tendon_actfrcrange: array(ndim=2, dtype=vec2f), tendon_margin: array(ndim=2, dtype=float32), tendon_stiffness: array(ndim=2, dtype=float32), tendon_stiffnesspoly: array(ndim=2, dtype=vec2f), tendon_damping: array(ndim=2, dtype=float32), tendon_dampingpoly: array(ndim=2, dtype=vec2f), tendon_armature: array(ndim=2, dtype=float32), tendon_frictionloss: array(ndim=2, dtype=float32), tendon_lengthspring: array(ndim=2, dtype=vec2f), tendon_length0: array(ndim=2, dtype=float32), tendon_invweight0: array(ndim=2, dtype=float32), wrap_type: array(ndim=1, dtype=int32), wrap_objid: array(ndim=1, dtype=int32), wrap_prm: array(ndim=1, dtype=float32), actuator_trntype: array(ndim=1, dtype=int32), actuator_dyntype: array(ndim=1, dtype=int32), actuator_gaintype: array(ndim=1, dtype=int32), actuator_biastype: array(ndim=1, dtype=int32), actuator_ctrladr: array(ndim=1, dtype=int32), actuator_ctrlnum: array(ndim=1, dtype=int32), actuator_ctrlspec: array(ndim=1, dtype=int32), actuator_actadr: array(ndim=1, dtype=int32), actuator_actnum: array(ndim=1, dtype=int32), actuator_trnid: array(ndim=1, dtype=vec2i), actuator_cranklength: array(ndim=2, dtype=float32), actuator_dynprm: array(ndim=2, dtype=vector(length=10, dtype=float32)), actuator_gainprm: array(ndim=2, dtype=vector(length=10, dtype=float32)), actuator_biasprm: array(ndim=2, dtype=vector(length=10, dtype=float32)), actuator_actlimited: array(ndim=1, dtype=warp._src.types.bool), actuator_actrange: array(ndim=2, dtype=vec2f), actuator_actearly: array(ndim=1, dtype=warp._src.types.bool), actuator_history: array(ndim=2, dtype=vec2i), actuator_historyadr: array(ndim=2, dtype=int32), actuator_delay: array(ndim=2, dtype=float32), actuator_forcelimited: array(ndim=1, dtype=warp._src.types.bool), actuator_forcerange: array(ndim=2, dtype=vec2f), actuator_ctrllimited: array(ndim=1, dtype=warp._src.types.bool), actuator_ctrlrange: array(ndim=2, dtype=vec2f), actuator_gear: array(ndim=2, dtype=vector(length=6, dtype=float32)), actuator_acc0: array(ndim=2, dtype=float32), actuator_lengthrange: array(ndim=2, dtype=vec2f), sensor_type: array(ndim=1, dtype=int32), sensor_datatype: array(ndim=1, dtype=int32), sensor_objtype: array(ndim=1, dtype=int32), sensor_objid: array(ndim=1, dtype=int32), sensor_reftype: array(ndim=1, dtype=int32), sensor_refid: array(ndim=1, dtype=int32), sensor_intprm: array(ndim=2, dtype=int32), sensor_dim: array(ndim=1, dtype=int32), sensor_adr: array(ndim=1, dtype=int32), sensor_cutoff: array(ndim=1, dtype=float32), sensor_history: array(ndim=2, dtype=vec2i), sensor_historyadr: array(ndim=2, dtype=int32), sensor_delay: array(ndim=2, dtype=float32), sensor_interval: array(ndim=2, dtype=vec2f), plugin: array(ndim=1, dtype=int32), plugin_attr: array(ndim=1, dtype=vector(length=128, dtype=float32)), key_time: array(ndim=1, dtype=float32), key_qpos: array(ndim=2, dtype=float32), key_qvel: array(ndim=2, dtype=float32), key_act: array(ndim=2, dtype=float32), key_mpos: array(ndim=2, dtype=vec3f), key_mquat: array(ndim=2, dtype=quatf), key_ctrl: array(ndim=2, dtype=float32), M_rownnz: array(ndim=1, dtype=int32), M_rowadr: array(ndim=1, dtype=int32), M_colind: array(ndim=1, dtype=int32), mapM2M: array(ndim=1, dtype=int32), D_rownnz: array(ndim=1, dtype=int32), D_rowadr: array(ndim=1, dtype=int32), D_diag: array(ndim=1, dtype=int32), D_colind: array(ndim=1, dtype=int32), mapM2D: array(ndim=1, dtype=int32), mapD2M: array(ndim=1, dtype=int32), callback: Callback, nbranch: int, nv_pad: int, nacttrnbody: int, nsensorcollision: int, nsensortaxel: int, ntactileweld: int, nsensorcontact: int, nrangefinder: int, nmaxcondim: int, nmaxpyramid: int, nflexintcell: int, is_sparse: bool, qLD_block_total: int, qLD_block_adr: array(ndim=1, dtype=int32), flg_adhesion: bool, has_fluid: bool, flg_surfacevel: bool, has_sdf_geom: bool, has_flex_selfcollide: bool, has_ellipsoid_geom: bool, has_plane_geom: bool, has_1d_flex: bool, has_2d_flex: bool, has_3d_flex: bool, max_flex_dim: int, block_dim: BlockDim, body_tree: tuple[array(ndim=1, dtype=int32), ...], body_branches: array(ndim=1, dtype=int32), body_branch_start: array(ndim=1, dtype=int32), mocap_bodyid: array(ndim=1, dtype=int32), body_fluid_ellipsoid: array(ndim=1, dtype=warp._src.types.bool), body_is_free: array(ndim=1, dtype=warp._src.types.bool), body_fluid_ellipsoid_adr: array(ndim=1, dtype=int32), body_fluid_box_adr: array(ndim=1, dtype=int32), body_freeadr: array(ndim=1, dtype=int32), jnt_limited_slide_hinge_adr: array(ndim=1, dtype=int32), jnt_limited_ball_adr: array(ndim=1, dtype=int32), body_isdofancestor: array(ndim=2, dtype=int32), dof_tri_row: array(ndim=1, dtype=int32), dof_tri_col: array(ndim=1, dtype=int32), nxn_geom_pair: array(ndim=1, dtype=vec2i), nxn_geom_pair_filtered: array(ndim=1, dtype=vec2i), nxn_pairid: array(ndim=1, dtype=vec2i), nxn_pairid_filtered: array(ndim=1, dtype=vec2i), geom_pair_type_count: tuple[int, ...], geom_plugin_index: array(ndim=1, dtype=int32), eq_connect_adr: array(ndim=1, dtype=int32), eq_wld_adr: array(ndim=1, dtype=int32), eq_jnt_adr: array(ndim=1, dtype=int32), eq_ten_adr: array(ndim=1, dtype=int32), eq_flex_adr: array(ndim=1, dtype=int32), eq_flexstrain_adr: array(ndim=1, dtype=int32), tendon_jnt_adr: array(ndim=1, dtype=int32), tendon_site_pair_adr: array(ndim=1, dtype=int32), tendon_geom_adr: array(ndim=1, dtype=int32), tendon_limited_adr: array(ndim=1, dtype=int32), max_ten_J_rownnz: int, ten_wrapadr_site: array(ndim=1, dtype=int32), ten_wrapnum_site: array(ndim=1, dtype=int32), wrap_jnt_adr: array(ndim=1, dtype=int32), wrap_site_adr: array(ndim=1, dtype=int32), wrap_site_pair_adr: array(ndim=1, dtype=int32), wrap_geom_adr: array(ndim=1, dtype=int32), wrap_pulley_scale: array(ndim=1, dtype=float32), actuator_trntype_body_adr: array(ndim=1, dtype=int32), sensor_pos_adr: array(ndim=1, dtype=int32), sensor_limitpos_adr: array(ndim=1, dtype=int32), sensor_vel_adr: array(ndim=1, dtype=int32), sensor_limitvel_adr: array(ndim=1, dtype=int32), sensor_acc_adr: array(ndim=1, dtype=int32), sensor_rangefinder_adr: array(ndim=1, dtype=int32), rangefinder_sensor_adr: array(ndim=1, dtype=int32), sensor_collision_start_adr: array(ndim=1, dtype=int32), collision_sensor_adr: array(ndim=1, dtype=int32), sensor_touch_adr: array(ndim=1, dtype=int32), sensor_limitfrc_adr: array(ndim=1, dtype=int32), sensor_e_potential: bool, sensor_e_kinetic: bool, sensor_tendonactfrc_adr: array(ndim=1, dtype=int32), sensor_subtree_vel: bool, sensor_contact_adr: array(ndim=1, dtype=int32), sensor_adr_to_contact_adr: array(ndim=1, dtype=int32), sensor_rne_postconstraint: bool, sensor_rangefinder_bodyid: array(ndim=1, dtype=int32), weld_tactile_id: array(ndim=1, dtype=int32), taxel_vertadr: array(ndim=1, dtype=int32), taxel_sensorid: array(ndim=1, dtype=int32), M_tiles: tuple[TileSet, ...], qLD_updates: tuple[array(ndim=1, dtype=vec3i), ...], qLD_all_updates: array(ndim=1, dtype=vec3i), qLD_level_offsets: array(ndim=1, dtype=int32), M_fullm_i: array(ndim=1, dtype=int32), M_fullm_j: array(ndim=1, dtype=int32), M_elemid: array(ndim=2, dtype=int32), M_hinit_i: array(ndim=1, dtype=int32), M_fullm_upper_i: array(ndim=1, dtype=int32), M_fullm_upper_j: array(ndim=1, dtype=int32), M_fullm_upper_elemid: array(ndim=1, dtype=int32), qD_fullm_i: array(ndim=1, dtype=int32), qD_fullm_j: array(ndim=1, dtype=int32), M_mulm_rowadr: array(ndim=1, dtype=int32), M_mulm_col: array(ndim=1, dtype=int32), M_mulm_madr: array(ndim=1, dtype=int32), flex_elemflexid: array(ndim=1, dtype=int32), flex_shellflexid: array(ndim=1, dtype=int32), flex_vertflexid: array(ndim=1, dtype=int32), flex_shelladr: array(ndim=1, dtype=int32), flex_faceadr: array(ndim=1, dtype=int32), flex_cell_map: array(ndim=1, dtype=vec4i), flexstrain_J_rownnz: array(ndim=1, dtype=int32), flexstrain_J_rowadr: array(ndim=1, dtype=int32), flexstrain_J_colind: array(ndim=1, dtype=int32), neq_flexstrain: int, nJfs: int, nflexbend_interp: int, flex_bend_interp_map: array(ndim=1, dtype=vec2i), nflexface: int, flex_face_map: array(ndim=1, dtype=vec2i), flex_face: array(ndim=2, dtype=int32)) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class ContactType(enum.IntFlag):
    """
    Type of contact.
    
    Attributes:
      CONSTRAINT: contact for constraint solver
      SENSOR: contact for collision sensor (GEOMDIST, GEOMNORMAL, GEOMFROMTO)
    """
    CONSTRAINT: typing.ClassVar[ContactType]
    SENSOR: typing.ClassVar[ContactType]
    _all_bits_: typing.ClassVar[int] = 3
    _boundary_: typing.ClassVar[enum.FlagBoundary]
    _flag_mask_: typing.ClassVar[int] = 3
    _hashable_values_: typing.ClassVar[list] = [1, 2]
    _inverted_ = None
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['CONSTRAINT', 'SENSOR']
    _singles_mask_: typing.ClassVar[int] = 3
    _unhashable_values_: typing.ClassVar[list] = list()
    _unhashable_values_map_: typing.ClassVar[dict] = {}
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.
        
        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __and__(self, other):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __invert__(self):
        ...
    def __or__(self, other):
        ...
    def __rand__(self, other):
        ...
    def __ror__(self, other):
        ...
    def __rxor__(self, other):
        ...
    def __xor__(self, other):
        ...
    def _value_repr_(self):
        """
        Return repr(self).
        """
class Contact:
    """
    Contact data.
    
    Attributes:
      dist: distance between nearest points; neg: penetration          (naconmax,)
      pos: position of contact point: midpoint between geoms           (naconmax, 3)
      frame: normal is in [0-2], points from geom[0] to geom[1]        (naconmax, 3, 3)
      includemargin: include if dist<includemargin=margin              (naconmax,)
      friction: tangent1, 2, spin, roll1, 2                            (naconmax, 5)
      solref: constraint solver reference, normal direction            (naconmax, 2)
      solreffriction: constraint solver reference, friction directions (naconmax, 2)
      solimp: constraint solver impedance                              (naconmax, 5)
      dim: contact space dimensionality: 1, 3, 4 or 6                  (naconmax,)
      geom: geom ids; -1 for flex                                      (naconmax, 2)
      flex: flex ids; -1 for geom                                      (naconmax, 2)
      elem: element ids; -1 for geom or flex vertex                    (naconmax, 2)
      vert: vertex ids for flex/mesh contact                           (naconmax, 2)
      efc_address: address in efc; -1: not included                    (naconmax, nmaxpyramid)
      worldid: world id                                                (naconmax,)
      type: ContactType                                                (naconmax,)
      geomcollisionid: i-th contact generated for geom                 (naconmax,)
                       helps uniquely identity contact when multiple
                       contacts are generated for geom pair
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('dist', 'pos', 'frame', 'includemargin', 'friction', 'solref', 'solreffriction', 'solimp', 'dim', 'geom', 'flex', 'elem', 'vert', 'efc_address', 'worldid', 'type', 'geomcollisionid', 'adhesion')
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, dist: array(ndim=1, dtype=float32), pos: array(ndim=1, dtype=vec3f), frame: array(ndim=1, dtype=mat33f), includemargin: array(ndim=1, dtype=float32), friction: array(ndim=1, dtype=vector(length=5, dtype=float32)), solref: array(ndim=1, dtype=vec2f), solreffriction: array(ndim=1, dtype=vec2f), solimp: array(ndim=1, dtype=vector(length=5, dtype=float32)), dim: array(ndim=1, dtype=int32), geom: array(ndim=1, dtype=vec2i), flex: array(ndim=1, dtype=vec2i), elem: array(ndim=1, dtype=vec2i), vert: array(ndim=1, dtype=vec2i), efc_address: array(ndim=2, dtype=int32), worldid: array(ndim=1, dtype=int32), type: array(ndim=1, dtype=int32), geomcollisionid: array(ndim=1, dtype=int32), adhesion: array(ndim=1, dtype=float32)) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class Constraint:
    """
    Constraint data.
    
    Attributes:
      type: constraint type (ConstraintType)            (nworld, njmax)
      id: id of object of specific type                 (nworld, njmax)
      jtdaj_adr: first efc row of each JTDAJ block      (nworld, njmax)
      jtdaj_nrow: efc rows per JTDAJ block              (nworld, njmax)
      jtdaj_nblock: number of JTDAJ blocks              (nworld,)
      J_rownnz: number of non-zeros in J row            (nworld, 0) dense
                                                        (nworld, njmax) sparse
      J_rowadr: row start address in colind array       (nworld, 0) dense
                                                        (nworld, njmax) sparse
      J_colind: column indices in J                     (nworld, 0, 0) dense
                                                        (nworld, 1, njmax_nnz) sparse
      J: constraint Jacobian                            (nworld, njmax_pad, nv_pad) dense
                                                        (nworld, 1, njmax_nnz) sparse
      pos: constraint position (equality, contact)      (nworld, njmax)
      margin: inclusion margin (contact)                (nworld, njmax)
      D: constraint mass                                (nworld, njmax_pad)
      vel: velocity in constraint space: J*qvel         (nworld, njmax)
      aref: reference pseudo-acceleration               (nworld, njmax)
      frictionloss: frictionloss (friction)             (nworld, njmax)
      force: constraint force in constraint space       (nworld, njmax)
      state: constraint state                           (nworld, njmax_pad)
      island: island ID per constraint                  (nworld, njmax)
    warp only fields:
      Ma: M*qacc                                        (nworld, nv)
      Jqvel: J*qvel                                     (nworld, njmax)
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('type', 'id', 'jtdaj_adr', 'jtdaj_nrow', 'jtdaj_nblock', 'J_rownnz', 'J_rowadr', 'J_colind', 'J', 'pos', 'margin', 'D', 'vel', 'aref', 'frictionloss', 'force', 'state', 'island', 'Ma', 'Jqvel')
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, type: array(ndim=2, dtype=int32), id: array(ndim=2, dtype=int32), jtdaj_adr: array(ndim=2, dtype=int32), jtdaj_nrow: array(ndim=2, dtype=int32), jtdaj_nblock: array(ndim=1, dtype=int32), J_rownnz: array(ndim=2, dtype=int32), J_rowadr: array(ndim=2, dtype=int32), J_colind: array(ndim=3, dtype=int32), J: array(ndim=3, dtype=float32), pos: array(ndim=2, dtype=float32), margin: array(ndim=2, dtype=float32), D: array(ndim=2, dtype=float32), vel: array(ndim=2, dtype=float32), aref: array(ndim=2, dtype=float32), frictionloss: array(ndim=2, dtype=float32), force: array(ndim=2, dtype=float32), state: array(ndim=2, dtype=int32), island: array(ndim=2, dtype=int32), Ma: array(ndim=2, dtype=float32), Jqvel: array(ndim=2, dtype=float32)) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class Data:
    """
    Dynamic state that updates each step.
    
    Attributes:
      solver_niter: number of solver iterations                   (nworld,)
      ne: number of equality constraints                          (nworld,)
      nf: number of friction constraints                          (nworld,)
      nl: number of limit constraints                             (nworld,)
      nefc: number of constraints                                 (nworld,)
      nisland: number of constraint islands                       (nworld,)
      nidof: total DOFs in islands                                (nworld,)
      ntree_awake: number of awake trees                          (nworld,)
      nbody_awake: number of awake bodies                         (nworld,)
      nv_awake: number of awake dofs                              (nworld,)
      time: simulation time                                       (nworld,)
      energy: potential, kinetic energy                           (nworld, 2)
      qpos: position                                              (nworld, nq)
      qvel: velocity                                              (nworld, nv)
      act: actuator activation                                    (nworld, na)
      history: history buffer for delays                          (nworld, nhistory)
      qacc_warmstart: acceleration used for warmstart             (nworld, nv)
      ctrl: control                                               (nworld, nu)
      qfrc_applied: applied generalized force                     (nworld, nv)
      xfrc_applied: applied Cartesian force/torque                (nworld, nbody, 6)
      eq_active: enable/disable constraints                       (nworld, neq)
      mocap_pos: position of mocap bodies                         (nworld, nmocap, 3)
      mocap_quat: orientation of mocap bodies                     (nworld, nmocap, 4)
      qacc: acceleration                                          (nworld, nv)
      act_dot: time-derivative of actuator activation             (nworld, na)
      userdata: custom user data                                  (nworld, nuserdata)
      sensordata: sensor data array                               (nworld, nsensordata,)
      tree_asleep: tree asleep counter; >=0: asleep cycle         (nworld, ntree)
      xpos: Cartesian position of body frame                      (nworld, nbody, 3)
      xquat: Cartesian orientation of body frame                  (nworld, nbody, 4)
      xmat: Cartesian orientation of body frame                   (nworld, nbody, 3, 3)
      xipos: Cartesian position of body com                       (nworld, nbody, 3)
      ximat: Cartesian orientation of body inertia                (nworld, nbody, 3, 3)
      xanchor: Cartesian position of joint anchor                 (nworld, njnt, 3)
      xaxis: Cartesian joint axis                                 (nworld, njnt, 3)
      geom_xpos: Cartesian geom position                          (nworld, ngeom, 3)
      geom_xmat: Cartesian geom orientation                       (nworld, ngeom, 3, 3)
      site_xpos: Cartesian site position                          (nworld, nsite, 3)
      site_xmat: Cartesian site orientation                       (nworld, nsite, 3, 3)
      cam_xpos: Cartesian camera position                         (nworld, ncam, 3)
      cam_xmat: Cartesian camera orientation                      (nworld, ncam, 3, 3)
      light_xpos: Cartesian light position                        (nworld, nlight, 3)
      light_xdir: Cartesian light direction                       (nworld, nlight, 3)
      subtree_com: center of mass of each subtree                 (nworld, nbody, 3)
      cdof: com-based motion axis of each dof (rot:lin)           (nworld, nv, 6)
      cinert: com-based body inertia and mass                     (nworld, nbody, 10)
      flexvert_xpos: cartesian flex vertex positions              (nworld, nflexvert, 3)
      flexedge_J: edge length Jacobian                            (nworld, nJfe)
      flexedge_length: flex edge lengths                          (nworld, nflexedge)
      ten_wrapadr: start address of tendon's path                 (nworld, ntendon)
      ten_wrapnum: number of wrap points in path                  (nworld, ntendon)
      ten_J: tendon Jacobian                                      (nworld, nJten)
      ten_length: tendon lengths                                  (nworld, ntendon)
      wrap_obj: geomid; -1: site; -2: pulley                      (nworld, nwrap, 2)
      wrap_xpos: Cartesian 3D points in all paths                 (nworld, nwrap, 6)
      actuator_length: actuator lengths                           (nworld, nu)
      moment_rownnz: number of non-zeros in actuator_moment row   (nworld, nu)
      moment_rowadr: row start address in actuator_moment         (nworld, nu)
      moment_colind: column indices in sparse actuator_moment     (nworld, nJmom)
      actuator_moment: actuator moments                           (nworld, nJmom)
      crb: com-based composite inertia and mass                   (nworld, nbody, 10)
      M: total inertia, CSR                                       (nworld, nC)
      qLD: per-block factor: packed dense region, then the nC     (nworld, qLD_block_total + nC)
           L'*D*L region at offset qLD_block_total (nC=0 if no sparse block)
      qLDiagInv: reciprocal diagonal for compact and sparse blocks (nworld, nv)
      tree_awake: is tree awake; 0: asleep; 1: awake              (nworld, ntree)
      body_awake: body sleep state (SleepState)                   (nworld, nbody)
      body_awake_ind: indices of awake/static bodies              (nworld, nbody)
      dof_awake_ind: indices of awake dofs                        (nworld, nv)
      flexedge_velocity: flex edge velocities                     (nworld, nflexedge)
      ten_velocity: tendon velocities                             (nworld, ntendon)
      actuator_velocity: actuator velocities                      (nworld, nu)
      cvel: com-based velocity (rot:lin)                          (nworld, nbody, 6)
      cdof_dot: time-derivative of cdof (rot:lin)                 (nworld, nv, 6)
      qfrc_bias: C(qpos,qvel)                                     (nworld, nv)
      qfrc_spring: passive spring force                           (nworld, nv)
      qfrc_damper: passive damper force                           (nworld, nv)
      qfrc_gravcomp: passive gravity compensation force           (nworld, nv)
      qfrc_fluid: passive fluid force                             (nworld, nv)
      qfrc_adhesion: passive adhesion force                       (nworld, nv)
      qfrc_passive: total passive force                           (nworld, nv)
      subtree_linvel: linear velocity of subtree com              (nworld, nbody, 3)
      subtree_angmom: angular momentum about subtree com          (nworld, nbody, 3)
      qLU: sparse LU factorization of (M - dt*qDeriv)             (nworld, nD)
      actuator_force: actuator force in actuation space           (nworld, nu)
      qfrc_actuator: actuator force                               (nworld, nv)
      qfrc_smooth: net unconstrained force                        (nworld, nv)
      qacc_smooth: unconstrained acceleration                     (nworld, nv)
      qfrc_constraint: constraint force                           (nworld, nv)
      qfrc_inverse: net external force; should equal:             (nworld, nv)
                    qfrc_applied + J.T @ xfrc_applied
                    + qfrc_actuator
      cacc: com-based acceleration                                (nworld, nbody, 6)
      cfrc_int: com-based interaction force with parent           (nworld, nbody, 6)
      cfrc_ext: com-based external force on body                  (nworld, nbody, 6)
      contact: contact data
      efc: constraint data
      tree_island: island ID per tree (-1 if unconstrained)       (nworld, ntree)
      dof_island: island ID per DOF (-1 if unconstrained)         (nworld, nv)
      island_dofadr: island start address in dof vector           (nworld, ntree)
      island_idofadr: island start address in idof vector         (nworld, ntree)
      island_nv: DOFs per island                                  (nworld, ntree)
      island_nefc: constraints per island                         (nworld, ntree)
      island_ne: equality constraints per island                  (nworld, ntree)
      island_nf: friction constraints per island                  (nworld, ntree)
      island_iefcadr: island start address in efc vector          (nworld, ntree)
      map_dof2idof: global DOF -> island-local DOF                (nworld, nv)
      map_idof2dof: island-local DOF -> global DOF                (nworld, nv)
      map_efc2iefc: global EFC -> island-local EFC                (nworld, njmax)
      map_iefc2efc: island-local EFC -> global EFC                (nworld, njmax)
      dof_islandid: island ID per island-DOF                      (nworld, nv)
      efc_islandid: island ID per island-EFC                      (nworld, njmax)
      ncdof: number of active (compacted) DOFs per world          (nworld,)
      dof_cdof: global DOF -> compacted DOF; -1 if inactive       (nworld, nv)
      cdof_dof: compacted DOF -> global DOF; -1 if unused         (nworld, nvmax_pad)
      ctol: compacted-solve main tolerance (nv/nvmax_pad scaled)  (1,)
      cls_tol: compacted-solve linesearch tolerance               (1,)
      cdof_tri_row: row index of compacted Hessian dof-pairs      (nvmax_pad_sq,)
      cdof_tri_col: col index of compacted Hessian dof-pairs      (nvmax_pad_sq,)
      cM: compacted dense inertia                                 (nworld, nvmax_pad, nvmax_pad)
      cqLD: compacted upper Cholesky factor                       (nworld, nvmax_pad, nvmax_pad)
      crhs: compacted smooth-solve right-hand side                (nworld, nvmax_pad, 1)
      cx: compacted smooth-solve solution                         (nworld, nvmax_pad, 1)
      cJ: compacted dense constraint Jacobian                     (nworld, njmax_pad, nvmax_pad)
      cMa: compacted M @ qacc workspace                           (nworld, nvmax_pad)
      cqfrc_smooth: compacted net unconstrained force             (nworld, nvmax_pad)
      cqacc_smooth: compacted unconstrained acceleration          (nworld, nvmax_pad)
      cqacc_warmstart: compacted warmstart acceleration           (nworld, nvmax_pad)
      cqacc: compacted acceleration (solve output)                (nworld, nvmax_pad)
      cqfrc_constraint: compacted constraint force                (nworld, nvmax_pad)
    
    warp only fields:
      nworld: number of worlds
      naconmax: maximum number of contacts (shared across all worlds)
      naccdmax: maximum number of contacts for CCD (all worlds)
      njmax: maximum number of constraints per world
      nvmax: capacity for compacted active DOFs per world
      nvmax_pad: nvmax rounded up to the nearest multiple of TILE_SIZE_JTDAJ_DENSE
      njmax_pad: njmax rounded up to the nearest multiple of TILE_SIZE_JTDAJ
      njmax_nnz: number of non-zeros in constraint Jacobian
      nacon: number of detected contacts (across all worlds)      (1,)
      ncollision: collision count from broadphase                 (1,)
      flex_aabb_min: dynamic flex object bounding box min         (nworld, nflex, 3)
      flex_aabb_max: dynamic flex object bounding box max         (nworld, nflex, 3)
      flexnode_xpos: cartesian flex node positions                (nworld, nflexnode, 3)
      overflow: overflow bitmask (OverflowType)                   (nworld,)
      face_xpos: cartesian flex face positions                    (nworld, nflexface, 9, 3)
      face_quat: cartesian flex face orientations                 (nworld, nflexface, 4)
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('solver_niter', 'ne', 'nf', 'nl', 'nefc', 'nisland', 'nidof', 'ntree_awake', 'nbody_awake', 'nv_awake', 'time', 'energy', 'qpos', 'qvel', 'act', 'history', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat', 'qacc', 'act_dot', 'userdata', 'sensordata', 'tree_asleep', 'xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'xanchor', 'xaxis', 'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir', 'subtree_com', 'cdof', 'cinert', 'flexvert_xpos', 'flexedge_J', 'flexedge_length', 'ten_wrapadr', 'ten_wrapnum', 'ten_J', 'ten_length', 'wrap_obj', 'wrap_xpos', 'actuator_length', 'moment_rownnz', 'moment_rowadr', 'moment_colind', 'actuator_moment', 'crb', 'M', 'qLD', 'qLDiagInv', 'tree_awake', 'body_awake', 'body_awake_ind', 'dof_awake_ind', 'flexedge_velocity', 'ten_velocity', 'actuator_velocity', 'cvel', 'cdof_dot', 'qfrc_bias', 'qfrc_spring', 'qfrc_damper', 'qfrc_gravcomp', 'qfrc_fluid', 'qfrc_adhesion', 'qfrc_passive', 'subtree_linvel', 'subtree_angmom', 'qLU', 'actuator_force', 'qfrc_actuator', 'qfrc_smooth', 'qacc_smooth', 'qfrc_constraint', 'qfrc_inverse', 'cacc', 'cfrc_int', 'cfrc_ext', 'contact', 'efc', 'tree_island', 'dof_island', 'island_dofadr', 'island_idofadr', 'island_nv', 'island_nefc', 'island_ne', 'island_nf', 'island_iefcadr', 'map_dof2idof', 'map_idof2dof', 'map_efc2iefc', 'map_iefc2efc', 'dof_islandid', 'efc_islandid', 'ncdof', 'dof_cdof', 'cdof_dof', 'ctol', 'cls_tol', 'cdof_tri_row', 'cdof_tri_col', 'cM', 'cqLD', 'crhs', 'cx', 'cJ', 'cMa', 'cqfrc_smooth', 'cqacc_smooth', 'cqacc_warmstart', 'cqacc', 'cqfrc_constraint', 'nworld', 'naconmax', 'naccdmax', 'njmax', 'nvmax', 'nvmax_pad', 'njmax_pad', 'njmax_nnz', 'nacon', 'ncollision', 'flex_aabb_min', 'flex_aabb_max', 'flexnode_xpos', 'overflow', 'face_xpos', 'face_quat')
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, solver_niter: array(ndim=1, dtype=int32), ne: array(ndim=1, dtype=int32), nf: array(ndim=1, dtype=int32), nl: array(ndim=1, dtype=int32), nefc: array(ndim=1, dtype=int32), nisland: array(ndim=1, dtype=int32), nidof: array(ndim=1, dtype=int32), ntree_awake: array(ndim=1, dtype=int32), nbody_awake: array(ndim=1, dtype=int32), nv_awake: array(ndim=1, dtype=int32), time: array(ndim=1, dtype=float32), energy: array(ndim=1, dtype=vec2f), qpos: array(ndim=2, dtype=float32), qvel: array(ndim=2, dtype=float32), act: array(ndim=2, dtype=float32), history: array(ndim=2, dtype=float32), qacc_warmstart: array(ndim=2, dtype=float32), ctrl: array(ndim=2, dtype=float32), qfrc_applied: array(ndim=2, dtype=float32), xfrc_applied: array(ndim=2, dtype=vector(length=6, dtype=float32)), eq_active: array(ndim=2, dtype=warp._src.types.bool), mocap_pos: array(ndim=2, dtype=vec3f), mocap_quat: array(ndim=2, dtype=quatf), qacc: array(ndim=2, dtype=float32), act_dot: array(ndim=2, dtype=float32), userdata: array(ndim=2, dtype=float32), sensordata: array(ndim=2, dtype=float32), tree_asleep: array(ndim=2, dtype=int32), xpos: array(ndim=2, dtype=vec3f), xquat: array(ndim=2, dtype=quatf), xmat: array(ndim=2, dtype=mat33f), xipos: array(ndim=2, dtype=vec3f), ximat: array(ndim=2, dtype=mat33f), xanchor: array(ndim=2, dtype=vec3f), xaxis: array(ndim=2, dtype=vec3f), geom_xpos: array(ndim=2, dtype=vec3f), geom_xmat: array(ndim=2, dtype=mat33f), site_xpos: array(ndim=2, dtype=vec3f), site_xmat: array(ndim=2, dtype=mat33f), cam_xpos: array(ndim=2, dtype=vec3f), cam_xmat: array(ndim=2, dtype=mat33f), light_xpos: array(ndim=2, dtype=vec3f), light_xdir: array(ndim=2, dtype=vec3f), subtree_com: array(ndim=2, dtype=vec3f), cdof: array(ndim=2, dtype=vector(length=6, dtype=float32)), cinert: array(ndim=2, dtype=vector(length=10, dtype=float32)), flexvert_xpos: array(ndim=2, dtype=vec3f), flexedge_J: array(ndim=2, dtype=float32), flexedge_length: array(ndim=2, dtype=float32), ten_wrapadr: array(ndim=2, dtype=int32), ten_wrapnum: array(ndim=2, dtype=int32), ten_J: array(ndim=2, dtype=float32), ten_length: array(ndim=2, dtype=float32), wrap_obj: array(ndim=2, dtype=vec2i), wrap_xpos: array(ndim=2, dtype=vector(length=6, dtype=float32)), actuator_length: array(ndim=2, dtype=float32), moment_rownnz: array(ndim=2, dtype=int32), moment_rowadr: array(ndim=2, dtype=int32), moment_colind: array(ndim=2, dtype=int32), actuator_moment: array(ndim=2, dtype=float32), crb: array(ndim=2, dtype=vector(length=10, dtype=float32)), M: array(ndim=2, dtype=float32), qLD: array(ndim=2, dtype=float32), qLDiagInv: array(ndim=2, dtype=float32), tree_awake: array(ndim=2, dtype=int32), body_awake: array(ndim=2, dtype=int32), body_awake_ind: array(ndim=2, dtype=int32), dof_awake_ind: array(ndim=2, dtype=int32), flexedge_velocity: array(ndim=2, dtype=float32), ten_velocity: array(ndim=2, dtype=float32), actuator_velocity: array(ndim=2, dtype=float32), cvel: array(ndim=2, dtype=vector(length=6, dtype=float32)), cdof_dot: array(ndim=2, dtype=vector(length=6, dtype=float32)), qfrc_bias: array(ndim=2, dtype=float32), qfrc_spring: array(ndim=2, dtype=float32), qfrc_damper: array(ndim=2, dtype=float32), qfrc_gravcomp: array(ndim=2, dtype=float32), qfrc_fluid: array(ndim=2, dtype=float32), qfrc_adhesion: array(ndim=2, dtype=float32), qfrc_passive: array(ndim=2, dtype=float32), subtree_linvel: array(ndim=2, dtype=vec3f), subtree_angmom: array(ndim=2, dtype=vec3f), qLU: array(ndim=2, dtype=float32), actuator_force: array(ndim=2, dtype=float32), qfrc_actuator: array(ndim=2, dtype=float32), qfrc_smooth: array(ndim=2, dtype=float32), qacc_smooth: array(ndim=2, dtype=float32), qfrc_constraint: array(ndim=2, dtype=float32), qfrc_inverse: array(ndim=2, dtype=float32), cacc: array(ndim=2, dtype=vector(length=6, dtype=float32)), cfrc_int: array(ndim=2, dtype=vector(length=6, dtype=float32)), cfrc_ext: array(ndim=2, dtype=vector(length=6, dtype=float32)), contact: Contact, efc: Constraint, tree_island: array(ndim=2, dtype=int32), dof_island: array(ndim=2, dtype=int32), island_dofadr: array(ndim=2, dtype=int32), island_idofadr: array(ndim=2, dtype=int32), island_nv: array(ndim=2, dtype=int32), island_nefc: array(ndim=2, dtype=int32), island_ne: array(ndim=2, dtype=int32), island_nf: array(ndim=2, dtype=int32), island_iefcadr: array(ndim=2, dtype=int32), map_dof2idof: array(ndim=2, dtype=int32), map_idof2dof: array(ndim=2, dtype=int32), map_efc2iefc: array(ndim=2, dtype=int32), map_iefc2efc: array(ndim=2, dtype=int32), dof_islandid: array(ndim=2, dtype=int32), efc_islandid: array(ndim=2, dtype=int32), ncdof: array(ndim=1, dtype=int32), dof_cdof: array(ndim=2, dtype=int32), cdof_dof: array(ndim=2, dtype=int32), ctol: array(ndim=1, dtype=float32), cls_tol: array(ndim=1, dtype=float32), cdof_tri_row: array(ndim=1, dtype=int32), cdof_tri_col: array(ndim=1, dtype=int32), cM: array(ndim=3, dtype=float32), cqLD: array(ndim=3, dtype=float32), crhs: array(ndim=3, dtype=float32), cx: array(ndim=3, dtype=float32), cJ: array(ndim=3, dtype=float32), cMa: array(ndim=2, dtype=float32), cqfrc_smooth: array(ndim=2, dtype=float32), cqacc_smooth: array(ndim=2, dtype=float32), cqacc_warmstart: array(ndim=2, dtype=float32), cqacc: array(ndim=2, dtype=float32), cqfrc_constraint: array(ndim=2, dtype=float32), nworld: int, naconmax: int, naccdmax: int, njmax: int, nvmax: int, nvmax_pad: int, njmax_pad: int, njmax_nnz: int, nacon: array(ndim=1, dtype=int32), ncollision: array(ndim=1, dtype=int32), flex_aabb_min: array(ndim=2, dtype=vec3f), flex_aabb_max: array(ndim=2, dtype=vec3f), flexnode_xpos: array(ndim=2, dtype=vec3f), overflow: array(ndim=1, dtype=int32), face_xpos: array(ndim=3, dtype=vec3f), face_quat: array(ndim=2, dtype=quatf)) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class InverseContext:
    """
    Workspace arrays for inverse dynamics.
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('Jaref', 'search_dot', 'done', 'quad_changed_ids', 'quad_changed_count', 'state_changed_count', 'ls_exhausted', 'compact_m_full', 'compact_d_full')
    compact_d_full = None
    compact_m_full = None
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, Jaref: wp.array2d[wp.float32], search_dot: wp.array[wp.float32], done: wp.array[wp.bool], quad_changed_ids: wp.array2d[wp.int32], quad_changed_count: wp.array[wp.int32], state_changed_count: wp.array[wp.int32], ls_exhausted: wp.array[wp.bool], compact_m_full: ForwardRef('Model') | None = None, compact_d_full: ForwardRef('Data') | None = None) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class SolverContext:
    """
    Workspace arrays for constraint solver.
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('Jaref', 'search_dot', 'done', 'grad', 'grad_dot', 'newton_decrement', 'Mgrad', 'search', 'mv', 'jv', 'quad', 'alpha', 'grad_scale', 'state_changed_count', 'improvement', 'ls_exhausted', 'search_unchanged', 'prev_grad', 'prev_Mgrad', 'beta', 'h', 'hfactor', 'quad_changed_ids', 'quad_changed_count', 'compact_m_full', 'compact_d_full')
    compact_d_full = None
    compact_m_full = None
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, Jaref: wp.array2d[wp.float32], search_dot: wp.array[wp.float32], done: wp.array[wp.bool], grad: wp.array2d[wp.float32], grad_dot: wp.array[wp.float32], newton_decrement: wp.array[wp.float32], Mgrad: wp.array2d[wp.float32], search: wp.array2d[wp.float32], mv: wp.array2d[wp.float32], jv: wp.array2d[wp.float32], quad: wp.array2d[wp.vec3f], alpha: wp.array[wp.float32], grad_scale: wp.array[wp.float32], state_changed_count: wp.array[wp.int32], improvement: wp.array[wp.float32], ls_exhausted: wp.array[wp.bool], search_unchanged: wp.array[wp.bool], prev_grad: wp.array2d[wp.float32], prev_Mgrad: wp.array2d[wp.float32], beta: wp.array[wp.float32], h: wp.array3d[wp.float32], hfactor: wp.array3d[wp.float32], quad_changed_ids: wp.array2d[wp.int32], quad_changed_count: wp.array[wp.int32], compact_m_full: ForwardRef('Model') | None = None, compact_d_full: ForwardRef('Data') | None = None) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
class RenderContext:
    """
    Context for rendering.
    
    Attributes:
      nrender: number of actively rendering cameras
      cam_res: camera resolution for actively rendering cameras
      cam_id_map: camera id map
      use_textures: whether to use textures
      use_fast_math: whether to enable fast math for the render kernel
      use_shadows: whether to use shadows
      use_ambient_lighting: top-level switch for ambient contributions
      background_color: color used for missed rays when no skybox is rendered
      use_precomputed_rays: whether to use precomputed rays
      bvh_ngeom: number of geometries in the BVH
      enabled_geom_ids: enabled geometry ids
      mesh_registry: mesh BVH id to warp mesh mapping
      mesh_bvh_id: mesh BVH ids
      mesh_bounds_size: mesh bounds size
      mesh_texcoord: mesh texture coordinates
      mesh_texcoord_offsets: mesh texture coordinate offsets
      mesh_facetexcoord: mesh face texture coordinates
      mesh_facenormal: per-face indices into Model.mesh_normal
      samples_per_pixel: sub-pixel samples per axis; 1 disables supersampling
      aa_accum: colour accumulator, unused when samples_per_pixel is 1
      textures: textures
      textures_registry: texture registry
      hfield_registry: hfield BVH id to warp mesh mapping
      hfield_bvh_id: hfield BVH ids
      hfield_bounds_size: hfield bounds half-extents
      flex_mesh_registry: per-flex mesh BVH registry (prevents garbage collection)
      flex_rgba: flex rgba
      flex_bvh_id: per-flex BVH ids
      flex_group_root: per-flex group roots (nworld x n_flex_bvh)
      flex_render_smooth: whether to render flex meshes smoothly
      bvh_nflexgeom: number of flex geometries in the BVH
      flex_dim_np: flex dimension per flex (1D/2D/3D)
      flex_geom_flexid: map from flex geom ID to flex ID
      flex_geom_edgeid: map from flex geom ID to flex edge ID
      bvh: scene BVH
      bvh_id: scene BVH id
      lower: lower bounds
      upper: upper bounds
      group: groups
      group_root: group roots
      ray: per-pixel rays direction
      ray_offset: per-pixel ray offset from camera center (for orthographic
        camera projections only)
      rgb_data: RGB data
      rgb_adr: RGB addresses
      depth_data: depth data
      depth_adr: depth addresses
      render_rgb: per-camera RGB render flags
      render_depth: per-camera depth render flags
      seg_data: segmentation data (per-pixel object ID/type pairs)
      seg_adr: segmentation addresses
      render_seg: per-camera segmentation render flags
      znear: near plane distance
      zfar: far plane distance
      total_rays: total number of rays
      render_skybox: whether to shade missed rays with a MuJoCo skybox texture
      skybox_tex_id: per-world indices into textures of the skybox
      skybox_face_width: per-world pixel widths of the skybox cube face
      headlight_active: whether to inject MuJoCo's vis.headlight as a synthetic
        directional light at the active camera. Read from `mjm.vis.headlight.active`
        at context creation; users disable the headlight by configuring it on the
        MuJoCo model (e.g. `<visual><headlight active="0"/></visual>` in XML).
      headlight_ambient: RGB ambient color of the headlight (from vis.headlight).
      headlight_diffuse: RGB diffuse color of the headlight.
      headlight_specular: RGB specular color of the headlight.
      shadow_light_fraction: fraction of a light's direct contribution reaching an
        occluded point; 0 is a true shadow
      enable_backface_culling: drop primitive ray hits whose normal faces away
        from the ray (i.e. the ray origin is inside the geom). Matches MuJoCo's
        mesh-ray rule. When False, the renderer reports inner-surface hits, which
        is faster but causes a camera placed inside a geom to render that geom's
        back wall.
      light_attenuation_is_default: True iff every light in the model has the
        MuJoCo default `attenuation = (1, 0, 0)`. Computed once at context
        creation; when True the kernel skips the per-light polynomial
        attenuation evaluation (a divide + 3 multiplies + an add per
        non-directional light per pixel) via `wp.static`.
      has_spot_lights: True iff any light in the model has `type == SPOT`.
        When False, the kernel skips the spot-cone branch (cos cutoff +
        pow exponent) per non-directional light per pixel via `wp.static`.
      has_orthographic_camera: True iff any actively rendering camera uses
        orthographic projection. When False, the kernel skips the ray origin
        offset since it is always zero for perspective-only scenes.
      enable_vertex_normals: when True, shade meshes from their authored vertex
        normals, matching mjr_uploadMesh; when False, use the face normal.
      enable_specular: when True, evaluate the Phong specular highlight per
        light per pixel (uses `mat_specular` / `mat_shininess`). When False,
        the entire specular branch is removed at compile time. Useful for
        depth/segmentation-only workflows or when materials are matte.
      enable_emission: when True, add `mat_emission * base_color` to each
        shaded pixel. When False the term is dropped at compile time.
      enable_per_light_ambient: when True and `use_ambient_lighting` is also
        True, sum the per-light `light_ambient` colors into each shaded pixel
        even when the surface normal is perpendicular to the light direction
        or the pixel is shadowed. When False the second per-light loop for
        ambient is removed at compile time. Headlight ambient and the no-light
        fallback are controlled by `use_ambient_lighting`.
      geom_ray_types: tuple of GeomType int values present in the scene, used to
        statically eliminate unused intersection branches in the ray-cast kernels.
      splat_position: Splat centers in world coordinates (nsplat, 3)
      splat_rotation: Splat rotation as (w, x, y, z) (nsplat, 4)
      splat_scale: Splat scale as standard deviation in each dimension (nsplat, 3)
      splat_rgba: Splat color and opacity (nsplat, 4)
      splat_bvh: Splat BVH
      splat_lower: Splat lower bounds
      splat_upper: Splat upper bounds
      splat_bvh_id: Splat BVH id
      splat_group_root: Per-world selected splat BVH root
      splat_count: Number of splats
    """
    __annotations_cache__: typing.ClassVar[dict]
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('nrender', 'cam_res', 'cam_id_map', 'use_textures', 'use_fast_math', 'use_shadows', 'use_ambient_lighting', 'background_color', 'use_precomputed_rays', 'render_skybox', 'skybox_tex_id', 'skybox_face_width', 'headlight_active', 'headlight_ambient', 'headlight_diffuse', 'headlight_specular', 'bvh_ngeom', 'enabled_geom_ids', 'mesh_registry', 'mesh_bvh_id', 'mesh_bounds_size', 'mesh_texcoord', 'mesh_texcoord_offsets', 'mesh_facetexcoord', 'mesh_facenormal', 'samples_per_pixel', 'aa_accum', 'textures', 'textures_registry', 'hfield_registry', 'hfield_bvh_id', 'hfield_bounds_size', 'flex_mesh_registry', 'flex_rgba', 'flex_bvh_id', 'flex_group_root', 'flex_render_smooth', 'bvh_nflexgeom', 'flex_dim_np', 'flex_geom_flexid', 'flex_geom_edgeid', 'bvh', 'bvh_id', 'lower', 'upper', 'group', 'group_root', 'ray', 'ray_offset', 'rgb_data', 'rgb_adr', 'depth_data', 'depth_adr', 'render_rgb', 'render_depth', 'seg_data', 'seg_adr', 'render_seg', 'znear', 'zfar', 'total_rays', 'enable_backface_culling', 'shadow_light_fraction', 'enable_vertex_normals', 'enable_specular', 'enable_emission', 'enable_per_light_ambient', 'light_attenuation_is_default', 'has_spot_lights', 'has_orthographic_camera', 'splat_position', 'splat_rotation', 'splat_scale', 'splat_rgba', 'splat_bvh', 'splat_lower', 'splat_upper', 'splat_bvh_id', 'splat_group_root', 'splat_count', 'geom_ray_types', '_megakernel')
    _megakernel = None
    geom_ray_types: typing.ClassVar[tuple] = tuple()
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, nrender: int, cam_res: array(ndim=1, dtype=vec2i), cam_id_map: array(ndim=1, dtype=int32), use_textures: bool, use_fast_math: bool, use_shadows: bool, use_ambient_lighting: bool, background_color: warp._src.types.uint32, use_precomputed_rays: bool, render_skybox: bool, skybox_tex_id: array(ndim=1, dtype=int32), skybox_face_width: array(ndim=1, dtype=int32), headlight_active: bool, headlight_ambient: warp._src.types.vec3f, headlight_diffuse: warp._src.types.vec3f, headlight_specular: warp._src.types.vec3f, bvh_ngeom: int, enabled_geom_ids: array(ndim=1, dtype=int32), mesh_registry: dict, mesh_bvh_id: array(ndim=1, dtype=uint64), mesh_bounds_size: array(ndim=1, dtype=vec3f), mesh_texcoord: array(ndim=1, dtype=vec2f), mesh_texcoord_offsets: array(ndim=1, dtype=int32), mesh_facetexcoord: array(ndim=1, dtype=vec3i), mesh_facenormal: array(ndim=1, dtype=vec3i), samples_per_pixel: int, aa_accum: array(ndim=1, dtype=vec3f), textures: array(ndim=1, dtype=warp._src.texture.Texture2D), textures_registry: list[warp._src.texture.Texture2D], hfield_registry: dict, hfield_bvh_id: array(ndim=1, dtype=uint64), hfield_bounds_size: array(ndim=1, dtype=vec3f), flex_mesh_registry: dict, flex_rgba: array(ndim=1, dtype=vec4f), flex_bvh_id: array(ndim=1, dtype=uint64), flex_group_root: array(ndim=2, dtype=int32), flex_render_smooth: bool, bvh_nflexgeom: int, flex_dim_np: array(ndim=1, dtype=int32), flex_geom_flexid: array(ndim=1, dtype=int32), flex_geom_edgeid: array(ndim=1, dtype=int32), bvh: warp._src.types.Bvh, bvh_id: warp._src.types.uint64, lower: array(ndim=1, dtype=vec3f), upper: array(ndim=1, dtype=vec3f), group: array(ndim=1, dtype=int32), group_root: array(ndim=1, dtype=int32), ray: array(ndim=1, dtype=vec3f), ray_offset: array(ndim=1, dtype=vec3f), rgb_data: array(ndim=1, dtype=uint32), rgb_adr: array(ndim=1, dtype=int32), depth_data: array(ndim=1, dtype=float32), depth_adr: array(ndim=1, dtype=int32), render_rgb: array(ndim=1, dtype=warp._src.types.bool), render_depth: array(ndim=1, dtype=warp._src.types.bool), seg_data: array(ndim=1, dtype=vec2i), seg_adr: array(ndim=1, dtype=int32), render_seg: array(ndim=1, dtype=warp._src.types.bool), znear: float, zfar: float, total_rays: int, enable_backface_culling: bool, shadow_light_fraction: float, enable_vertex_normals: bool, enable_specular: bool, enable_emission: bool, enable_per_light_ambient: bool, light_attenuation_is_default: bool, has_spot_lights: bool, has_orthographic_camera: bool, splat_position: array(ndim=1, dtype=vec3f), splat_rotation: array(ndim=1, dtype=quatf), splat_scale: array(ndim=1, dtype=vec3f), splat_rgba: array(ndim=1, dtype=vec4f), splat_bvh: warp._src.types.Bvh | None, splat_lower: array(ndim=1, dtype=vec3f), splat_upper: array(ndim=1, dtype=vec3f), splat_bvh_id: warp._src.types.uint64, splat_group_root: array(ndim=1, dtype=int32), splat_count: int, geom_ray_types: tuple = tuple(), _megakernel: warp._src.context.Kernel | None = None) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
def array(*args) -> warp._src.types.array:
    """
    A wrapper around wp.array that adds extra metadata to ease type introspection.
    
    Format is array(dim_1, dim_2, ..., dtype).  dim may be a constant int, or reference a size from
    Model or Data (e.g. "nq" or "nworld").  dim may also be "*", which means any nonzero size.
    """
MJ_MAXCONPAIR: int = 50
MJ_MAXIMP: float = 0.9999
MJ_MAXVAL: float = 10000000000.0
MJ_MAX_EPAFACES: int = 5
MJ_MAX_EPAHORIZON: int = 24
MJ_MINAWAKE: int = 10
MJ_MINIMP: float = 0.0001
MJ_MINMU: float = 1e-05
MJ_MINVAL: float = 1e-15
M_BLOCK_DENSE_MAX: int = 64
M_BLOCK_SCALAR_MAX: int = 6
Q_LD_BLOCK_COMPACT: int = -2
Q_LD_BLOCK_SPARSE: int = -1
TILE_SIZE_JTDAJ_DENSE: int = 16
TILE_SIZE_JTDAJ_SPARSE: int = 16
_NPLUGINATTR: int = 128
vec5 = vec5f
vec6 = vec6f
vec8 = vec8f
vec10 = vec10f
vec11 = vec11f
vec16 = vec16f
vec128 = vec_pluginattr
mat23 = mat23f
mat43 = mat43f
mat63 = mat63f
mat66 = mat66f
