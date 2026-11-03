from __future__ import annotations
from mujoco_warp._src import types
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import SleepPolicy
from mujoco_warp._src.types import SleepState
from mujoco_warp._src.types import WrapType
from mujoco_warp._src.warp_util import event_scope
import warp as wp
import warp._src.context
__all__: list[str] = ['EqType', 'K_AWAKE_VAL', 'ObjType', 'SleepPolicy', 'SleepState', 'WrapType', 'event_scope', 'sleep', 'types', 'update_sleep', 'update_sleep_trees', 'wake', 'wake_collision', 'wake_equality', 'wake_tendon', 'wp']
def sleep(*args, **kwargs):
    """
    Puts trees to sleep according to velocity tolerance.
    """
def update_sleep(*args, **kwargs):
    """
    Computes sleeping arrays from tree_asleep.
    """
def update_sleep_trees(*args, **kwargs):
    """
    Lightweight update of tree_awake array only, avoiding body/dof kernel launches.
    """
def wake(*args, **kwargs):
    """
    Wakes sleeping trees due to user changes/perturbations.
    """
def wake_collision(*args, **kwargs):
    """
    Wakes sleeping trees that touch awake trees.
    """
def wake_equality(*args, **kwargs):
    """
    Wakes sleeping trees connected by equality constraints.
    """
def wake_tendon(*args, **kwargs):
    """
    Wakes sleeping trees with constrained tendons.
    """
K_AWAKE_VAL: int = -11
_build_cycles: warp._src.context.Kernel
_check_island_can_sleep: warp._src.context.Kernel
_sleep_cycle: warp._src.context.Function
_sweep_awake_trees: warp._src.context.Kernel
_tendon_limit_active: warp._src.context.Function
_tendon_wake_val: warp._src.context.Function
_tree_can_sleep: warp._src.context.Function
_update_sleep_bodies: warp._src.context.Kernel
_update_sleep_dofs: warp._src.context.Kernel
_update_sleep_trees: warp._src.context.Kernel
_wake_collision_kernel: warp._src.context.Kernel
_wake_equality_kernel: warp._src.context.Kernel
_wake_kernel: warp._src.context.Kernel
_wake_tendon_kernel: warp._src.context.Kernel
_wake_tendon_trees: warp._src.context.Function
_wake_tree: warp._src.context.Function
_zero_sleep_counters: warp._src.context.Kernel
