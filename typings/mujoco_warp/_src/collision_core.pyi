"""
Core collision types and utilities shared across collision modules.
"""
from __future__ import annotations
import dataclasses as dataclasses
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import mat63f as mat63
from mujoco_warp._src.types import vec5f as vec5
import typing
from typing import Any
import warp as wp
import warp._src.codegen
import warp._src.context
import warp._src.types
__all__: list[str] = ['Any', 'CollisionContext', 'ContactType', 'Geom', 'GeomType', 'MJ_MINMU', 'MJ_MINVAL', 'contact_margin_gap', 'contact_material_params', 'contact_params', 'create_collision_context', 'dataclasses', 'geom_collision_pair', 'geom_collision_pair_from_types', 'mat63', 'safe_div', 'sap_binary_search', 'sap_range', 'vec5', 'wp', 'write_contact']
class CollisionContext:
    """
    Collision driver intermediate arrays.
    
    Attributes:
      collision_pair: collision pairs from broadphase             (naconmax, 2)
      collision_pairid: ids from broadphase                       (naconmax, 2)
      collision_worldid: collision world ids from broadphase      (naconmax,)
    """
    __annotations_cache__: typing.ClassVar[dict] = {'collision_pair': warp._src.types.array, 'collision_pairid': warp._src.types.array, 'collision_worldid': warp._src.types.array}
    __dataclass_fields__: typing.ClassVar[dict]
    __dataclass_params__: typing.ClassVar[dataclasses._DataclassParams]
    __hash__: typing.ClassVar[None] = None
    __match_args__: typing.ClassVar[tuple] = ('collision_pair', 'collision_pairid', 'collision_worldid')
    @staticmethod
    def __annotate_func__(format):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, collision_pair: warp._src.types.array, collision_pairid: warp._src.types.array, collision_worldid: warp._src.types.array) -> None:
        ...
    def __replace__(self, **changes):
        ...
    def __repr__(self):
        ...
def create_collision_context(naconmax: int) -> CollisionContext:
    """
    Create a CollisionContext with allocated arrays.
    """
Geom: warp._src.codegen.Struct
MJ_MINMU: float = 1e-05
MJ_MINVAL: float = 1e-15
contact_margin_gap: warp._src.context.Function
contact_material_params: warp._src.context.Function
contact_params: warp._src.context.Function
geom_collision_pair: warp._src.context.Function
geom_collision_pair_from_types: warp._src.context.Function
safe_div: warp._src.context.Function
sap_binary_search: warp._src.context.Function
sap_range: warp._src.context.Kernel
write_contact: warp._src.context.Function
