from __future__ import annotations
import dataclasses as dataclasses
import functools as functools
import inspect as inspect
import warnings as warnings
import warp as wp
__all__: list[str] = ['EventTracer', 'cache_kernel', 'check_toolkit_driver', 'dataclasses', 'event_scope', 'functools', 'inspect', 'is_array_spec', 'mark_batched', 'warnings', 'wp']
class EventTracer:
    """
    Calculates elapsed times of functions annotated with `event_scope`.
    
    Use as a context manager like so:
    
      @event_trace
      def my_warp_function(...):
        ...
    
      with EventTracer() as tracer:
        my_warp_function(...)
        print(tracer.trace())
    """
    def __enter__(self):
        ...
    def __exit__(self, type, value, traceback):
        ...
    def __init__(self, enabled: bool = True):
        ...
    def trace(self) -> dict:
        """
        Calculates elapsed times for every node of the trace.
        """
def _merge(a: dict, b: dict) -> dict:
    """
    Merges two event trace stacks.
    """
def cache_kernel(func):
    ...
def check_toolkit_driver():
    ...
def event_scope(fn, name: str = ''):
    """
    Wraps a function and records an event before and after the function invocation.
    """
def is_array_spec(typ) -> bool:
    """
    Check if a type annotation is an array spec (wp.array instance or bracket annotation).
    """
def mark_batched(obj):
    """
    Recursively set _is_batched = True on all batched warp arrays within obj.
    """
_KERNEL_CACHE: dict = {}
_STACK = None
