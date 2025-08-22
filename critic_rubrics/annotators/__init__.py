"""
Annotator classes for different rubric types.
"""

from .base import BaseAnnotator

try:
    from .solvability import SolvabilityAnnotator
except Exception:  # optional
    SolvabilityAnnotator = None  # type: ignore

try:
    from .trajectory import TrajectoryAnnotator
except Exception:  # optional
    TrajectoryAnnotator = None  # type: ignore

__all__ = [
    "BaseAnnotator",
    "SolvabilityAnnotator",
    "TrajectoryAnnotator",
]