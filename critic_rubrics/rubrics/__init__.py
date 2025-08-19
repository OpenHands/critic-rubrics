"""
Rubric dataclasses for different analysis types.
"""

from .solvability import SolvabilityRubrics
from .trajectory import TrajectoryRubrics

__all__ = [
    "SolvabilityRubrics",
    "TrajectoryRubrics", 
]