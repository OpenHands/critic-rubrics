"""
Rubric dataclasses for different analysis types.
"""

try:
    from .solvability import SolvabilityRubrics  # type: ignore
except Exception:  # pragma: no cover
    SolvabilityRubrics = None  # type: ignore
try:
    from .trajectory import TrajectoryRubrics  # type: ignore
except Exception:  # pragma: no cover
    TrajectoryRubrics = None  # type: ignore

from .trajectory_with_user_followup import BaseRubrics, TrajectoryUserFollowupRubrics

__all__ = [
    "SolvabilityRubrics",
    "TrajectoryRubrics",
    "BaseRubrics",
    "TrajectoryUserFollowupRubrics",
]