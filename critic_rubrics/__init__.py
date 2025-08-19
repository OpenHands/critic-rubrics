"""
Critic Rubrics - Unified rubrics system for LLM-based feature extraction and conversation analysis.
"""

from .core import Prediction
from .rubrics.solvability import SolvabilityRubrics
from .rubrics.trajectory import TrajectoryRubrics
from .annotators.base import BaseAnnotator
from .annotators.solvability.annotator import SolvabilityAnnotator
from .annotators.trajectory.annotator import TrajectoryAnnotator

__version__ = "0.1.0"

# Convenience factory functions
def create_solvability_annotator(
    model: str = "gpt-4o-mini",
    api_key: str = None,
) -> SolvabilityAnnotator:
    """Create a solvability annotator with default settings."""
    return SolvabilityAnnotator(model=model, api_key=api_key)


def create_trajectory_annotator(
    model: str = "gpt-4o-mini", 
    api_key: str = None,
) -> TrajectoryAnnotator:
    """Create a trajectory annotator with default settings."""
    return TrajectoryAnnotator(model=model, api_key=api_key)


__all__ = [
    "Prediction",
    "SolvabilityRubrics",
    "TrajectoryRubrics", 
    "BaseAnnotator",
    "SolvabilityAnnotator",
    "TrajectoryAnnotator",
    "create_solvability_annotator",
    "create_trajectory_annotator",
]