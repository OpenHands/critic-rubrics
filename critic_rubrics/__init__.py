"""
Critic Rubrics Package

A comprehensive system for LLM-based feature extraction and conversation analysis.
Supports multiple rubric types including solvability analysis, trajectory annotation,
and custom feature extraction.
"""

from .core import (
    Prediction,
    RubricItem,
    RubricCategory,
    RubricSet,
    AnnotationResult,
    RubricAnnotator,
)

from .rubrics import (
    SolvabilityRubrics,
    TrajectoryRubrics,
    ConversationRubrics,
)

from .annotators import (
    BaseAnnotator,
    SolvabilityAnnotator,
    TrajectoryAnnotator,
)

# Convenience functions for creating annotators
def create_solvability_annotator(
    model: str = "gpt-4o-mini",
    api_key: str = None,
    **kwargs
) -> SolvabilityAnnotator:
    """Create a solvability annotator with default settings."""
    return SolvabilityAnnotator(model=model, api_key=api_key, **kwargs)

def create_trajectory_annotator(
    model: str = "gpt-4o-mini", 
    api_key: str = None,
    **kwargs
) -> TrajectoryAnnotator:
    """Create a trajectory annotator with default settings."""
    return TrajectoryAnnotator(model=model, api_key=api_key, **kwargs)

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "Prediction",
    "RubricItem",
    "RubricCategory", 
    "RubricSet",
    "AnnotationResult",
    "RubricAnnotator",
    
    # Rubric dataclasses
    "SolvabilityRubrics",
    "TrajectoryRubrics",
    "ConversationRubrics",
    
    # Annotators
    "BaseAnnotator",
    "SolvabilityAnnotator",
    "TrajectoryAnnotator",
    
    # Convenience functions
    "create_solvability_annotator",
    "create_trajectory_annotator",
]