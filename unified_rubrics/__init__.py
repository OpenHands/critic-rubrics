"""
Unified Rubrics Package

A comprehensive system for LLM-based feature extraction and conversation analysis.
Supports multiple rubric types including solvability analysis, conversation annotation,
and custom feature extraction.
"""

from .core import (
    RubricItem,
    RubricCategory,
    RubricSet,
    AnnotationResult,
    RubricAnnotator,
)

from .rubrics import (
    # Pre-defined rubric sets
    SOLVABILITY_RUBRICS,
    AGENT_BEHAVIORAL_RUBRICS,
    USER_FOLLOWUP_RUBRICS,
    INFRASTRUCTURE_RUBRICS,
    CONVERSATION_RUBRICS,
    
    # Rubric builders
    create_solvability_rubric,
    create_conversation_rubric,
    create_custom_rubric,
    
    # Feature lists
    DEFAULT_SOLVABILITY_FEATURES,
)

from .annotators import (
    SolvabilityAnnotator,
    ConversationAnnotator,
    CustomAnnotator,
)

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "RubricItem",
    "RubricCategory", 
    "RubricSet",
    "AnnotationResult",
    "RubricAnnotator",
    
    # Pre-defined rubrics
    "SOLVABILITY_RUBRICS",
    "AGENT_BEHAVIORAL_RUBRICS", 
    "USER_FOLLOWUP_RUBRICS",
    "INFRASTRUCTURE_RUBRICS",
    "CONVERSATION_RUBRICS",
    "DEFAULT_SOLVABILITY_FEATURES",
    
    # Builders
    "create_solvability_rubric",
    "create_conversation_rubric", 
    "create_custom_rubric",
    
    # Annotators
    "SolvabilityAnnotator",
    "ConversationAnnotator",
    "CustomAnnotator",
]