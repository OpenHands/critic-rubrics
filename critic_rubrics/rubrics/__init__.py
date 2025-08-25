"""
Rubric dataclasses for different analysis types.
"""

from .base import BaseRubrics
from .trajectory import AnnotateConversationRubric, AnnotateConversationWithUserRubric


__all__ = [
    "BaseRubrics",
    "AnnotateConversationRubric",
    "AnnotateConversationWithUserRubric",
]
