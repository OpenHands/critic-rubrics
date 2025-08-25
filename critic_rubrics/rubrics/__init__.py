"""
Rubric dataclasses for different analysis types.
"""

from .base import BaseRubrics
from .trajectory import AnnotateConversationRubric


__all__ = [
    "BaseRubrics",
    "AnnotateConversationRubric",
]
