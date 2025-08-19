"""
Core classes for the critic rubrics system.
"""

from pydantic import BaseModel


class Prediction(BaseModel):
    """Represents a prediction with detection result and rationale."""
    detected: bool
    rationale: str