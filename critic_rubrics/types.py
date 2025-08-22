"""
Core classes for the critic rubrics system.
"""

from pydantic import BaseModel


class Prediction(BaseModel):
    """Represents a prediction with detection result and rationale."""
    detected: bool
    rationale: str

class BatchConfig(BaseModel):
    """Configuration for batch processing.

    Notes:
    - Temperature and max_tokens are sourced from the annotator to avoid drift.
    - batch_size was removed to avoid confusion; file creation writes all requests.
    """
    provider: str = "openai"  # openai, anthropic
    max_retries: int = 3
    rate_limit_rpm: int = 60
    output_folder: str = "./batch_results"

    request_timeout: float | None = None
