"""
Critic Rubrics - Unified rubrics system for LLM-based feature extraction and conversation analysis.
"""

from .types import Prediction

# Optional, guarded imports so partial environments can still import package
try:
    from .rubrics.solvability import SolvabilityRubrics  # type: ignore
except Exception:  # pragma: no cover - optional module
    SolvabilityRubrics = None  # type: ignore

try:
    from .rubrics.trajectory import TrajectoryRubrics  # type: ignore
except Exception:  # pragma: no cover - optional module
    TrajectoryRubrics = None  # type: ignore

from .annotators.base import BaseAnnotator
try:
    from .annotators.solvability.annotator import SolvabilityAnnotator  # type: ignore
except Exception:  # pragma: no cover
    SolvabilityAnnotator = None  # type: ignore
try:
    from .annotators.trajectory.annotator import TrajectoryAnnotator  # type: ignore
except Exception:  # pragma: no cover
    TrajectoryAnnotator = None  # type: ignore

try:
    from .batch_processor import BatchProcessor, BatchConfig  # type: ignore
except Exception:  # pragma: no cover
    BatchProcessor = None  # type: ignore
    BatchConfig = None  # type: ignore

__version__ = "0.1.0"

# Convenience factory functions
def create_solvability_annotator(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    *,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    request_timeout: float | None = None,
) -> SolvabilityAnnotator:
    """Create a solvability annotator with configurable defaults."""
    return SolvabilityAnnotator(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )


def create_trajectory_annotator(
    model: str = "gpt-4o-mini", 
    api_key: str | None = None,
    *,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    request_timeout: float | None = None,
) -> TrajectoryAnnotator:
    """Create a trajectory annotator with configurable defaults."""
    return TrajectoryAnnotator(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )


def create_batch_processor(annotator, config: BatchConfig | None = None) -> BatchProcessor:
    """Create a batch processor for an annotator."""
    return BatchProcessor(annotator, config)


__all__ = [
    "Prediction",
    "SolvabilityRubrics",
    "TrajectoryRubrics", 
    "BaseAnnotator",
    "SolvabilityAnnotator",
    "TrajectoryAnnotator",
    "BatchProcessor",
    "BatchConfig",
    "create_solvability_annotator",
    "create_trajectory_annotator",
    "create_batch_processor",
]