# Critic Rubrics

A unified rubrics system for LLM-based feature extraction and conversation analysis.

## Overview

This package consolidates rubric systems from multiple All-Hands-AI research projects:
- **Calvin Featurizer** (solvability analysis)
- **Xingyao Rubrics** (trajectory annotation)

## Installation

```bash
pip install -e .
```

## Quick Start

### Solvability Analysis

```python
from critic_rubrics import create_solvability_annotator

annotator = create_solvability_annotator(
    model="gpt-4o-mini",
    api_key="your-api-key"
)

result = annotator.annotate(issue_text)
print(f"Clear problem statement: {result.has_clear_problem_statement.detected}")
```

### Trajectory Analysis

```python
from critic_rubrics import create_trajectory_annotator

annotator = create_trajectory_annotator(
    model="gpt-4o-mini",
    api_key="your-api-key"
)

result = annotator.annotate(conversation)
print(f"Quality score: {result.get_quality_score():.2f}")
```

## Structure

```
critic_rubrics/
├── core.py                    # Core Prediction dataclass
├── rubrics/                   # Rubric definitions
│   ├── solvability.py         # SolvabilityRubrics (12 features)
│   └── trajectory.py          # TrajectoryRubrics (23 features)
└── annotators/                # LLM annotator implementations
    ├── base.py               # BaseAnnotator
    ├── solvability/
    │   └── annotator.py
    └── trajectory/
        └── annotator.py
```

## Key Features

- **Structured Predictions**: All rubric features use `Prediction(detected: bool, rationale: str)`
- **Minimal Dependencies**: Only requires `pydantic` and `litellm`
- **Simple API**: Easy-to-use factory functions for common use cases
- **Extensible**: Clean base classes for custom rubrics

## Dependencies

- `pydantic>=2.0.0`
- `litellm` (for LLM calls)