# Critic Rubrics

A unified rubrics system for LLM-based feature extraction and conversation analysis.

## Overview

This package consolidates rubric systems from multiple All-Hands-AI research projects:
- **Calvin Featurizer** (solvability analysis)
- **Xingyao Rubrics** (trajectory annotation)

## Installation

This repository is installable as a Python package.

Option A — install via editable mode (recommended for contributors):
```bash
pip install -e .
```

Minimal runtime deps:
```bash
pip install pydantic litellm
```

Optional providers (for batch APIs / clients):
```bash
# If installing from a published package:
pip install 'critic-rubrics[providers]'

# If installing locally from this repo:
pip install -e '.[providers]'
```

## Quick Start

### Solvability Analysis

```python
from critic_rubrics import create_solvability_annotator

annotator = create_solvability_annotator(
    model="gpt-4o-mini",
    # Pass api_key or rely on environment variables supported by litellm
    request_timeout=20.0,  # seconds
)

result = annotator.annotate(issue_text)
print(f"Clear problem statement: {result.has_clear_problem_statement.detected}")
```


Note: Annotators provide the full tool schema internally. You do not need to (and should not) pass custom tool schemas; this avoids schema drift.

### Trajectory Analysis

```python
from critic_rubrics import create_trajectory_annotator

annotator = create_trajectory_annotator(
    model="gpt-4o-mini",
    request_timeout=20.0,
)

result = annotator.annotate(conversation)
print(f"Quality score: {result.get_quality_score():.2f}")
```

## API Keys & Timeouts

litellm supports provider-specific environment variables. For example, to use OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

You can also pass `api_key` to the annotator factories directly, but environment
variables keep secrets out of code.

## Structure

```
critic_rubrics/
├── core.py                    # Core Prediction dataclass
├── rubrics/                   # Rubric definitions
│   ├── solvability.py         # SolvabilityRubrics (12 features)
│   └── trajectory.py          # TrajectoryRubrics (23 features)
└── annotators/                # LLM annotator implementations

Request timeouts: factories accept `request_timeout` which is propagated to litellm as `timeout`.
This helps avoid long-hanging requests and improves robustness in batch/async usage.

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