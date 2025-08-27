# Critic Rubrics

Building blocks for LLM “rubric” tool-schemas and typed predictions.

This repo currently provides:
- Typed prediction models (binary, text, single-label classification)
- A BaseRubrics class that turns Pydantic fields into a Litellm/OpenAI tool schema

What it does NOT provide yet:
- Predefined rubrics (e.g., solvability/trajectory)
- Annotator helpers/factories

Some legacy tests reference trajectory-style rubrics for schema compatibility, but the concrete classes are not shipped yet.

## Install

Python 3.12+ is required.

- From source (editable):
```bash
pip install -e .
```
This installs runtime deps defined in pyproject.toml:
- pydantic (v2)
- litellm

For development:
```bash
uv sync --group dev
```
This sets up ruff, pyright, pytest, and pre-commit.

## Quick start: define your own rubric

```python
from typing import Literal
from pydantic import Field
from critic_rubrics import BaseRubrics, BinaryPrediction, ClassificationPrediction, TextPrediction

class SimpleConversationRubric(BaseRubrics):
    TOOL_NAME = "annotate_conversation"
    TOOL_DESCRIPTION = "Annotate an agent conversation after the work is done."

    misunderstood_intention: BinaryPrediction = Field(
        description="Agent misunderstood the user's goal/intent."
    )

    outcome: ClassificationPrediction[Literal["pass", "fail"]] = Field(
        description="High-level outcome classification."
    )

    summary: TextPrediction = Field(
        description="Short summary of the session."
    )

r = SimpleConversationRubric()
# Tool choice and tool schema for use with litellm/openai-compatible chat APIs
print(r.tool_choice)
print(r.tools)
```

You are responsible for constructing messages and parsing tool outputs. BaseRubrics focuses on generating a consistent tool schema from typed fields.

## How it works

- Each Pydantic field of a rubric must be a subclass of BasePrediction. The provided ones are:
  - BinaryPrediction: exposes `<name>_detected: bool` and `<name>_rationale: str`
  - TextPrediction: exposes `<name>_text: str`
  - ClassificationPrediction[Literal[...]]: exposes `<name>: str` (+ optional enum) and `<name>_rationale: str`
- BaseRubrics.tools flattens all fields into a single function tool with required properties by default (REQUIRED_ALL=True).
- You must implement system_message (and optionally user_message) in your rubric subclass. create_annotation_request is provided as an abstract hook if you want to standardize message formatting.

## Minimal example with litellm

```python
from litellm import completion

messages = [
    {"role": "system", "content": "You are a careful annotator."},
    {"role": "user", "content": "<paste conversation here>"},
]

rubric = SimpleConversationRubric()
resp = completion(
    model="openai/o4-mini",  # any litellm-supported provider
    messages=messages,
    tools=rubric.tools,
    tool_choice=rubric.tool_choice,
)

# Parse resp.choices[0].message.tool_calls[0].function.arguments (JSON)
# back into your own result object if desired.
```

## Project layout (current)

```
critic_rubrics/
├── __init__.py
├── prediction.py
└── rubrics/
    ├── __init__.py  # exports BaseRubrics; references trajectory types that are not yet implemented
    └── base.py      # BaseRubrics implementation
```

Note: tests include legacy-compat checks for trajectory rubrics that aren't present in the package yet.

## Contributing

- Dev environment: `uv sync --group dev`
- Run formatting/lint/type-check via pre-commit:
  - `uv run pre-commit run --all-files`
- Run tests:
  - `uv run pytest -vv` 

## Roadmap

- Ship concrete solvability/trajectory rubrics built on BaseRubrics
- Provide parsing helpers to convert tool-call JSON back into typed prediction results
- Optional annotator utilities for common providers
