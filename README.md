# Critic Rubrics

Type-safe function-calling-based LLM-as-judge evaluation framework for structured prediction and analysis.

> [!WARNING]
> This repository is an active research project. APIs and implementations are subject to major changes. Use with caution.

## To Install

```bash
pip install git+https://github.com/All-Hands-AI/critic-rubrics
```

## Core Data Structures

### Prediction Types

All predictions inherit from `BasePrediction` and define how data is flattened into OpenAI tool schemas:

```python
from critic_rubrics import BinaryPrediction, TextPrediction, ClassificationPrediction
from typing import Literal

# Boolean detection with evidence
class BinaryPrediction(BasePrediction):
    detected: bool          # Flattened as: <name>_detected
    rationale: str         # Flattened as: <name>_rationale

# Free text output  
class TextPrediction(BasePrediction):
    text: str              # Flattened as: <name>_text

# Single-label classification with evidence
class ClassificationPrediction[L](BasePrediction):
    label: L               # Flattened as: <name> (with enum constraint)
    rationale: str         # Flattened as: <name>_rationale
```

### Feature Definition

```python
from critic_rubrics import Feature

feature = Feature(
    name="task_complexity",
    description="Assess the complexity level of the given task",
    prediction_type=ClassificationPrediction[Literal["simple", "moderate", "complex"]]
)
```

### Feature Data

```python
from critic_rubrics import FeatureData

# Combines feature definition with actual prediction data
feature_data = FeatureData(
    feature=feature,
    prediction=ClassificationPrediction(label="complex", rationale="Multiple dependencies")
)
```

## Core APIs

### Rubric Definition

```python
from critic_rubrics import BaseRubrics, Feature, BinaryPrediction, ClassificationPrediction
from typing import Literal, Any
from litellm import ChatCompletionRequest

class TaskAnalysisRubric(BaseRubrics):
    def __init__(self):
        super().__init__(
            tool_name="analyze_task",
            tool_description="Analyze task characteristics and complexity",
            features=[
                Feature(
                    name="requires_clarification",
                    description="Task requires additional clarification from user",
                    prediction_type=BinaryPrediction
                ),
                Feature(
                    name="complexity_level", 
                    description="Overall complexity assessment",
                    prediction_type=ClassificationPrediction[Literal["simple", "moderate", "complex"]]
                )
            ],
            system_message="You are an expert task analyzer.",
            user_message="Analyze the following task:"
        )
    
    def create_annotation_request(self, inputs: dict[str, Any], model: str = "openai/o3-2025-04-16") -> ChatCompletionRequest:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"{self.user_message}\n\n{inputs['task_description']}"}
            ],
            "tools": self.tools,
            "tool_choice": self.tool_choice
        }
```

### Tool Schema Generation

```python
rubric = TaskAnalysisRubric()

# OpenAI-compatible tool schema
print(rubric.tools)
# [{"type": "function", "function": {"name": "analyze_task", "parameters": {...}}}]

print(rubric.tool_choice) 
# {"type": "function", "function": {"name": "analyze_task"}}
```

### LLM Integration

```python
from critic_rubrics import Annotator
from litellm import completion

# Single request
rubric = TaskAnalysisRubric()
request = rubric.create_annotation_request({"task_description": "Build a web scraper"})
response = Annotator.annotate(request, model="openai/gpt-4")

# Extract structured data
tool_call = response.choices[0].message.tool_calls[0]
feature_data_list = rubric.tool_call_to_feature_data(tool_call)

for feature_data in feature_data_list:
    print(f"{feature_data.feature.name}: {feature_data.prediction.to_dict()}")
```

### Batch Processing

```python
# Send batch requests
requests = [rubric.create_annotation_request({"task_description": task}) for task in tasks]
batch_ids = Annotator.batch_annotate(
    requests, 
    output_dir="./batch_results",
    custom_llm_provider="openai",
    model="openai/gpt-4"
)

# Retrieve results
for batch_id in batch_ids:
    status, results = Annotator.get_batch_results(batch_id, custom_llm_provider="openai")
    if status["status"] == "completed":
        for result in results:
            # Process batch result
            pass
```

## Data Flow

```
1. Define Rubric → 2. Generate Tool Schema → 3. Send to LLM → 4. Parse Response → 5. Typed FeatureData
     ↓                      ↓                      ↓                ↓                    ↓
BaseRubrics.tools    ChatCompletionRequest    ModelResponse    tool_call_to_feature_data()    List[FeatureData]
```

## Key Methods

### Prediction Methods
- `to_tool_properties(field_name, field_description, rationale_description)` → `dict[str, Any]`
- `from_tool_args(feature_name, tool_args)` → `BasePrediction`
- `to_dict()` → `dict[str, Any]`

### Rubric Methods  
- `tools` → `list[ChatCompletionToolParam]`
- `tool_choice` → `ChatCompletionToolChoiceObjectParam`
- `create_annotation_request(inputs, model)` → `ChatCompletionRequest`
- `tool_call_to_feature_data(tool_call)` → `list[FeatureData]`

### Annotator Methods
- `annotate(request, **kwargs)` → `ModelResponse`
- `batch_annotate(requests, output_dir, custom_llm_provider, **kwargs)` → `list[str]`
- `get_batch_results(batch_id, custom_llm_provider, **kwargs)` → `tuple[dict, list[dict]]`

## Installation

```bash
# Runtime dependencies
pip install -e .

# Development setup  
uv sync --group dev
```

## Requirements

- Python 3.12+
- pydantic >= 2.11.7
- litellm >= 1.76.0

## Example: Complete Workflow

```python
from critic_rubrics import BaseRubrics, Feature, BinaryPrediction, Annotator
from typing import Any
from litellm import ChatCompletionRequest

# 1. Define rubric
class CodeReviewRubric(BaseRubrics):
    def __init__(self):
        super().__init__(
            tool_name="review_code",
            tool_description="Review code for potential issues",
            features=[
                Feature("has_bugs", "Code contains potential bugs", BinaryPrediction),
                Feature("needs_refactor", "Code needs refactoring", BinaryPrediction)
            ],
            system_message="You are a senior code reviewer."
        )
    
    def create_annotation_request(self, inputs: dict[str, Any], model: str = "openai/gpt-4") -> ChatCompletionRequest:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"Review this code:\n\n{inputs['code']}"}
            ],
            "tools": self.tools,
            "tool_choice": self.tool_choice
        }

# 2. Use rubric
rubric = CodeReviewRubric()
request = rubric.create_annotation_request({"code": "def add(a, b): return a + b"})
response = Annotator.annotate(request)

# 3. Extract results
tool_call = response.choices[0].message.tool_calls[0]
features = rubric.tool_call_to_feature_data(tool_call)

for feature_data in features:
    pred = feature_data.prediction
    print(f"{feature_data.feature.name}: {pred.detected} - {pred.rationale}")
```
