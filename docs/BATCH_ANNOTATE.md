# Batch Annotation Workflow

Complete guide to the 3-step batch annotation pipeline using the `scripts/batch_annotate/` examples.

## Overview

The batch annotation system processes large datasets of conversation traces through LLM evaluation in three stages:

1. **Send Requests** - Generate and upload annotation requests to LiteLLM batch API
2. **Download Results** - Poll batch status and retrieve completed annotations  
3. **Consolidate Data** - Parse responses into typed `FeatureData` objects

## Data Flow

```
Trace Files → Annotation Requests → Batch Upload → LLM Processing → Raw Results → Typed FeatureData
    ↓              ↓                    ↓              ↓               ↓              ↓
  .jsonl.gz    ChatCompletionRequest   batch_ids    ModelResponse   .jsonl        JSON objects
```

## Step 1: Send Annotation Requests

**Script:** `scripts/batch_annotate/1_send_annotation_requests.py`

### Input Data Format

Trace files contain conversation segments in JSONL format:

```python
# Input: trace_file.jsonl.gz
{
    "conversation_id": "conv_12345",
    "segment_id": "seg_001", 
    "trace_segment": {
        "trace": [
            {"role": "user", "content": "Help me debug this code"},
            {"role": "assistant", "content": "I'll analyze your code..."}
        ],  # i.e., "messages" for OpenAI completion format
        "tools": [...],  # Available tools
        "follow_up_user_message": {"role": "user", "content": "Thanks!"} | None # Optional field
    }
}
```

### Key APIs Called

```python
from critic_rubrics.rubrics import get_trajectory_level_rubrics
from critic_rubrics import Annotator

# 1. Select appropriate rubric based on conversation structure
has_user_follow_up = trace_segment["follow_up_user_message"] is not None
rubric = get_trajectory_level_rubrics(has_user_follow_up=has_user_follow_up)

# 2. Generate annotation request from trace data
annotation_request = rubric.create_annotation_request(
    inputs={
        "messages": messages,  # List[ChatMessage]
        "tools": trace_segment["tools"]  # List[Tool]
    }
)

# 3. Send batch requests to LiteLLM
batch_ids = Annotator.batch_annotate(
    requests=requests,  # Iterable[ChatCompletionRequest]
    output_dir="./batch_results",
    custom_llm_provider="openai",
    model="openai/o3-2025-04-16",
    max_requests=10_000,
    max_bytes=100 * 1024 * 1024
)
```

### Expected Data Types

- **Input:** `Iterable[ChatCompletionRequest]`
- **Output:** `list[str]` (batch IDs)
- **Side Effects:** Creates batch metadata files in `output_dir/`

### Usage Example

```bash
python scripts/batch_annotate/1_send_annotation_requests.py \
    --trace-dir ./data/traces \
    --pattern "*.jsonl.gz" \
    --output-dir ./batch_results \
    --model "openai/o3-2025-04-16" \
    --model-provider openai \
    --limit 1000
```

## Step 2: Download Annotations

**Script:** `scripts/batch_annotate/2_download_annotations.py`

### Key APIs Called

```python
from critic_rubrics import Annotator

# Poll batch status and download results
status, results = Annotator.get_batch_results(
    batch_id="batch_12345",
    custom_llm_provider="openai",
    base_url="https://llm-proxy.eval.all-hands.dev",
    api_key=api_key
)
```

### Expected Data Types

**Input:** `str` (batch_id)

**Output:** `tuple[dict[str, Any], list[dict[str, Any]]]`
- `status`: Batch metadata with completion status
- `results`: List of batch result objects

### Status Response Format

```python
status = {
    "batch_id": "batch_12345",
    "status": "completed" | "in_progress" | "failed",
    "created_at": 1234567890,
    "completed_at": 1234567890,
    "request_counts": {"total": 100, "completed": 100, "failed": 0},
    "error": False
}
```

### Results Format

```python
# Each result in the results list:
{
    "id": "batch_12345_req_001",
    "custom_id": "req__conv_12345__seg_001", 
    "response": {
        "status_code": 200,
        "body": {
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function", 
                        "function": {
                            "name": "annotate_conversation",
                            "arguments": "{\"task_type\": \"Debug Code\", ...}"
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 150, "completion_tokens": 75}
        }
    }
}
```

### Usage Example

```bash
python scripts/batch_annotate/2_download_annotations.py \
    --batch-dir ./batch_results \
    --model-provider openai \
    --poll \
    --poll-interval 30
```

## Step 3: Consolidate and Convert Outputs

**Script:** `scripts/batch_annotate/3_consolidate_and_convert_outputs.py`

### Key APIs Called

```python
from litellm.types.utils import ModelResponse
from critic_rubrics.rubrics.trajectory import (
    annotate_conversation_rubrics,
    annotate_conversation_with_user_rubrics
)

# 1. Parse batch output to ModelResponse
model_response = ModelResponse(**response_data["body"])
tool_call = model_response.choices[0].message.tool_calls[0]

# 2. Match rubric and convert to FeatureData
if annotate_conversation_rubrics.tool_call_match_rubrics(tool_call):
    feature_data_list = annotate_conversation_rubrics.tool_call_to_feature_data(tool_call)
elif annotate_conversation_with_user_rubrics.tool_call_match_rubrics(tool_call):
    feature_data_list = annotate_conversation_with_user_rubrics.tool_call_to_feature_data(tool_call)

# 3. Convert to serializable format
features = {
    fd.feature.name: fd.prediction.to_dict()
    for fd in feature_data_list
}
```

### Expected Data Types

**Input:** Batch output JSONL files with `ModelResponse` data

**Output:** Consolidated JSONL with typed feature data

### Output Format

```python
# Final consolidated output per conversation:
{
    "batch_id": "batch_12345_req_001",
    "custom_id": "req__conv_12345__seg_001",
    "model": "openai/o3-2025-04-16",
    "usage": {"prompt_tokens": 150, "completion_tokens": 75},
    "features": {
        "task_type": {
            "type": "ClassificationPrediction",
            "label": "Debug Code",
            "rationale": "User explicitly asks to debug code"
        },
        "misunderstood_intention": {
            "type": "BinaryPrediction", 
            "detected": False,
            "rationale": ""
        },
        "user_goal_summary": {
            "type": "TextPrediction",
            "text": "User wants help debugging their Python code"
        }
    },
    "feature_count": 3
}
```

### Usage Example

```bash
python scripts/batch_annotate/3_consolidate_and_convert_outputs.py \
    ./batch_results \
    --output-name consolidated_features.jsonl
```

## Complete Workflow Example

```bash
# Step 1: Send requests
python scripts/batch_annotate/1_send_annotation_requests.py \
    --trace-dir ./data/conversation_traces \
    --output-dir ./batch_results_20240127 \
    --model "openai/o3-2025-04-16" \
    --model-provider openai \
    --limit 5000

# Step 2: Download results (with polling)
python scripts/batch_annotate/2_download_annotations.py \
    --batch-dir ./batch_results_20240127 \
    --model-provider openai \
    --poll \
    --poll-interval 60

# Step 3: Consolidate outputs
python scripts/batch_annotate/3_consolidate_and_convert_outputs.py \
    ./batch_results_20240127 \
    --output-name trajectory_features.jsonl
```

## Key Data Transformations

### 1. Trace → ChatCompletionRequest

```python
# Input trace segment
trace_data = {
    "trace": [{"role": "user", "content": "..."}, ...],
    "tools": [...],
    "follow_up_user_message": {...}
}

# Output request
request = {
    "model": "openai/o3-2025-04-16",
    "messages": [...],  # Formatted conversation
    "tools": rubric.tools,  # Generated tool schema
    "tool_choice": rubric.tool_choice,
    "metadata": {"custom_request_id": "req__conv_123__seg_001"}
}
```

### 2. ModelResponse → FeatureData

```python
# Input: LLM tool call
tool_call = {
    "function": {
        "name": "annotate_conversation",
        "arguments": '{"task_type": "Debug", "task_type_rationale": "..."}'
    }
}

# Output: Typed feature data
feature_data = FeatureData(
    feature=Feature(name="task_type", ...),
    prediction=ClassificationPrediction(label="Debug", rationale="...")
)
```

### 3. FeatureData → JSON

```python
# Serializable output
{
    "task_type": {
        "type": "ClassificationPrediction",
        "label": "Debug", 
        "rationale": "User explicitly asks for debugging help"
    }
}
```

## Error Handling

The scripts handle common failure modes:

- **Invalid JSON arguments:** Raises `PredictionMissingFieldError`
- **Wrong function names:** Raises `ValueError` 
- **Missing required fields:** Raises `PredictionMissingFieldError`
- **Batch failures:** Downloads error files to `*_errors.jsonl`
- **Network issues:** Implements exponential backoff retry

## Performance Considerations

- **Batch size limits:** 50K requests or 200MB per batch
- **Rate limiting:** Handled by LiteLLM proxy
- **Memory usage:** Processes files in streaming fashion
- **Disk space:** Temporary batch files deleted after upload
