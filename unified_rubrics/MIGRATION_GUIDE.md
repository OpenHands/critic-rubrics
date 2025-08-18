# Migration Guide: Unified Rubrics System

This guide helps you migrate from existing rubric systems to the new unified approach.

## Quick Start

### Installation
```bash
# From the research directory
cd /path/to/research
pip install -e ./unified_rubrics

# Optional dependencies for LLM support
pip install litellm openai anthropic
```

### Basic Usage
```python
from unified_rubrics import create_solvability_annotator, create_conversation_annotator

# For issue solvability analysis
solvability_annotator = create_solvability_annotator(
    model="gpt-4o-mini",
    api_key="your-api-key"
)
result = solvability_annotator.annotate(issue_text)

# For conversation analysis
conversation_annotator = create_conversation_annotator(
    model="gpt-4o-mini", 
    api_key="your-api-key"
)
result = conversation_annotator.annotate(conversation_text)
```

## Migration from Calvin Featurizer

### Before (Calvin)
```python
from solvability.models.featurizer import Featurizer, Feature
from solvability.llm import completion
from solvability.models.config import LLMConfig

# Define features
features = [
    Feature(identifier="has_repro_steps", description="Has reproduction steps"),
    Feature(identifier="has_error_logs", description="Has error messages"),
]

# Create featurizer
featurizer = Featurizer(
    system_prompt="You are analyzing issue solvability...",
    message_prefix="Issue: ",
    features=features
)

# Generate embedding
embedding = featurizer.embed(
    issue_description=issue_text,
    temperature=1.0,
    samples=10,
    llm_config=llm_config
)

# Access results
detection_rates = {dim: embedding.coefficient(dim) for dim in embedding.dimensions}
```

### After (Unified)
```python
from unified_rubrics import create_multi_sample_solvability_annotator

# Define custom features (optional - defaults provided)
custom_features = [
    {"identifier": "has_repro_steps", "description": "Has reproduction steps"},
    {"identifier": "has_error_logs", "description": "Has error messages"},
]

# Create annotator
annotator = create_multi_sample_solvability_annotator(
    model="gpt-4o-mini",
    api_key="your-api-key",
    custom_features=custom_features
)

# Generate multi-sample analysis
multi_result = annotator.annotate_with_samples(
    content=issue_text,
    temperature=1.0,
    samples=10
)

# Access results
detection_rates = multi_result.get_detection_rates()
consensus = multi_result.get_consensus_result(threshold=0.5)
```

### Key Differences
- **Simpler API**: No need to manage LLM configs separately
- **Built-in Multi-sampling**: Statistical analysis included
- **Flexible Features**: Easy to customize or use defaults
- **Rich Results**: Entropy, consensus, and detailed statistics

## Migration from Xingyao Rubrics

### Before (Xingyao)
```python
# Hardcoded tool definition (300+ lines)
ANNOTATION_TOOL = {
    "type": "function",
    "function": {
        "name": "annotate_conversation",
        "description": "Annotate agent conversation...",
        "parameters": {
            "type": "object",
            "properties": {
                "misunderstood_intention_detected": {
                    "type": "boolean",
                    "description": "Agent misunderstood the user's goal/intent..."
                },
                "misunderstood_intention_rationale": {
                    "type": "string",
                    "description": "Quote evidence concisely..."
                },
                # ... 50+ more fields
            },
            "required": [...]  # All fields required
        }
    }
}

# Manual LLM call
response = completion(
    messages=[system_message, user_message],
    tools=[ANNOTATION_TOOL],
    tool_choice={"type": "function", "function": {"name": "annotate_conversation"}},
    temperature=0.0
)

# Manual result parsing
annotation_data = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
```

### After (Unified)
```python
from unified_rubrics import create_conversation_annotator

# Create annotator with built-in rubrics
annotator = create_conversation_annotator(
    model="gpt-4o-mini",
    api_key="your-api-key",
    include_timing=True,
    include_task_type=True
)

# Simple annotation
result = annotator.annotate(conversation_text)

# Structured results
detected_issues = result.get_detected_items()
follow_up_timing = result.additional_data.get('follow_up_timing')
task_type = result.additional_data.get('task_type')
```

### Key Differences
- **No Hardcoding**: Rubrics defined programmatically
- **Automatic Tool Schema**: Generated from rubric definitions
- **Structured Results**: Rich result objects with helper methods
- **Easy Customization**: Modify rubrics without touching tool schemas

## Common Migration Patterns

### 1. Custom Rubric Creation
```python
from unified_rubrics import create_custom_rubric, RubricItem, CustomAnnotator

# Define your rubric items
items = [
    RubricItem(identifier="quality_issue", description="Code quality problems detected"),
    RubricItem(identifier="security_issue", description="Security vulnerabilities found"),
]

# Create custom rubric
rubric = create_custom_rubric(
    name="code_review",
    description="Code review rubric",
    items=items,
    additional_fields={
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "required": True
        }
    }
)

# Create annotator
annotator = CustomAnnotator(
    rubric_set=rubric,
    system_prompt="You are a code reviewer...",
    instruction_prompt="Review the code and identify issues.",
    model="gpt-4o-mini"
)
```

### 2. Batch Processing
```python
# Process multiple items efficiently
items = ["item1", "item2", "item3", ...]
results = annotator.annotate_batch(items, max_workers=3)

# Analyze batch results
total_detections = sum(len(r.get_detected_items()) for r in results)
avg_detection_rate = sum(r.get_detection_rate() for r in results) / len(results)
```

### 3. Statistical Analysis
```python
# Multi-sample analysis for robust results
multi_result = annotator.annotate_with_samples(
    content, samples=10, temperature=0.7
)

# Analyze uncertainty
detection_rates = multi_result.get_detection_rates()
entropies = multi_result.get_detection_entropy()

# Get consensus with custom threshold
consensus = multi_result.get_consensus_result(threshold=0.6)

# Convert to DataFrame for analysis
df = multi_result.to_dataframe()
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure you're importing from the right location
   import sys
   sys.path.insert(0, '/path/to/research')
   from unified_rubrics import create_solvability_annotator
   ```

2. **API Key Issues**
   ```python
   import os
   annotator = create_solvability_annotator(
       model="gpt-4o-mini",
       api_key=os.getenv("OPENAI_API_KEY")  # Use environment variable
   )
   ```

3. **Custom LLM Backend**
   ```python
   from unified_rubrics.core import RubricAnnotator
   
   class MyCustomAnnotator(RubricAnnotator):
       def _call_llm(self, messages, tools, tool_choice, temperature=0.0, **kwargs):
           # Implement your LLM call here
           return your_llm_api.complete(...)
   ```

### Performance Tips

1. **Use Batch Processing**: For multiple items, use `annotate_batch()` instead of individual calls
2. **Adjust Workers**: Set `max_workers` based on your API rate limits
3. **Cache Results**: Store results to avoid re-processing the same content
4. **Temperature Settings**: Use 0.0 for consistent results, >0.0 for diversity

## Validation

### Test Your Migration
```python
# Test basic functionality
from unified_rubrics import create_solvability_annotator

annotator = create_solvability_annotator(model="gpt-4o-mini")
test_result = annotator.annotate("Test issue description")

print(f"✅ Annotation successful: {len(test_result.get_detected_items())} items detected")
print(f"✅ Detection rate: {test_result.get_detection_rate():.1%}")
```

### Compare Results
```python
# Compare old vs new system results on the same data
old_results = your_old_system.process(test_data)
new_results = [annotator.annotate(item) for item in test_data]

# Analyze differences
for old, new in zip(old_results, new_results):
    # Compare detection patterns, rates, etc.
    pass
```

## Support

For questions or issues during migration:

1. Check the [README.md](README.md) for detailed documentation
2. Review [examples.py](examples.py) for usage patterns
3. Look at the test files for implementation details
4. Create an issue in the research repository

## Next Steps

After successful migration:

1. **Optimize Performance**: Tune batch sizes and worker counts
2. **Custom Rubrics**: Create domain-specific rubrics for your use cases
3. **Integration**: Connect with your existing analysis pipelines
4. **Monitoring**: Track rubric performance and accuracy over time