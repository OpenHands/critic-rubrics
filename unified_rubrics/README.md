# Unified Rubrics

A comprehensive Python package for LLM-based feature extraction and conversation analysis. This package unifies different rubric systems used across the All-Hands-AI research projects, providing a consistent interface for annotation tasks.

## Features

- **Flexible Rubric System**: Define custom rubrics with categories, items, and additional fields
- **Multiple LLM Backends**: Support for OpenAI, Anthropic, and other providers via LiteLLM
- **Multi-Sample Analysis**: Generate multiple samples for robust statistical analysis
- **Batch Processing**: Efficiently process multiple items in parallel
- **Pre-built Rubrics**: Ready-to-use rubrics for solvability analysis and conversation annotation
- **Extensible Design**: Easy to add new rubric types and annotation strategies

## Installation

```bash
# Install the package (from the research directory)
pip install -e ./unified_rubrics

# Install optional dependencies
pip install litellm openai anthropic  # For LLM support
pip install pandas matplotlib seaborn  # For analysis and visualization
```

## Quick Start

### Solvability Analysis

Analyze whether issues/bugs are solvable based on the information provided:

```python
from unified_rubrics import create_solvability_annotator

# Create annotator
annotator = create_solvability_annotator(
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

# Analyze an issue
issue_text = """
Bug: Login form doesn't work on mobile Safari

When I try to log in using Safari on my iPhone 13, the login button doesn't respond.
Steps to reproduce: 1) Open app in Safari 2) Enter credentials 3) Tap login
Expected: Should log in. Actual: Button doesn't respond.
Environment: iPhone 13, iOS 16.2, Safari
"""

result = annotator.annotate(issue_text)
print(f"Detected features: {result.get_detected_items()}")
print(f"Detection rate: {result.get_detection_rate():.1%}")
```

### Conversation Analysis

Analyze agent-user conversations to identify failure patterns:

```python
from unified_rubrics import create_conversation_annotator

# Create annotator
annotator = create_conversation_annotator(
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

# Analyze a conversation
conversation = """
User: Create a Python script to parse CSV files.

Agent: Here's a basic CSV parser:
```python
import csv
def parse_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
```

User: That's not what I need. I need to convert CSV to JSON, not print rows.
"""

result = annotator.annotate(conversation)
print(f"Follow-up timing: {result.additional_data['follow_up_timing']}")
print(f"Issues detected: {result.get_detected_items()}")
```

### Multi-Sample Analysis

Generate multiple samples for robust statistical analysis:

```python
from unified_rubrics import create_multi_sample_solvability_annotator

annotator = create_multi_sample_solvability_annotator(
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

# Generate 10 samples with temperature=0.7 for diversity
multi_result = annotator.annotate_with_samples(
    issue_text, 
    samples=10, 
    temperature=0.7
)

# Analyze results
detection_rates = multi_result.get_detection_rates()
entropies = multi_result.get_detection_entropy()
consensus = multi_result.get_consensus_result(threshold=0.6)

print(f"Detection rates: {detection_rates}")
print(f"Consensus result: {consensus.get_detected_items()}")
```

### Custom Rubrics

Create your own rubrics for specific analysis needs:

```python
from unified_rubrics import create_custom_rubric, CustomAnnotator, RubricItem

# Define custom rubric items
items = [
    RubricItem(
        identifier="has_tests",
        description="The code includes unit tests"
    ),
    RubricItem(
        identifier="has_documentation", 
        description="The code includes docstrings or comments"
    ),
]

# Create custom rubric
rubric = create_custom_rubric(
    name="code_quality",
    description="Evaluate code quality characteristics",
    items=items,
    additional_fields={
        "complexity": {
            "type": "string",
            "enum": ["simple", "moderate", "complex"],
            "required": True
        }
    }
)

# Create annotator
annotator = CustomAnnotator(
    rubric_set=rubric,
    system_prompt="You are a code quality analyzer.",
    instruction_prompt="Analyze the code quality.",
    model="gpt-4o-mini"
)

result = annotator.annotate(code_text)
```

## Architecture

The package is organized into several key components:

### Core Classes

- **`RubricItem`**: Individual evaluation criteria with identifier and description
- **`RubricCategory`**: Groups of related items with optional mutual exclusivity rules
- **`RubricSet`**: Complete evaluation framework combining categories and standalone items
- **`AnnotationResult`**: Results of annotation with detections, rationales, and metadata
- **`RubricAnnotator`**: Abstract base class for all annotators

### Pre-built Rubrics

- **Solvability Rubrics**: Analyze issue completeness and solvability
- **Agent Behavioral Rubrics**: Identify agent failure patterns
- **User Follow-up Rubrics**: Classify user intervention types
- **Infrastructure Rubrics**: Detect infrastructure-related issues
- **Conversation Rubrics**: Comprehensive conversation analysis

### Annotator Types

- **`LiteLLMAnnotator`**: Multi-provider support via LiteLLM
- **`OpenAIAnnotator`**: Direct OpenAI API integration
- **`MultiSampleAnnotator`**: Generate multiple samples for robust analysis
- **`SolvabilityAnnotator`**: Specialized for issue solvability analysis
- **`ConversationAnnotator`**: Specialized for conversation analysis
- **`CustomAnnotator`**: Fully customizable annotator

## Advanced Usage

### Batch Processing

Process multiple items efficiently:

```python
issues = ["Bug 1 description", "Bug 2 description", "Feature request"]
results = annotator.annotate_batch(issues, max_workers=3)

for i, result in enumerate(results):
    print(f"Issue {i+1}: {result.get_detection_rate():.1%} detection rate")
```

### Statistical Analysis

Analyze multi-sample results:

```python
# Convert to DataFrame for analysis
df = multi_result.to_dataframe()

# Calculate statistics
import pandas as pd
detection_stats = df.groupby('sample_id')[['has_clear_requirements', 'has_reproduction_steps']].mean()

# Visualize results
import matplotlib.pyplot as plt
detection_rates.plot(kind='bar')
plt.title('Feature Detection Rates')
plt.show()
```

### Custom LLM Backends

Extend the system with custom LLM integrations:

```python
from unified_rubrics.core import RubricAnnotator

class CustomLLMAnnotator(RubricAnnotator):
    def _call_llm(self, messages, tools, tool_choice, temperature=0.0, **kwargs):
        # Implement your custom LLM call here
        return your_llm_api.complete(
            messages=messages,
            tools=tools,
            temperature=temperature
        )
    
    def _extract_annotation_from_response(self, response):
        # Custom response parsing
        return parse_your_response_format(response)
```

## Migration from Existing Systems

### From Calvin Featurizer

```python
# Old Calvin approach
from solvability.models.featurizer import Featurizer, Feature

features = [
    Feature(identifier="has_repro", description="Has reproduction steps"),
    Feature(identifier="has_error", description="Has error messages"),
]
featurizer = Featurizer(system_prompt="...", message_prefix="...", features=features)
embedding = featurizer.embed(issue_text, samples=10)

# New unified approach
from unified_rubrics import create_solvability_annotator

custom_features = [
    {"identifier": "has_repro", "description": "Has reproduction steps"},
    {"identifier": "has_error", "description": "Has error messages"},
]
annotator = create_solvability_annotator(custom_features=custom_features)
result = annotator.annotate_with_samples(issue_text, samples=10)
```

### From Xingyao Rubrics

```python
# Old approach with hardcoded rubrics
ANNOTATION_TOOL = {...}  # Large hardcoded tool definition

# New unified approach
from unified_rubrics import CONVERSATION_RUBRICS, CustomAnnotator

annotator = CustomAnnotator(
    rubric_set=CONVERSATION_RUBRICS,
    system_prompt=CONVERSATION_SYSTEM_PROMPT,
    instruction_prompt=CONVERSATION_INSTRUCTION_PROMPT
)
```

## Contributing

To add new rubric types or annotator backends:

1. Define rubric items using `RubricItem` and `RubricCategory`
2. Create rubric sets with `create_custom_rubric()` or `RubricSet`
3. Implement custom annotators by extending `RubricAnnotator`
4. Add appropriate system and instruction prompts
5. Include examples and tests

## Examples

See `examples.py` for comprehensive usage examples including:

- Basic solvability and conversation analysis
- Multi-sample statistical analysis
- Custom rubric creation
- Batch processing
- Integration with pandas for data analysis

## License

This package is part of the All-Hands-AI research project.