# Critic Rubrics

A unified system for LLM-based feature extraction and conversation analysis.

This repository contains the unified rubrics package that consolidates different rubric systems used across All-Hands-AI research projects, including:

- **Solvability Analysis**: LLM-based feature extraction for issue solvability evaluation
- **Trajectory Analysis**: Agent behavior analysis and conversation quality assessment  
- **Custom Rubrics**: Flexible framework for creating domain-specific evaluation criteria

## Features

- 🔧 **Structured Rubric System**: Define rubrics with dataclasses and Prediction types (bool + rationale)
- 🤖 **Multiple LLM Backends**: Support for OpenAI, Anthropic, and other providers via LiteLLM
- 📊 **Multi-Sample Analysis**: Generate multiple samples for robust statistical analysis
- ⚡ **Batch Processing**: Efficiently process multiple items in parallel
- 📋 **Pre-built Rubrics**: Ready-to-use rubrics for solvability and trajectory analysis
- 🏗️ **Modular Architecture**: Separate annotator classes organized in folders
- 🔌 **Extensible Design**: Easy to add new rubric types and annotation strategies

## Architecture

The package is organized into separate modules:

```
critic_rubrics/
├── core.py                    # Core classes including Prediction dataclass
├── rubrics/                   # Rubric definitions as dataclasses
│   ├── solvability.py         # SolvabilityRubrics dataclass
│   ├── trajectory.py          # TrajectoryRubrics dataclass  
│   └── conversation.py        # ConversationRubrics dataclass
├── annotators/                # Annotator implementations
│   ├── base.py               # Base annotator class
│   ├── solvability/          # Solvability annotators
│   │   └── annotator.py
│   └── trajectory/           # Trajectory annotators
│       └── annotator.py
└── examples.py               # Usage examples
```

## Quick Start

### Installation

```bash
pip install critic-rubrics

# With LLM support
pip install critic-rubrics[llm]

# With analysis tools
pip install critic-rubrics[analysis]

# Full installation
pip install critic-rubrics[all]
```

### Basic Usage

#### Solvability Analysis

```python
from critic_rubrics import create_solvability_annotator

# Create an annotator
annotator = create_solvability_annotator(
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

# Analyze an issue
issue_text = """
Bug: Application crashes when loading large files
When I try to load a CSV file larger than 100MB, the application crashes.
Steps to reproduce: 1. Open app 2. Load large CSV 3. Crash occurs
Expected: File should load successfully
Actual: Application crashes with MemoryError
Environment: Windows 10, Python 3.9, 8GB RAM
"""

result = annotator.annotate(issue_text)
print(f"Detection rate: {result.get_detection_rate():.2%}")
print(f"Detected features: {result.get_detected_features()}")

# Access individual predictions
print(f"Has clear problem: {result.has_clear_problem_statement.detected}")
print(f"Rationale: {result.has_clear_problem_statement.rationale}")
```

#### Trajectory Analysis

```python
from critic_rubrics import create_trajectory_annotator

# Create trajectory annotator
annotator = create_trajectory_annotator(
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

# Analyze a conversation
conversation = """
User: I need help setting up a Python virtual environment
Agent: I'll help you set up a virtual environment. Let me create one:
python -m venv myproject
source myproject/bin/activate

User: I'm on Windows, will that work?
Agent: Good point! On Windows use: myproject\\Scripts\\activate
User: Perfect, thanks!
"""

result = annotator.annotate(conversation)
print(f"Quality score: {result.get_quality_score():.2f}")
print(f"Issues detected: {result.get_issue_count()}")
print(f"Positive indicators: {result.get_positive_indicators_count()}")

# Check specific behaviors
print(f"Agent adapted approach: {result.agent_adapted_approach.detected}")
print(f"Task completed successfully: {result.task_completed_successfully.detected}")
```

### Advanced Usage

#### Multi-Sample Analysis

```python
# Generate multiple samples for statistical reliability
multi_result = annotator.annotate_with_samples(
    content=issue_text,
    samples=5,
    temperature=0.7
)

print(f"Sample count: {multi_result.sample_count}")
print(f"Sample diversity: {multi_result.get_sample_diversity():.3f}")

# Get consensus result
consensus = multi_result.get_consensus_result(threshold=0.6)
if consensus:
    print(f"Consensus detection rate: {consensus.get_detection_rate():.2%}")
```

#### Batch Processing

```python
issues = [
    "Bug: Login button doesn't work",
    "Feature request: Add export functionality", 
    "Issue: App crashes on startup"
]

results = annotator.annotate_batch(issues, max_workers=3)
for i, result in enumerate(results):
    print(f"Issue {i+1}: {result.get_detection_rate():.2%} solvability")
```

## Rubric Types

### Solvability Rubrics

Evaluates issue reports based on:
- **Problem Definition**: Clear problem statement, expected vs actual behavior
- **Reproduction**: Steps to reproduce, minimal examples
- **Technical Details**: Error messages, environment info, version info
- **Context**: Scope definition, impact description, investigation effort

### Trajectory Rubrics

Comprehensive analysis of agent-user conversations:
- **Agent Issues**: Misunderstanding, incorrect approach, poor communication
- **User Patterns**: Follow-up requests, corrections, frustration indicators
- **Technical Issues**: Tool errors, environment problems, infrastructure issues
- **Quality Metrics**: Task completion, efficiency, user satisfaction
- **Behavioral Patterns**: Learning, adaptation, proactivity

### Prediction Structure

All rubric features use the `Prediction` dataclass:

```python
@dataclass
class Prediction:
    detected: bool      # Whether the feature was detected
    rationale: str      # Explanation for the decision
```

## Extending the System

### Creating Custom Rubrics

```python
from pydantic import BaseModel, Field
from critic_rubrics.core import Prediction

class CustomRubrics(BaseModel):
    custom_feature: Prediction = Field(
        description="Description of the custom feature"
    )
    
    def get_custom_score(self) -> float:
        return 1.0 if self.custom_feature.detected else 0.0
```

### Creating Custom Annotators

```python
from critic_rubrics.annotators.base import BaseAnnotator

class CustomAnnotator(BaseAnnotator[CustomRubrics]):
    def _get_default_system_prompt(self) -> str:
        return "You are an expert at analyzing custom content..."
    
    def _get_default_instruction_prompt(self) -> str:
        return "Analyze the following content..."
    
    def _get_tool_schema(self) -> Dict[str, Any]:
        # Define your tool schema
        pass
    
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> CustomRubrics:
        # Parse LLM response into your rubric dataclass
        pass
```

## Development

### Setup

```bash
git clone https://github.com/All-Hands-AI/critic-rubrics.git
cd critic-rubrics
pip install -e .[dev]
```

### Running Tests

```bash
pytest critic_rubrics/tests/
```

### Code Quality

```bash
black critic_rubrics/
isort critic_rubrics/
mypy critic_rubrics/
```

## Migration from Previous Systems

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions on migrating from:
- Calvin featurizer system
- Xingyao rubrics system
- Legacy annotation frameworks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{critic_rubrics,
  title={Critic Rubrics: A Unified System for LLM-based Analysis},
  author={All-Hands-AI Research Team},
  year={2024},
  url={https://github.com/All-Hands-AI/critic-rubrics}
}
```