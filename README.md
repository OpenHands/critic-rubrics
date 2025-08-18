# Critic Rubrics

A unified system for LLM-based feature extraction and conversation analysis.

This repository contains the unified rubrics package that consolidates different rubric systems used across All-Hands-AI research projects, including:

- **Calvin Featurizer**: LLM-based feature extraction for solvability analysis
- **Xingyao Rubrics**: Agent behavior analysis and conversation annotation
- **Custom Rubrics**: Flexible framework for creating domain-specific evaluation criteria

## Features

- 🔧 **Flexible Rubric System**: Define custom rubrics with categories, items, and additional fields
- 🤖 **Multiple LLM Backends**: Support for OpenAI, Anthropic, and other providers via LiteLLM
- 📊 **Multi-Sample Analysis**: Generate multiple samples for robust statistical analysis
- ⚡ **Batch Processing**: Efficiently process multiple items in parallel
- 📋 **Pre-built Rubrics**: Ready-to-use rubrics for solvability analysis and conversation annotation
- 🔌 **Extensible Design**: Easy to add new rubric types and annotation strategies

## Quick Start

```bash
# Install the package
pip install -e ./unified_rubrics

# Optional dependencies for LLM support
pip install litellm openai anthropic
```

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

## Documentation

- 📖 **[Package Documentation](unified_rubrics/README.md)**: Comprehensive usage guide
- 🚀 **[Migration Guide](unified_rubrics/MIGRATION_GUIDE.md)**: How to migrate from existing systems
- 📋 **[Proposal Document](UNIFIED_RUBRICS_PROPOSAL.md)**: Detailed design and architecture
- 💡 **[Examples](unified_rubrics/examples.py)**: Complete usage examples

## Repository Structure

```
critic-rubrics/
├── unified_rubrics/           # Main package
│   ├── core.py               # Core classes and interfaces
│   ├── rubrics.py            # Pre-defined rubric sets
│   ├── annotators.py         # LLM annotator implementations
│   ├── examples.py           # Usage examples
│   ├── tests/                # Test suite
│   └── README.md             # Package documentation
├── UNIFIED_RUBRICS_PROPOSAL.md  # Design proposal
└── README.md                 # This file
```

## Contributing

This package consolidates rubric systems from multiple All-Hands-AI research projects. For contributions:

1. Review the [proposal document](UNIFIED_RUBRICS_PROPOSAL.md) for architecture details
2. Check existing [examples](unified_rubrics/examples.py) for usage patterns
3. Run tests to ensure compatibility
4. Follow the established patterns for new rubric types

## License

This project is part of the All-Hands-AI research ecosystem.

