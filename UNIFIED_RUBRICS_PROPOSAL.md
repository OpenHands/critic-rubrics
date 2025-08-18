# Unified Rubrics System Proposal

## Executive Summary

This proposal presents a unified Python package that consolidates and standardizes the different rubric systems currently used across All-Hands-AI research projects. The package merges functionality from:

1. **Calvin Featurizer** (`calvin/packages/solvability/models/featurizer.py`) - LLM-based feature extraction for solvability analysis
2. **Non-User Traces Rubrics** (`xingyao/2025_prod_data/annotate_non_user_traj/2_process_non_user_traces_to_rubrics_requests.py`) - Agent behavior analysis without user follow-up
3. **Post-Finish Traces Rubrics** (`xingyao/2025_prod_data/process_prod_data_to_dataset/4_process_postfinish_traces_to_rubrics_requests.py`) - Comprehensive conversation analysis with user interactions

## Problem Statement

### Current State Issues

1. **Code Duplication**: Significant overlap between the two Xingyao rubric files (~90% shared rubric items)
2. **Inconsistent Interfaces**: Different APIs and patterns across the three systems
3. **Maintenance Burden**: Changes need to be made in multiple places
4. **Limited Reusability**: Hard to adapt existing rubrics for new use cases
5. **No Standardization**: Different prompt formats, tool schemas, and result structures

### Specific Pain Points

- **Calvin Featurizer**: Great multi-sample approach but limited to simple boolean features
- **Xingyao Rubrics**: Comprehensive but hardcoded, difficult to customize or extend
- **Integration Challenges**: No easy way to combine different rubric types or create hybrid analyses

## Proposed Solution

### Architecture Overview

The unified system is built around four core concepts:

1. **RubricItem**: Individual evaluation criteria (e.g., "has_clear_requirements")
2. **RubricCategory**: Groups of related items with optional mutual exclusivity rules
3. **RubricSet**: Complete evaluation frameworks combining categories and standalone items
4. **RubricAnnotator**: Configurable annotators supporting different LLM backends

### Key Features

#### 1. Flexible Rubric Definition
```python
# Simple feature extraction (Calvin-style)
solvability_rubric = create_solvability_rubric([
    {"identifier": "has_repro_steps", "description": "Issue includes reproduction steps"},
    {"identifier": "has_error_logs", "description": "Issue includes error messages"}
])

# Complex conversation analysis (Xingyao-style)
conversation_rubric = create_conversation_rubric(
    include_timing=True,
    include_task_type=True
)

# Custom rubrics for new use cases
custom_rubric = create_custom_rubric(
    name="code_quality",
    items=[...],
    additional_fields={"complexity": {"type": "string", "enum": [...]}}
)
```

#### 2. Multi-Backend LLM Support
```python
# LiteLLM for multi-provider support
annotator = LiteLLMAnnotator(rubric_set, model="gpt-4o-mini")

# Direct OpenAI integration
annotator = OpenAIAnnotator(rubric_set, model="gpt-4o-mini")

# Custom backends
class CustomAnnotator(RubricAnnotator):
    def _call_llm(self, messages, tools, tool_choice, **kwargs):
        return your_llm_api.complete(...)
```

#### 3. Multi-Sample Analysis (Calvin-inspired)
```python
# Generate multiple samples for robust statistics
multi_result = annotator.annotate_with_samples(
    content, samples=10, temperature=0.7
)

# Analyze uncertainty and consensus
detection_rates = multi_result.get_detection_rates()
entropies = multi_result.get_detection_entropy()
consensus = multi_result.get_consensus_result(threshold=0.6)
```

#### 4. Batch Processing
```python
# Efficient parallel processing
results = annotator.annotate_batch(
    contents_list, max_workers=5
)
```

## Migration Path

### From Calvin Featurizer

**Before:**
```python
from solvability.models.featurizer import Featurizer, Feature

features = [Feature(identifier="has_repro", description="Has reproduction steps")]
featurizer = Featurizer(system_prompt="...", message_prefix="...", features=features)
embedding = featurizer.embed(issue_text, samples=10)
```

**After:**
```python
from unified_rubrics import create_multi_sample_solvability_annotator

annotator = create_multi_sample_solvability_annotator(
    custom_features=[{"identifier": "has_repro", "description": "Has reproduction steps"}]
)
result = annotator.annotate_with_samples(issue_text, samples=10)
```

### From Xingyao Rubrics

**Before:**
```python
# Hardcoded ANNOTATION_TOOL dictionary with 300+ lines
ANNOTATION_TOOL = {
    "type": "function",
    "function": {
        "name": "annotate_conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "misunderstood_intention_detected": {"type": "boolean", ...},
                "misunderstood_intention_rationale": {"type": "string", ...},
                # ... 50+ more fields
            }
        }
    }
}
```

**After:**
```python
from unified_rubrics import create_conversation_annotator

annotator = create_conversation_annotator(
    model="gpt-4o-mini",
    include_timing=True,
    include_task_type=True
)
result = annotator.annotate(conversation_text)
```

## Benefits

### 1. Reduced Code Duplication
- **Before**: ~800 lines of duplicated rubric definitions
- **After**: Single source of truth with reusable components

### 2. Improved Maintainability
- Centralized rubric definitions
- Consistent API across all use cases
- Easy to add new rubric types or modify existing ones

### 3. Enhanced Flexibility
- Mix and match rubric categories
- Custom additional fields (timing, task type, etc.)
- Support for mutual exclusivity rules
- Configurable rationale requirements

### 4. Better Analysis Capabilities
- Multi-sample statistical analysis
- Uncertainty quantification via entropy
- Consensus building across samples
- Integration with pandas for data analysis

### 5. Standardized Interface
- Consistent result format across all annotators
- Unified tool schema generation
- Standard batch processing capabilities

## Implementation Details

### Core Classes

```python
class RubricItem(BaseModel):
    identifier: str
    description: str
    category: Optional[str] = None
    requires_rationale: bool = True

class RubricCategory(BaseModel):
    name: str
    description: str
    items: List[RubricItem] = []
    mutually_exclusive: bool = False
    max_selections: Optional[int] = None

class RubricSet(BaseModel):
    name: str
    description: str
    categories: List[RubricCategory] = []
    standalone_items: List[RubricItem] = []
    additional_fields: Dict[str, Dict[str, Any]] = {}

class AnnotationResult(BaseModel):
    rubric_set_name: str
    detections: Dict[str, bool] = {}
    rationales: Dict[str, str] = {}
    additional_data: Dict[str, Any] = {}
    # Metadata
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    response_latency: Optional[float] = None
```

### Pre-defined Rubric Sets

1. **SOLVABILITY_RUBRICS**: Issue completeness and solvability analysis
2. **AGENT_BEHAVIORAL_RUBRICS**: Agent failure pattern detection
3. **USER_FOLLOWUP_RUBRICS**: User intervention classification
4. **INFRASTRUCTURE_RUBRICS**: Infrastructure issue detection
5. **CONVERSATION_RUBRICS**: Comprehensive conversation analysis

### Specialized Annotators

1. **SolvabilityAnnotator**: Optimized for issue analysis
2. **ConversationAnnotator**: Full conversation analysis with timing
3. **CustomAnnotator**: Fully configurable for new use cases

## Testing Strategy

The package includes comprehensive tests covering:

- Core functionality (rubric items, categories, sets)
- Tool schema generation and validation
- Annotation result processing
- Multi-sample statistical analysis
- Pre-defined rubric correctness
- Migration compatibility

## Performance Considerations

### Efficiency Improvements

1. **Batch Processing**: Parallel annotation with configurable worker pools
2. **Caching**: Optional response caching for repeated analyses
3. **Token Optimization**: Efficient prompt construction and tool schema generation
4. **Streaming**: Support for streaming responses where available

### Resource Management

- Configurable rate limiting for API calls
- Memory-efficient batch processing
- Optional result persistence for large-scale analyses

## Future Extensions

### Planned Features

1. **Rubric Validation**: Automatic validation of rubric definitions
2. **Result Visualization**: Built-in plotting and analysis tools
3. **Rubric Composition**: Combine multiple rubric sets dynamically
4. **Active Learning**: Identify uncertain cases for human review
5. **Rubric Evolution**: Track rubric performance and suggest improvements

### Integration Opportunities

1. **OpenHands Integration**: Direct integration with OpenHands evaluation pipeline
2. **Dashboard Support**: Real-time rubric analysis in research dashboard
3. **MLflow Integration**: Experiment tracking and model versioning
4. **Database Backends**: Direct integration with research databases

## Adoption Plan

### Phase 1: Core Implementation (Completed)
- ✅ Core classes and interfaces
- ✅ Pre-defined rubric sets
- ✅ Basic annotator implementations
- ✅ Comprehensive test suite
- ✅ Documentation and examples

### Phase 2: Migration Support
- Create migration utilities for existing codebases
- Provide backward compatibility layers
- Update existing scripts to use unified system
- Performance benchmarking against current systems

### Phase 3: Advanced Features
- Multi-sample analysis optimization
- Advanced statistical analysis tools
- Integration with research infrastructure
- Custom LLM backend implementations

### Phase 4: Ecosystem Integration
- OpenHands evaluation pipeline integration
- Research dashboard integration
- Automated rubric performance monitoring
- Community contribution guidelines

## Risk Assessment

### Technical Risks

1. **Performance Regression**: Mitigation through benchmarking and optimization
2. **API Compatibility**: Careful design of migration path and backward compatibility
3. **LLM Provider Changes**: Abstraction layer protects against provider-specific changes

### Adoption Risks

1. **Learning Curve**: Comprehensive documentation and examples minimize this
2. **Migration Effort**: Automated migration tools and gradual adoption strategy
3. **Feature Gaps**: Extensible design allows for rapid feature addition

## Conclusion

The Unified Rubrics System represents a significant improvement over the current fragmented approach to rubric-based analysis. By consolidating functionality, standardizing interfaces, and providing flexible extension points, this system will:

1. **Reduce maintenance burden** through code consolidation
2. **Improve research velocity** through standardized, reusable components
3. **Enable new research directions** through flexible rubric composition
4. **Enhance result quality** through multi-sample analysis and uncertainty quantification

The implementation is complete and ready for adoption, with a clear migration path for existing systems and extensive documentation for new users.

## Next Steps

1. **Review and Feedback**: Gather feedback from research team members
2. **Integration Planning**: Plan integration with existing research workflows
3. **Migration Timeline**: Establish timeline for migrating existing systems
4. **Training and Documentation**: Prepare training materials for team adoption
5. **Performance Validation**: Benchmark against existing systems to ensure performance parity

The unified rubrics system is positioned to become the standard foundation for all rubric-based analysis across All-Hands-AI research projects.