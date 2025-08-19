"""
Solvability rubrics dataclass for issue analysis.
"""

from typing import Optional
from pydantic import BaseModel, Field

from ..core import Prediction


class SolvabilityRubrics(BaseModel):
    """
    Dataclass for solvability analysis results.
    
    Each field represents a specific aspect of issue solvability,
    with both detection result and rationale.
    """
    
    # Core issue characteristics
    has_clear_problem_statement: Prediction = Field(
        description="Issue has a clear, well-defined problem statement"
    )
    
    has_reproduction_steps: Prediction = Field(
        description="Issue includes steps to reproduce the problem"
    )
    
    has_expected_behavior: Prediction = Field(
        description="Issue describes what the expected behavior should be"
    )
    
    has_actual_behavior: Prediction = Field(
        description="Issue describes what actually happens (the bug/problem)"
    )
    
    # Technical details
    has_error_messages: Prediction = Field(
        description="Issue includes relevant error messages or logs"
    )
    
    has_environment_info: Prediction = Field(
        description="Issue provides environment/system information"
    )
    
    has_version_info: Prediction = Field(
        description="Issue specifies software/library versions"
    )
    
    has_code_examples: Prediction = Field(
        description="Issue includes relevant code examples or snippets"
    )
    
    # Context and scope
    has_minimal_example: Prediction = Field(
        description="Issue provides a minimal reproducible example"
    )
    
    has_scope_definition: Prediction = Field(
        description="Issue clearly defines the scope and boundaries of the problem"
    )
    
    has_impact_description: Prediction = Field(
        description="Issue describes the impact or consequences of the problem"
    )
    
    # Research and investigation
    shows_investigation_effort: Prediction = Field(
        description="Issue shows evidence of investigation or debugging attempts"
    )
    
    # Additional context
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional observations about issue solvability"
    )
    
    def get_detection_rate(self) -> float:
        """Calculate the overall detection rate across all rubric items."""
        predictions = [
            self.has_clear_problem_statement,
            self.has_reproduction_steps,
            self.has_expected_behavior,
            self.has_actual_behavior,
            self.has_error_messages,
            self.has_environment_info,
            self.has_version_info,
            self.has_code_examples,
            self.has_minimal_example,
            self.has_scope_definition,
            self.has_impact_description,
            self.shows_investigation_effort,
        ]
        
        detected_count = sum(1 for p in predictions if p.detected)
        return detected_count / len(predictions)
    
    def get_detected_features(self) -> list[str]:
        """Get list of feature names that were detected."""
        features = []
        if self.has_clear_problem_statement.detected:
            features.append("has_clear_problem_statement")
        if self.has_reproduction_steps.detected:
            features.append("has_reproduction_steps")
        if self.has_expected_behavior.detected:
            features.append("has_expected_behavior")
        if self.has_actual_behavior.detected:
            features.append("has_actual_behavior")
        if self.has_error_messages.detected:
            features.append("has_error_messages")
        if self.has_environment_info.detected:
            features.append("has_environment_info")
        if self.has_version_info.detected:
            features.append("has_version_info")
        if self.has_code_examples.detected:
            features.append("has_code_examples")
        if self.has_minimal_example.detected:
            features.append("has_minimal_example")
        if self.has_scope_definition.detected:
            features.append("has_scope_definition")
        if self.has_impact_description.detected:
            features.append("has_impact_description")
        if self.shows_investigation_effort.detected:
            features.append("shows_investigation_effort")
        return features
    
    def get_missing_features(self) -> list[str]:
        """Get list of feature names that were not detected."""
        features = []
        if not self.has_clear_problem_statement.detected:
            features.append("has_clear_problem_statement")
        if not self.has_reproduction_steps.detected:
            features.append("has_reproduction_steps")
        if not self.has_expected_behavior.detected:
            features.append("has_expected_behavior")
        if not self.has_actual_behavior.detected:
            features.append("has_actual_behavior")
        if not self.has_error_messages.detected:
            features.append("has_error_messages")
        if not self.has_environment_info.detected:
            features.append("has_environment_info")
        if not self.has_version_info.detected:
            features.append("has_version_info")
        if not self.has_code_examples.detected:
            features.append("has_code_examples")
        if not self.has_minimal_example.detected:
            features.append("has_minimal_example")
        if not self.has_scope_definition.detected:
            features.append("has_scope_definition")
        if not self.has_impact_description.detected:
            features.append("has_impact_description")
        if not self.shows_investigation_effort.detected:
            features.append("shows_investigation_effort")
        return features