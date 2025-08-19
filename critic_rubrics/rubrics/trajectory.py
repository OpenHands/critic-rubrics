"""
Trajectory rubrics dataclass for agent conversation analysis.
"""

from typing import Optional
from pydantic import BaseModel, Field

from ..core import Prediction


class TrajectoryRubrics(BaseModel):
    """
    Essential trajectory/conversation analysis rubrics.
    
    Focuses on the most important aspects of agent-user interactions.
    """
    
    # Agent Issues
    misunderstood_intention: Prediction = Field(
        description="Agent misunderstood the user's goal/intent"
    )
    
    incomplete_task_execution: Prediction = Field(
        description="Agent left the task incomplete or partially done"
    )
    
    unclear_communication: Prediction = Field(
        description="Agent's communication was unclear or confusing"
    )
    
    # User Patterns
    user_requested_clarification: Prediction = Field(
        description="User asked for clarification or more details"
    )
    
    user_corrected_agent: Prediction = Field(
        description="User corrected the agent's understanding or approach"
    )
    
    # Quality Indicators
    task_completed_successfully: Prediction = Field(
        description="The task was completed successfully"
    )
    
    user_satisfied_with_result: Prediction = Field(
        description="User expressed satisfaction with the result"
    )
    
    # Optional context fields
    additional_notes: Optional[str] = Field(None, description="Additional observations")
    
    def get_issue_count(self) -> int:
        """Count the number of issues detected."""
        issues = [
            self.misunderstood_intention,
            self.incomplete_task_execution,
            self.unclear_communication,
            self.user_requested_clarification,
            self.user_corrected_agent,
        ]
        return sum(1 for issue in issues if issue.detected)
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        positive_indicators = [
            self.task_completed_successfully,
            self.user_satisfied_with_result,
        ]
        positive_count = sum(1 for indicator in positive_indicators if indicator.detected)
        
        issue_count = self.get_issue_count()
        
        # Simple scoring: positive indicators boost score, issues reduce it
        base_score = positive_count / len(positive_indicators)
        penalty = min(0.5, issue_count * 0.1)  # Max 50% penalty
        
        return max(0.0, base_score - penalty)