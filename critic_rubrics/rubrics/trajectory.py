"""
Trajectory rubrics dataclass for agent conversation analysis.
"""

from typing import Optional
from pydantic import BaseModel, Field

from ..core import Prediction


class TrajectoryRubrics(BaseModel):
    """
    Comprehensive dataclass for trajectory/conversation analysis results.
    
    This represents a superset of all rubric features from different trajectory
    analysis systems, with each property mapping to a Prediction type.
    """
    
    # === AGENT BEHAVIORAL ISSUES ===
    
    # Understanding and Intent
    misunderstood_intention: Prediction = Field(
        description="Agent misunderstood the user's goal/intent"
    )
    
    ignored_user_request: Prediction = Field(
        description="Agent ignored or failed to address user's explicit request"
    )
    
    misinterpreted_context: Prediction = Field(
        description="Agent misinterpreted the context or situation"
    )
    
    # Task Execution Issues
    incomplete_task_execution: Prediction = Field(
        description="Agent left the task incomplete or partially done"
    )
    
    incorrect_approach: Prediction = Field(
        description="Agent used an incorrect or suboptimal approach"
    )
    
    unnecessary_actions: Prediction = Field(
        description="Agent performed unnecessary or redundant actions"
    )
    
    failed_to_verify: Prediction = Field(
        description="Agent failed to verify results or test their work"
    )
    
    # Communication Issues
    unclear_communication: Prediction = Field(
        description="Agent's communication was unclear or confusing"
    )
    
    insufficient_explanation: Prediction = Field(
        description="Agent provided insufficient explanation of actions"
    )
    
    overly_verbose: Prediction = Field(
        description="Agent was unnecessarily verbose or repetitive"
    )
    
    # Error Handling
    poor_error_handling: Prediction = Field(
        description="Agent handled errors poorly or ignored them"
    )
    
    failed_to_recover: Prediction = Field(
        description="Agent failed to recover from errors or setbacks"
    )
    
    repeated_same_mistake: Prediction = Field(
        description="Agent repeated the same mistake multiple times"
    )
    
    # === USER FOLLOW-UP PATTERNS ===
    
    # Clarification Requests
    user_requested_clarification: Prediction = Field(
        description="User asked for clarification or more details"
    )
    
    user_corrected_agent: Prediction = Field(
        description="User corrected the agent's understanding or approach"
    )
    
    user_provided_additional_context: Prediction = Field(
        description="User provided additional context or information"
    )
    
    # Dissatisfaction Indicators
    user_expressed_frustration: Prediction = Field(
        description="User expressed frustration or dissatisfaction"
    )
    
    user_requested_different_approach: Prediction = Field(
        description="User requested a different approach or method"
    )
    
    user_abandoned_task: Prediction = Field(
        description="User abandoned or gave up on the task"
    )
    
    # === INFRASTRUCTURE AND TECHNICAL ISSUES ===
    
    # Tool and Command Issues
    tool_usage_error: Prediction = Field(
        description="Agent made errors using tools or commands"
    )
    
    environment_setup_issue: Prediction = Field(
        description="Issues with environment setup or configuration"
    )
    
    permission_or_access_issue: Prediction = Field(
        description="Permission denied or access-related problems"
    )
    
    network_or_connectivity_issue: Prediction = Field(
        description="Network connectivity or external service issues"
    )
    
    # File and Data Handling
    file_handling_error: Prediction = Field(
        description="Errors in file operations or data handling"
    )
    
    data_corruption_or_loss: Prediction = Field(
        description="Data corruption or unintended data loss"
    )
    
    version_control_issue: Prediction = Field(
        description="Problems with git or version control operations"
    )
    
    # === CONVERSATION QUALITY METRICS ===
    
    # Efficiency
    conversation_too_long: Prediction = Field(
        description="Conversation was unnecessarily long or inefficient"
    )
    
    multiple_iterations_needed: Prediction = Field(
        description="Multiple iterations were needed to complete the task"
    )
    
    backtracking_required: Prediction = Field(
        description="Agent had to backtrack or undo previous work"
    )
    
    # Success Indicators
    task_completed_successfully: Prediction = Field(
        description="Task was completed successfully and correctly"
    )
    
    user_satisfied_with_result: Prediction = Field(
        description="User expressed satisfaction with the final result"
    )
    
    efficient_problem_solving: Prediction = Field(
        description="Agent demonstrated efficient problem-solving"
    )
    
    # === SPECIFIC BEHAVIORAL PATTERNS ===
    
    # Learning and Adaptation
    agent_learned_from_feedback: Prediction = Field(
        description="Agent successfully learned from user feedback"
    )
    
    agent_adapted_approach: Prediction = Field(
        description="Agent adapted their approach based on new information"
    )
    
    agent_asked_clarifying_questions: Prediction = Field(
        description="Agent asked appropriate clarifying questions"
    )
    
    # Proactivity
    agent_anticipated_needs: Prediction = Field(
        description="Agent anticipated user needs or potential issues"
    )
    
    agent_provided_alternatives: Prediction = Field(
        description="Agent provided alternative solutions or approaches"
    )
    
    agent_offered_additional_help: Prediction = Field(
        description="Agent offered additional help or related suggestions"
    )
    
    # === DOMAIN-SPECIFIC ISSUES ===
    
    # Code-related
    code_quality_issues: Prediction = Field(
        description="Generated code had quality, style, or best practice issues"
    )
    
    security_concerns: Prediction = Field(
        description="Code or approach introduced security vulnerabilities"
    )
    
    performance_issues: Prediction = Field(
        description="Solution had performance problems or inefficiencies"
    )
    
    # Documentation and Testing
    insufficient_documentation: Prediction = Field(
        description="Insufficient documentation or comments provided"
    )
    
    missing_tests: Prediction = Field(
        description="Failed to provide necessary tests or validation"
    )
    
    incomplete_examples: Prediction = Field(
        description="Examples or demonstrations were incomplete or unclear"
    )
    
    # === ADDITIONAL CONTEXT ===
    
    # Timing and Urgency
    follow_up_timing: Optional[str] = Field(
        default=None,
        description="Timing of user follow-up (immediate, delayed, etc.)"
    )
    
    task_complexity: Optional[str] = Field(
        default=None,
        description="Assessed complexity of the task (simple, moderate, complex)"
    )
    
    task_type: Optional[str] = Field(
        default=None,
        description="Type of task (coding, debugging, analysis, etc.)"
    )
    
    # Overall Assessment
    overall_quality_score: Optional[float] = Field(
        default=None,
        description="Overall quality score for the trajectory (0.0-1.0)"
    )
    
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any additional observations about the trajectory"
    )
    
    def get_issue_count(self) -> int:
        """Count the number of issues detected."""
        issue_predictions = [
            self.misunderstood_intention,
            self.ignored_user_request,
            self.misinterpreted_context,
            self.incomplete_task_execution,
            self.incorrect_approach,
            self.unnecessary_actions,
            self.failed_to_verify,
            self.unclear_communication,
            self.insufficient_explanation,
            self.overly_verbose,
            self.poor_error_handling,
            self.failed_to_recover,
            self.repeated_same_mistake,
            self.tool_usage_error,
            self.environment_setup_issue,
            self.permission_or_access_issue,
            self.network_or_connectivity_issue,
            self.file_handling_error,
            self.data_corruption_or_loss,
            self.version_control_issue,
            self.conversation_too_long,
            self.multiple_iterations_needed,
            self.backtracking_required,
            self.code_quality_issues,
            self.security_concerns,
            self.performance_issues,
            self.insufficient_documentation,
            self.missing_tests,
            self.incomplete_examples,
        ]
        
        return sum(1 for p in issue_predictions if p.detected)
    
    def get_positive_indicators_count(self) -> int:
        """Count the number of positive indicators detected."""
        positive_predictions = [
            self.task_completed_successfully,
            self.user_satisfied_with_result,
            self.efficient_problem_solving,
            self.agent_learned_from_feedback,
            self.agent_adapted_approach,
            self.agent_asked_clarifying_questions,
            self.agent_anticipated_needs,
            self.agent_provided_alternatives,
            self.agent_offered_additional_help,
        ]
        
        return sum(1 for p in positive_predictions if p.detected)
    
    def get_user_followup_indicators(self) -> list[str]:
        """Get list of user follow-up indicators that were detected."""
        indicators = []
        followup_predictions = [
            ("user_requested_clarification", self.user_requested_clarification),
            ("user_corrected_agent", self.user_corrected_agent),
            ("user_provided_additional_context", self.user_provided_additional_context),
            ("user_expressed_frustration", self.user_expressed_frustration),
            ("user_requested_different_approach", self.user_requested_different_approach),
            ("user_abandoned_task", self.user_abandoned_task),
        ]
        
        for name, prediction in followup_predictions:
            if prediction.detected:
                indicators.append(name)
        
        return indicators
    
    def get_quality_score(self) -> float:
        """Calculate an overall quality score based on issues and positive indicators."""
        if self.overall_quality_score is not None:
            return self.overall_quality_score
        
        # Calculate based on issues vs positive indicators
        issue_count = self.get_issue_count()
        positive_count = self.get_positive_indicators_count()
        
        # Simple scoring: start at 1.0, subtract for issues, add for positive indicators
        score = 1.0 - (issue_count * 0.05) + (positive_count * 0.1)
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1