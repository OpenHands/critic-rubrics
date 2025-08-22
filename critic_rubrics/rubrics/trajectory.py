"""
Trajectory rubrics for conversation analysis.
"""

from pydantic import BaseModel, Field
from ..core import Prediction


class TrajectoryRubrics(BaseModel):
    """
    Comprehensive trajectory analysis features based on Xingyao rubrics.
    Includes user follow-up patterns, agent behavioral issues, and infrastructure problems.
    """
    
    # USER FOLLOW-UP PATTERNS
    clarification_or_restatement: Prediction = Field(
        description="User clarifies/restates or corrects interpretation"
    )
    correction: Prediction = Field(
        description="Agent understood intention but executed incorrectly"
    )
    direction_change: Prediction = Field(
        description="User adds new constraints/intent or redirects plan/scope"
    )
    vcs_update_requests: Prediction = Field(
        description="User instructs forward-moving VCS tasks (commit/push/PR)"
    )
    progress_or_scope_concern: Prediction = Field(
        description="User flags slowness, overcomplexity, or scope bloat"
    )
    frustration_or_complaint: Prediction = Field(
        description="User shows dissatisfaction or irritation"
    )
    removal_or_reversion_request: Prediction = Field(
        description="User asks to remove code/files or revert changes"
    )
    other_user_issue: Prediction = Field(
        description="Any other notable user concern not covered above"
    )
    
    # AGENT BEHAVIORAL ISSUES
    misunderstood_intention: Prediction = Field(
        description="Agent misunderstood the user's goal/intent"
    )
    did_not_follow_instruction: Prediction = Field(
        description="Agent ignored or failed to comply with explicit instructions"
    )
    insufficient_analysis: Prediction = Field(
        description="Didn't explore existing materials sufficiently before acting"
    )
    insufficient_clarification: Prediction = Field(
        description="Failed to ask necessary questions when requirements were ambiguous"
    )
    improper_tool_use_or_setup: Prediction = Field(
        description="Misused tools/commands or had missing/incorrect dependencies"
    )
    loop_behavior: Prediction = Field(
        description="Repeats the same failed action 3+ times without strategy change"
    )
    insufficient_testing: Prediction = Field(
        description="Skipped reasonable verification/tests for non-trivial changes"
    )
    insufficient_debugging: Prediction = Field(
        description="Did not investigate or reduce failing behavior when needed"
    )
    incomplete_implementation: Prediction = Field(
        description="Delivered unfinished or non-functioning work"
    )
    file_management_errors: Prediction = Field(
        description="Wrong paths, overwrites, misplaced/extra files"
    )
    scope_creep: Prediction = Field(
        description="Implemented unrequested features without approval"
    )
    risky_actions_or_permission: Prediction = Field(
        description="Risky steps without user's explicit consent"
    )
    other_agent_issue: Prediction = Field(
        description="Any other agent-side problem not covered above"
    )
    
    # INFRASTRUCTURE ISSUES
    infrastructure_external_issue: Prediction = Field(
        description="Environment/platform limits outside agent control"
    )
    infrastructure_agent_caused_issue: Prediction = Field(
        description="Infrastructure faults introduced by agent's prior actions"
    )
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score (0.0 to 1.0).

        Notes on category weighting:
        - We intentionally DO NOT penalize vcs_update_requests in quality scoring.
          These are often forward-moving workflow requests (commit/push/PR) that
          do not, by themselves, indicate poor agent behavior. They are still
          counted in get_issue_count() for observability, but excluded from the
          negative signal here.
        """
        # Count agent behavioral issues (negative indicators)
        agent_issues = [
            self.misunderstood_intention.detected,
            self.did_not_follow_instruction.detected,
            self.insufficient_analysis.detected,
            self.insufficient_clarification.detected,
            self.improper_tool_use_or_setup.detected,
            self.loop_behavior.detected,
            self.insufficient_testing.detected,
            self.insufficient_debugging.detected,
            self.incomplete_implementation.detected,
            self.file_management_errors.detected,
            self.scope_creep.detected,
            self.risky_actions_or_permission.detected,
            self.other_agent_issue.detected,
        ]
        
        # Count user follow-up issues (also negative indicators)
        user_issues = [
            self.clarification_or_restatement.detected,
            self.correction.detected,
            self.direction_change.detected,
            # Intentionally exclude vcs_update_requests from penalty
            self.progress_or_scope_concern.detected,
            self.frustration_or_complaint.detected,
            self.removal_or_reversion_request.detected,
            self.other_user_issue.detected,
        ]
        
        # Infrastructure issues (neutral - not agent's fault for external)
        infrastructure_issues = [
            self.infrastructure_agent_caused_issue.detected,  # This counts against agent
        ]
        
        total_negative = sum(agent_issues) + sum(user_issues) + sum(infrastructure_issues)
        total_possible = len(agent_issues) + len(user_issues) + len(infrastructure_issues)
        
        # Quality score: 1.0 - (issues / total_possible)
        return max(0.0, 1.0 - (total_negative / total_possible))
    
    def get_issue_count(self) -> int:
        """Count total number of issues detected."""
        all_issues = [
            # User follow-up patterns
            self.clarification_or_restatement.detected,
            self.correction.detected,
            self.direction_change.detected,
            self.vcs_update_requests.detected,
            self.progress_or_scope_concern.detected,
            self.frustration_or_complaint.detected,
            self.removal_or_reversion_request.detected,
            self.other_user_issue.detected,
            # Agent behavioral issues
            self.misunderstood_intention.detected,
            self.did_not_follow_instruction.detected,
            self.insufficient_analysis.detected,
            self.insufficient_clarification.detected,
            self.improper_tool_use_or_setup.detected,
            self.loop_behavior.detected,
            self.insufficient_testing.detected,
            self.insufficient_debugging.detected,
            self.incomplete_implementation.detected,
            self.file_management_errors.detected,
            self.scope_creep.detected,
            self.risky_actions_or_permission.detected,
            self.other_agent_issue.detected,
            # Infrastructure issues
            self.infrastructure_external_issue.detected,
            self.infrastructure_agent_caused_issue.detected,
        ]
        return sum(all_issues)
    
    def get_agent_issue_count(self) -> int:
        """Count agent-specific behavioral issues."""
        agent_issues = [
            self.misunderstood_intention.detected,
            self.did_not_follow_instruction.detected,
            self.insufficient_analysis.detected,
            self.insufficient_clarification.detected,
            self.improper_tool_use_or_setup.detected,
            self.loop_behavior.detected,
            self.insufficient_testing.detected,
            self.insufficient_debugging.detected,
            self.incomplete_implementation.detected,
            self.file_management_errors.detected,
            self.scope_creep.detected,
            self.risky_actions_or_permission.detected,
            self.other_agent_issue.detected,
            self.infrastructure_agent_caused_issue.detected,
        ]
        return sum(agent_issues)
    
    def get_user_followup_count(self) -> int:
        """Count user follow-up patterns."""
        user_issues = [
            self.clarification_or_restatement.detected,
            self.correction.detected,
            self.direction_change.detected,
            self.vcs_update_requests.detected,
            self.progress_or_scope_concern.detected,
            self.frustration_or_complaint.detected,
            self.removal_or_reversion_request.detected,
            self.other_user_issue.detected,
        ]
        return sum(user_issues)