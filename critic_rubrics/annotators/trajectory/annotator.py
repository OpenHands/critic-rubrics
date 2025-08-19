"""
Trajectory annotator implementation.
"""

from typing import Any, Dict

from ..base import BaseAnnotator
from ...rubrics.trajectory import TrajectoryRubrics
from ...core import Prediction


class TrajectoryAnnotator(BaseAnnotator[TrajectoryRubrics]):
    """Annotator for analyzing agent conversation trajectories."""
    
    def _get_tool_schema(self) -> Dict[str, Any]:
        """Generate tool schema with all trajectory rubric fields."""
        
        # Define all the fields from TrajectoryRubrics
        all_fields = [
            # User follow-up patterns
            "clarification_or_restatement",
            "correction", 
            "direction_change",
            "vcs_update_requests",
            "progress_or_scope_concern",
            "frustration_or_complaint",
            "removal_or_reversion_request",
            "other_user_issue",
            # Agent behavioral issues
            "misunderstood_intention",
            "did_not_follow_instruction",
            "insufficient_analysis",
            "insufficient_clarification",
            "improper_tool_use_or_setup",
            "loop_behavior",
            "insufficient_testing",
            "insufficient_debugging",
            "incomplete_implementation",
            "file_management_errors",
            "scope_creep",
            "risky_actions_or_permission",
            "other_agent_issue",
            # Infrastructure issues
            "infrastructure_external_issue",
            "infrastructure_agent_caused_issue",
        ]
        
        # Generate properties for each field
        properties = {}
        for field_name in all_fields:
            properties[field_name] = {
                "type": "object",
                "properties": {
                    "detected": {"type": "boolean"},
                    "rationale": {"type": "string"}
                },
                "required": ["detected", "rationale"]
            }
        
        return {
            "type": "function",
            "function": {
                "name": "analyze_conversation_trajectory",
                "description": "Analyze a conversation for trajectory patterns and issues",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": all_fields
                }
            }
        }
    
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> TrajectoryRubrics:
        """Parse LLM result into TrajectoryRubrics."""
        
        # All the fields from TrajectoryRubrics
        all_fields = [
            # User follow-up patterns
            "clarification_or_restatement",
            "correction", 
            "direction_change",
            "vcs_update_requests",
            "progress_or_scope_concern",
            "frustration_or_complaint",
            "removal_or_reversion_request",
            "other_user_issue",
            # Agent behavioral issues
            "misunderstood_intention",
            "did_not_follow_instruction",
            "insufficient_analysis",
            "insufficient_clarification",
            "improper_tool_use_or_setup",
            "loop_behavior",
            "insufficient_testing",
            "insufficient_debugging",
            "incomplete_implementation",
            "file_management_errors",
            "scope_creep",
            "risky_actions_or_permission",
            "other_agent_issue",
            # Infrastructure issues
            "infrastructure_external_issue",
            "infrastructure_agent_caused_issue",
        ]
        
        # Convert nested dicts to Prediction objects
        predictions = {}
        for field_name in all_fields:
            if field_name in tool_call_args:
                pred_data = tool_call_args[field_name]
                predictions[field_name] = Prediction(
                    detected=pred_data["detected"],
                    rationale=pred_data["rationale"]
                )
        
        return TrajectoryRubrics(**predictions)