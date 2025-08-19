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
        return {
            "type": "function",
            "function": {
                "name": "analyze_conversation_trajectory",
                "description": "Analyze a conversation for trajectory patterns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "misunderstood_intention": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "incomplete_task_execution": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "unclear_communication": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "user_requested_clarification": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "user_corrected_agent": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "task_completed_successfully": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "user_satisfied_with_result": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "additional_notes": {"type": "string"}
                    },
                    "required": [
                        "misunderstood_intention", "incomplete_task_execution", "unclear_communication",
                        "user_requested_clarification", "user_corrected_agent", 
                        "task_completed_successfully", "user_satisfied_with_result"
                    ]
                }
            }
        }
    
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> TrajectoryRubrics:
        """Parse LLM result into TrajectoryRubrics."""
        # Convert nested dicts to Prediction objects
        predictions = {}
        for field_name in [
            "misunderstood_intention", "incomplete_task_execution", "unclear_communication",
            "user_requested_clarification", "user_corrected_agent", 
            "task_completed_successfully", "user_satisfied_with_result"
        ]:
            if field_name in tool_call_args:
                pred_data = tool_call_args[field_name]
                predictions[field_name] = Prediction(
                    detected=pred_data["detected"],
                    rationale=pred_data["rationale"]
                )
        
        # Add optional fields
        if "additional_notes" in tool_call_args:
            predictions["additional_notes"] = tool_call_args["additional_notes"]
        
        return TrajectoryRubrics(**predictions)