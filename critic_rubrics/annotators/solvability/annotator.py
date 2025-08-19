"""
Solvability annotator implementation.
"""

from typing import Any, Dict

from ..base import BaseAnnotator
from ...rubrics.solvability import SolvabilityRubrics
from ...core import Prediction


class SolvabilityAnnotator(BaseAnnotator[SolvabilityRubrics]):
    """Annotator for analyzing issue solvability."""
    
    def _get_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "analyze_issue_solvability",
                "description": "Analyze an issue for solvability factors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "has_clear_problem_statement": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_reproduction_steps": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_expected_behavior": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_actual_behavior": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_error_messages": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_environment_info": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_version_info": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_code_examples": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_minimal_example": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_scope_definition": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "has_impact_description": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"]
                        },
                        "shows_investigation_effort": {
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
                        "has_clear_problem_statement", "has_reproduction_steps", 
                        "has_expected_behavior", "has_actual_behavior", "has_error_messages",
                        "has_environment_info", "has_version_info", "has_code_examples",
                        "has_minimal_example", "has_scope_definition", "has_impact_description",
                        "shows_investigation_effort"
                    ]
                }
            }
        }
    
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> SolvabilityRubrics:
        """Parse LLM result into SolvabilityRubrics."""
        # Convert nested dicts to Prediction objects
        predictions = {}
        for field_name in [
            "has_clear_problem_statement", "has_reproduction_steps", "has_expected_behavior",
            "has_actual_behavior", "has_error_messages", "has_environment_info",
            "has_version_info", "has_code_examples", "has_minimal_example",
            "has_scope_definition", "has_impact_description", "shows_investigation_effort"
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
        
        return SolvabilityRubrics(**predictions)