"""
Solvability annotator implementation.
"""

from typing import Any, Dict

from ..base import BaseAnnotator
from ...rubrics.solvability import SolvabilityRubrics
from ...core import Prediction


class SolvabilityAnnotator(BaseAnnotator[SolvabilityRubrics]):
    """Annotator for analyzing issue solvability."""
    
    def _get_system_message(self) -> str:
        """Get the system message for solvability analysis."""
        return """You are an AI issue analyzer evaluating GitHub issues, bug reports, and feature requests for solvability. Your task is to identify key characteristics that indicate whether an issue can be effectively resolved.

========================
ANALYSIS FRAMEWORK
========================
Analyze the provided issue text and determine which solvability factors are present. For each factor:
1) Set the corresponding boolean to TRUE if the factor is clearly present
2) Provide a brief rationale with specific evidence from the issue text

SOLVABILITY FACTORS

PROBLEM DEFINITION
• has_clear_problem_statement: The issue clearly describes what is wrong or what needs to be implemented.
  - Look for: Clear description of the problem, specific symptoms, or well-defined requirements

• has_reproduction_steps: Specific steps are provided to reproduce the issue.
  - Look for: Numbered steps, commands to run, or clear instructions to trigger the problem

• has_expected_behavior: What the user expects to happen is clearly stated.
  - Look for: "Expected:", "Should:", or clear statements of desired outcomes

• has_actual_behavior: What actually happens (the problem) is clearly described.
  - Look for: "Actual:", "Instead:", error descriptions, or unexpected outcomes

TECHNICAL DETAILS
• has_error_messages: Specific error messages, stack traces, or error codes are included.
  - Look for: Exception traces, error logs, HTTP status codes, or specific error text

• has_environment_info: Information about the system, platform, or environment is provided.
  - Look for: OS version, browser, device info, deployment environment details

• has_version_info: Version numbers of relevant software, libraries, or tools are specified.
  - Look for: Version numbers, commit hashes, release tags, or "latest" specifications

• has_code_examples: Code snippets, configuration files, or relevant code is included.
  - Look for: Code blocks, file contents, configuration examples, or command-line usage

CONTEXT AND SCOPE
• has_minimal_example: A minimal, focused example that demonstrates the issue.
  - Look for: Simple test cases, minimal reproduction code, or stripped-down examples

• has_scope_definition: The boundaries and scope of the issue are clearly defined.
  - Look for: Specific components affected, limitations, or boundaries of the problem

• has_impact_description: The impact, severity, or consequences of the issue are described.
  - Look for: Performance impact, user experience effects, or business consequences

• shows_investigation_effort: Evidence that the reporter has investigated or attempted solutions.
  - Look for: "I tried...", research mentions, attempted solutions, or diagnostic steps taken

========================
QUALITY STANDARDS
========================
• Evidence-based: Only mark TRUE if you can point to specific text in the issue
• Conservative approach: When uncertain, mark FALSE and explain why
• Brief rationales: Keep explanations concise but specific
• Quote evidence: Reference specific phrases or sections when possible"""
    
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