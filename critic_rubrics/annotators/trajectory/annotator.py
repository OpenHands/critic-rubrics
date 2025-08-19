"""
Trajectory annotator implementation.
"""

from typing import Any, Dict, List, Optional
from collections import Counter

from ..base import BaseAnnotator, MultiSampleResult
from ...rubrics.trajectory import TrajectoryRubrics
from ...core import Prediction


class TrajectoryAnnotator(BaseAnnotator[TrajectoryRubrics]):
    """
    Annotator for analyzing agent conversation trajectories.
    
    This annotator evaluates conversations based on a comprehensive superset
    of rubric features including:
    - Agent behavioral issues
    - User follow-up patterns  
    - Infrastructure and technical issues
    - Conversation quality metrics
    - Domain-specific issues
    """
    
    def _get_default_system_prompt(self) -> str:
        return """You are an expert at analyzing agent-user conversations to identify behavioral patterns, issues, and quality indicators. Your task is to evaluate conversations based on multiple criteria that indicate agent performance, user satisfaction, and overall interaction quality.

You should analyze the conversation content and determine whether each factor is present, providing clear rationale for your decisions. Focus on concrete evidence in the conversation rather than making assumptions.

Pay attention to:
- Agent understanding and task execution
- User responses and follow-up patterns
- Technical issues and infrastructure problems
- Communication quality and effectiveness
- Overall conversation outcomes

Be thorough but concise in your rationale - quote specific parts of the conversation when relevant."""
    
    def _get_default_instruction_prompt(self) -> str:
        return """Analyze the following agent-user conversation and evaluate it against the trajectory rubrics. For each criterion, determine whether it is present in the conversation and provide a brief rationale explaining your decision.

Conversation to analyze:"""
    
    def _get_tool_schema(self) -> Dict[str, Any]:
        """Generate comprehensive tool schema for trajectory analysis."""
        
        # Helper function to create prediction property
        def prediction_property(description: str) -> Dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "detected": {"type": "boolean"},
                    "rationale": {"type": "string"}
                },
                "required": ["detected", "rationale"],
                "description": description
            }
        
        return {
            "type": "function",
            "function": {
                "name": "analyze_conversation_trajectory",
                "description": "Analyze an agent-user conversation for behavioral patterns and quality indicators",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # Agent Behavioral Issues
                        "misunderstood_intention": prediction_property(
                            "Agent misunderstood the user's goal/intent"
                        ),
                        "ignored_user_request": prediction_property(
                            "Agent ignored or failed to address user's explicit request"
                        ),
                        "misinterpreted_context": prediction_property(
                            "Agent misinterpreted the context or situation"
                        ),
                        "incomplete_task_execution": prediction_property(
                            "Agent left the task incomplete or partially done"
                        ),
                        "incorrect_approach": prediction_property(
                            "Agent used an incorrect or suboptimal approach"
                        ),
                        "unnecessary_actions": prediction_property(
                            "Agent performed unnecessary or redundant actions"
                        ),
                        "failed_to_verify": prediction_property(
                            "Agent failed to verify results or test their work"
                        ),
                        "unclear_communication": prediction_property(
                            "Agent's communication was unclear or confusing"
                        ),
                        "insufficient_explanation": prediction_property(
                            "Agent provided insufficient explanation of actions"
                        ),
                        "overly_verbose": prediction_property(
                            "Agent was unnecessarily verbose or repetitive"
                        ),
                        "poor_error_handling": prediction_property(
                            "Agent handled errors poorly or ignored them"
                        ),
                        "failed_to_recover": prediction_property(
                            "Agent failed to recover from errors or setbacks"
                        ),
                        "repeated_same_mistake": prediction_property(
                            "Agent repeated the same mistake multiple times"
                        ),
                        
                        # User Follow-up Patterns
                        "user_requested_clarification": prediction_property(
                            "User asked for clarification or more details"
                        ),
                        "user_corrected_agent": prediction_property(
                            "User corrected the agent's understanding or approach"
                        ),
                        "user_provided_additional_context": prediction_property(
                            "User provided additional context or information"
                        ),
                        "user_expressed_frustration": prediction_property(
                            "User expressed frustration or dissatisfaction"
                        ),
                        "user_requested_different_approach": prediction_property(
                            "User requested a different approach or method"
                        ),
                        "user_abandoned_task": prediction_property(
                            "User abandoned or gave up on the task"
                        ),
                        
                        # Infrastructure and Technical Issues
                        "tool_usage_error": prediction_property(
                            "Agent made errors using tools or commands"
                        ),
                        "environment_setup_issue": prediction_property(
                            "Issues with environment setup or configuration"
                        ),
                        "permission_or_access_issue": prediction_property(
                            "Permission denied or access-related problems"
                        ),
                        "network_or_connectivity_issue": prediction_property(
                            "Network connectivity or external service issues"
                        ),
                        "file_handling_error": prediction_property(
                            "Errors in file operations or data handling"
                        ),
                        "data_corruption_or_loss": prediction_property(
                            "Data corruption or unintended data loss"
                        ),
                        "version_control_issue": prediction_property(
                            "Problems with git or version control operations"
                        ),
                        
                        # Conversation Quality Metrics
                        "conversation_too_long": prediction_property(
                            "Conversation was unnecessarily long or inefficient"
                        ),
                        "multiple_iterations_needed": prediction_property(
                            "Multiple iterations were needed to complete the task"
                        ),
                        "backtracking_required": prediction_property(
                            "Agent had to backtrack or undo previous work"
                        ),
                        "task_completed_successfully": prediction_property(
                            "Task was completed successfully and correctly"
                        ),
                        "user_satisfied_with_result": prediction_property(
                            "User expressed satisfaction with the final result"
                        ),
                        "efficient_problem_solving": prediction_property(
                            "Agent demonstrated efficient problem-solving"
                        ),
                        
                        # Behavioral Patterns
                        "agent_learned_from_feedback": prediction_property(
                            "Agent successfully learned from user feedback"
                        ),
                        "agent_adapted_approach": prediction_property(
                            "Agent adapted their approach based on new information"
                        ),
                        "agent_asked_clarifying_questions": prediction_property(
                            "Agent asked appropriate clarifying questions"
                        ),
                        "agent_anticipated_needs": prediction_property(
                            "Agent anticipated user needs or potential issues"
                        ),
                        "agent_provided_alternatives": prediction_property(
                            "Agent provided alternative solutions or approaches"
                        ),
                        "agent_offered_additional_help": prediction_property(
                            "Agent offered additional help or related suggestions"
                        ),
                        
                        # Domain-specific Issues
                        "code_quality_issues": prediction_property(
                            "Generated code had quality, style, or best practice issues"
                        ),
                        "security_concerns": prediction_property(
                            "Code or approach introduced security vulnerabilities"
                        ),
                        "performance_issues": prediction_property(
                            "Solution had performance problems or inefficiencies"
                        ),
                        "insufficient_documentation": prediction_property(
                            "Insufficient documentation or comments provided"
                        ),
                        "missing_tests": prediction_property(
                            "Failed to provide necessary tests or validation"
                        ),
                        "incomplete_examples": prediction_property(
                            "Examples or demonstrations were incomplete or unclear"
                        ),
                        
                        # Additional Context
                        "follow_up_timing": {
                            "type": "string",
                            "enum": ["immediate", "quick", "delayed", "very_delayed", "none"],
                            "description": "Timing of user follow-up"
                        },
                        "task_complexity": {
                            "type": "string", 
                            "enum": ["simple", "moderate", "complex", "very_complex"],
                            "description": "Assessed complexity of the task"
                        },
                        "task_type": {
                            "type": "string",
                            "enum": ["coding", "debugging", "analysis", "research", "configuration", "other"],
                            "description": "Type of task"
                        },
                        "overall_quality_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Overall quality score for the trajectory (0.0-1.0)"
                        },
                        "additional_notes": {
                            "type": "string",
                            "description": "Any additional observations about the trajectory"
                        }
                    },
                    "required": [
                        # All prediction fields are required
                        "misunderstood_intention", "ignored_user_request", "misinterpreted_context",
                        "incomplete_task_execution", "incorrect_approach", "unnecessary_actions",
                        "failed_to_verify", "unclear_communication", "insufficient_explanation",
                        "overly_verbose", "poor_error_handling", "failed_to_recover", "repeated_same_mistake",
                        "user_requested_clarification", "user_corrected_agent", "user_provided_additional_context",
                        "user_expressed_frustration", "user_requested_different_approach", "user_abandoned_task",
                        "tool_usage_error", "environment_setup_issue", "permission_or_access_issue",
                        "network_or_connectivity_issue", "file_handling_error", "data_corruption_or_loss",
                        "version_control_issue", "conversation_too_long", "multiple_iterations_needed",
                        "backtracking_required", "task_completed_successfully", "user_satisfied_with_result",
                        "efficient_problem_solving", "agent_learned_from_feedback", "agent_adapted_approach",
                        "agent_asked_clarifying_questions", "agent_anticipated_needs", "agent_provided_alternatives",
                        "agent_offered_additional_help", "code_quality_issues", "security_concerns",
                        "performance_issues", "insufficient_documentation", "missing_tests", "incomplete_examples"
                    ]
                }
            }
        }
    
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> TrajectoryRubrics:
        """Parse LLM result into TrajectoryRubrics dataclass."""
        parsed_args = {}
        
        # All the prediction fields
        prediction_fields = [
            "misunderstood_intention", "ignored_user_request", "misinterpreted_context",
            "incomplete_task_execution", "incorrect_approach", "unnecessary_actions",
            "failed_to_verify", "unclear_communication", "insufficient_explanation",
            "overly_verbose", "poor_error_handling", "failed_to_recover", "repeated_same_mistake",
            "user_requested_clarification", "user_corrected_agent", "user_provided_additional_context",
            "user_expressed_frustration", "user_requested_different_approach", "user_abandoned_task",
            "tool_usage_error", "environment_setup_issue", "permission_or_access_issue",
            "network_or_connectivity_issue", "file_handling_error", "data_corruption_or_loss",
            "version_control_issue", "conversation_too_long", "multiple_iterations_needed",
            "backtracking_required", "task_completed_successfully", "user_satisfied_with_result",
            "efficient_problem_solving", "agent_learned_from_feedback", "agent_adapted_approach",
            "agent_asked_clarifying_questions", "agent_anticipated_needs", "agent_provided_alternatives",
            "agent_offered_additional_help", "code_quality_issues", "security_concerns",
            "performance_issues", "insufficient_documentation", "missing_tests", "incomplete_examples"
        ]
        
        # Convert prediction fields
        for field in prediction_fields:
            if field in tool_call_args:
                pred_data = tool_call_args[field]
                parsed_args[field] = Prediction(
                    detected=pred_data["detected"],
                    rationale=pred_data["rationale"]
                )
        
        # Handle optional context fields
        optional_fields = [
            "follow_up_timing", "task_complexity", "task_type", 
            "overall_quality_score", "additional_notes"
        ]
        
        for field in optional_fields:
            if field in tool_call_args:
                parsed_args[field] = tool_call_args[field]
        
        return TrajectoryRubrics(**parsed_args)


class TrajectoryMultiSampleResult(MultiSampleResult[TrajectoryRubrics]):
    """
    Multi-sample result for trajectory analysis with specialized statistics.
    """
    
    def get_issue_detection_rates(self) -> Dict[str, float]:
        """Get detection rates for issue-related features."""
        if not self.samples:
            return {}
        
        issue_features = [
            "misunderstood_intention", "ignored_user_request", "misinterpreted_context",
            "incomplete_task_execution", "incorrect_approach", "unnecessary_actions",
            "failed_to_verify", "unclear_communication", "insufficient_explanation",
            "overly_verbose", "poor_error_handling", "failed_to_recover", "repeated_same_mistake",
            "tool_usage_error", "environment_setup_issue", "permission_or_access_issue",
            "network_or_connectivity_issue", "file_handling_error", "data_corruption_or_loss",
            "version_control_issue", "conversation_too_long", "multiple_iterations_needed",
            "backtracking_required", "code_quality_issues", "security_concerns",
            "performance_issues", "insufficient_documentation", "missing_tests", "incomplete_examples"
        ]
        
        detection_rates = {}
        for feature in issue_features:
            detected_count = sum(
                1 for sample in self.samples 
                if getattr(sample, feature).detected
            )
            detection_rates[feature] = detected_count / len(self.samples)
        
        return detection_rates
    
    def get_positive_detection_rates(self) -> Dict[str, float]:
        """Get detection rates for positive indicator features."""
        if not self.samples:
            return {}
        
        positive_features = [
            "task_completed_successfully", "user_satisfied_with_result", "efficient_problem_solving",
            "agent_learned_from_feedback", "agent_adapted_approach", "agent_asked_clarifying_questions",
            "agent_anticipated_needs", "agent_provided_alternatives", "agent_offered_additional_help"
        ]
        
        detection_rates = {}
        for feature in positive_features:
            detected_count = sum(
                1 for sample in self.samples 
                if getattr(sample, feature).detected
            )
            detection_rates[feature] = detected_count / len(self.samples)
        
        return detection_rates
    
    def get_average_quality_score(self) -> float:
        """Get average quality score across samples."""
        scores = [
            sample.get_quality_score() 
            for sample in self.samples
        ]
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_consensus_result(self, threshold: float = 0.5) -> Optional[TrajectoryRubrics]:
        """Get consensus trajectory result based on majority voting."""
        if not self.samples:
            return None
        
        # Get detection rates for all prediction fields
        all_features = [
            "misunderstood_intention", "ignored_user_request", "misinterpreted_context",
            "incomplete_task_execution", "incorrect_approach", "unnecessary_actions",
            "failed_to_verify", "unclear_communication", "insufficient_explanation",
            "overly_verbose", "poor_error_handling", "failed_to_recover", "repeated_same_mistake",
            "user_requested_clarification", "user_corrected_agent", "user_provided_additional_context",
            "user_expressed_frustration", "user_requested_different_approach", "user_abandoned_task",
            "tool_usage_error", "environment_setup_issue", "permission_or_access_issue",
            "network_or_connectivity_issue", "file_handling_error", "data_corruption_or_loss",
            "version_control_issue", "conversation_too_long", "multiple_iterations_needed",
            "backtracking_required", "task_completed_successfully", "user_satisfied_with_result",
            "efficient_problem_solving", "agent_learned_from_feedback", "agent_adapted_approach",
            "agent_asked_clarifying_questions", "agent_anticipated_needs", "agent_provided_alternatives",
            "agent_offered_additional_help", "code_quality_issues", "security_concerns",
            "performance_issues", "insufficient_documentation", "missing_tests", "incomplete_examples"
        ]
        
        consensus_predictions = {}
        
        for feature in all_features:
            detected_count = sum(
                1 for sample in self.samples 
                if getattr(sample, feature).detected
            )
            rate = detected_count / len(self.samples)
            detected = rate >= threshold
            
            # Get most common rationale from agreeing samples
            agreeing_rationales = [
                getattr(sample, feature).rationale
                for sample in self.samples
                if getattr(sample, feature).detected == detected
            ]
            
            if agreeing_rationales:
                rationale_counter = Counter(agreeing_rationales)
                consensus_rationale = rationale_counter.most_common(1)[0][0]
            else:
                consensus_rationale = f"Consensus: {rate:.1%} detection rate"
            
            consensus_predictions[feature] = Prediction(
                detected=detected,
                rationale=consensus_rationale
            )
        
        # Handle optional fields with majority voting
        timing_values = [s.follow_up_timing for s in self.samples if s.follow_up_timing]
        complexity_values = [s.task_complexity for s in self.samples if s.task_complexity]
        type_values = [s.task_type for s in self.samples if s.task_type]
        
        consensus_timing = Counter(timing_values).most_common(1)[0][0] if timing_values else None
        consensus_complexity = Counter(complexity_values).most_common(1)[0][0] if complexity_values else None
        consensus_type = Counter(type_values).most_common(1)[0][0] if type_values else None
        
        # Average quality score
        avg_quality = self.get_average_quality_score()
        
        # Combine notes
        all_notes = [s.additional_notes for s in self.samples if s.additional_notes]
        consensus_notes = "; ".join(set(all_notes)) if all_notes else None
        
        return TrajectoryRubrics(
            **consensus_predictions,
            follow_up_timing=consensus_timing,
            task_complexity=consensus_complexity,
            task_type=consensus_type,
            overall_quality_score=avg_quality,
            additional_notes=consensus_notes
        )