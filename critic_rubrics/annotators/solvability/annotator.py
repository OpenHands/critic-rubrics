"""
Solvability annotator implementation.
"""

from typing import Any, Dict, List, Optional
from collections import Counter
from math import log2

from ..base import BaseAnnotator, MultiSampleResult
from ...rubrics.solvability import SolvabilityRubrics
from ...core import Prediction


class SolvabilityAnnotator(BaseAnnotator[SolvabilityRubrics]):
    """
    Annotator for analyzing issue solvability using comprehensive rubrics.
    
    This annotator evaluates issues based on multiple criteria including:
    - Problem statement clarity
    - Reproduction steps
    - Technical details
    - Context and scope
    - Investigation effort
    """
    
    def _get_default_system_prompt(self) -> str:
        return """You are an expert at analyzing software issues and bug reports to determine their solvability. Your task is to evaluate issues based on multiple criteria that indicate how likely they are to be successfully resolved.

You should analyze the issue content and determine whether each solvability factor is present, providing clear rationale for your decisions. Focus on concrete evidence in the issue description rather than making assumptions.

Be thorough but concise in your rationale - quote specific parts of the issue when relevant."""
    
    def _get_default_instruction_prompt(self) -> str:
        return """Analyze the following issue and evaluate it against the solvability rubrics. For each criterion, determine whether it is present in the issue and provide a brief rationale explaining your decision.

Issue to analyze:"""
    
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
                            "required": ["detected", "rationale"],
                            "description": "Issue has a clear, well-defined problem statement"
                        },
                        "has_reproduction_steps": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue includes steps to reproduce the problem"
                        },
                        "has_expected_behavior": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue describes what the expected behavior should be"
                        },
                        "has_actual_behavior": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue describes what actually happens (the bug/problem)"
                        },
                        "has_error_messages": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue includes relevant error messages or logs"
                        },
                        "has_environment_info": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue provides environment/system information"
                        },
                        "has_version_info": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue specifies software/library versions"
                        },
                        "has_code_examples": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue includes relevant code examples or snippets"
                        },
                        "has_minimal_example": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue provides a minimal reproducible example"
                        },
                        "has_scope_definition": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue clearly defines the scope and boundaries of the problem"
                        },
                        "has_impact_description": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue describes the impact or consequences of the problem"
                        },
                        "shows_investigation_effort": {
                            "type": "object",
                            "properties": {
                                "detected": {"type": "boolean"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["detected", "rationale"],
                            "description": "Issue shows evidence of investigation or debugging attempts"
                        },
                        "additional_notes": {
                            "type": "string",
                            "description": "Any additional observations about issue solvability"
                        }
                    },
                    "required": [
                        "has_clear_problem_statement",
                        "has_reproduction_steps", 
                        "has_expected_behavior",
                        "has_actual_behavior",
                        "has_error_messages",
                        "has_environment_info",
                        "has_version_info",
                        "has_code_examples",
                        "has_minimal_example",
                        "has_scope_definition",
                        "has_impact_description",
                        "shows_investigation_effort"
                    ]
                }
            }
        }
    
    def _parse_result(self, tool_call_args: Dict[str, Any]) -> SolvabilityRubrics:
        """Parse LLM result into SolvabilityRubrics dataclass."""
        # Convert nested dicts to Prediction objects
        parsed_args = {}
        
        prediction_fields = [
            "has_clear_problem_statement",
            "has_reproduction_steps", 
            "has_expected_behavior",
            "has_actual_behavior",
            "has_error_messages",
            "has_environment_info",
            "has_version_info",
            "has_code_examples",
            "has_minimal_example",
            "has_scope_definition",
            "has_impact_description",
            "shows_investigation_effort"
        ]
        
        for field in prediction_fields:
            if field in tool_call_args:
                pred_data = tool_call_args[field]
                parsed_args[field] = Prediction(
                    detected=pred_data["detected"],
                    rationale=pred_data["rationale"]
                )
        
        # Handle additional notes
        if "additional_notes" in tool_call_args:
            parsed_args["additional_notes"] = tool_call_args["additional_notes"]
        
        return SolvabilityRubrics(**parsed_args)


class SolvabilityMultiSampleResult(MultiSampleResult[SolvabilityRubrics]):
    """
    Multi-sample result for solvability analysis with specialized statistics.
    """
    
    def get_detection_rates(self) -> Dict[str, float]:
        """Get detection rates for each solvability feature."""
        if not self.samples:
            return {}
        
        features = [
            "has_clear_problem_statement",
            "has_reproduction_steps", 
            "has_expected_behavior",
            "has_actual_behavior",
            "has_error_messages",
            "has_environment_info",
            "has_version_info",
            "has_code_examples",
            "has_minimal_example",
            "has_scope_definition",
            "has_impact_description",
            "shows_investigation_effort"
        ]
        
        detection_rates = {}
        for feature in features:
            detected_count = sum(
                1 for sample in self.samples 
                if getattr(sample, feature).detected
            )
            detection_rates[feature] = detected_count / len(self.samples)
        
        return detection_rates
    
    def get_detection_entropy(self) -> Dict[str, float]:
        """Calculate entropy for each feature's detection across samples."""
        detection_rates = self.get_detection_rates()
        entropy = {}
        
        for feature, rate in detection_rates.items():
            if rate == 0.0 or rate == 1.0:
                entropy[feature] = 0.0
            else:
                entropy[feature] = -(rate * log2(rate) + (1-rate) * log2(1-rate))
        
        return entropy
    
    def get_consensus_result(self, threshold: float = 0.5) -> Optional[SolvabilityRubrics]:
        """Get consensus solvability result based on majority voting."""
        if not self.samples:
            return None
        
        detection_rates = self.get_detection_rates()
        
        # Build consensus predictions
        consensus_predictions = {}
        features = [
            "has_clear_problem_statement",
            "has_reproduction_steps", 
            "has_expected_behavior",
            "has_actual_behavior",
            "has_error_messages",
            "has_environment_info",
            "has_version_info",
            "has_code_examples",
            "has_minimal_example",
            "has_scope_definition",
            "has_impact_description",
            "shows_investigation_effort"
        ]
        
        for feature in features:
            rate = detection_rates[feature]
            detected = rate >= threshold
            
            # Aggregate rationales from samples that agree with consensus
            agreeing_rationales = [
                getattr(sample, feature).rationale
                for sample in self.samples
                if getattr(sample, feature).detected == detected
            ]
            
            # Use the most common rationale or combine them
            if agreeing_rationales:
                rationale_counter = Counter(agreeing_rationales)
                most_common_rationale = rationale_counter.most_common(1)[0][0]
                consensus_rationale = most_common_rationale
            else:
                consensus_rationale = f"Consensus: {rate:.1%} detection rate"
            
            consensus_predictions[feature] = Prediction(
                detected=detected,
                rationale=consensus_rationale
            )
        
        # Combine additional notes
        all_notes = [s.additional_notes for s in self.samples if s.additional_notes]
        consensus_notes = "; ".join(set(all_notes)) if all_notes else None
        
        return SolvabilityRubrics(
            **consensus_predictions,
            additional_notes=consensus_notes
        )
    
    def get_sample_diversity(self) -> float:
        """Calculate overall diversity across all features."""
        entropies = list(self.get_detection_entropy().values())
        return sum(entropies) / len(entropies) if entropies else 0.0