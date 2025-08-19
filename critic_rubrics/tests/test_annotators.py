"""
Tests for annotator classes.
"""

import pytest
from unittest.mock import Mock, patch

from critic_rubrics.annotators import SolvabilityAnnotator, TrajectoryAnnotator
from critic_rubrics.core import Prediction
from critic_rubrics.rubrics import SolvabilityRubrics, TrajectoryRubrics


class TestSolvabilityAnnotator:
    """Tests for SolvabilityAnnotator."""
    
    def test_init(self):
        """Test annotator initialization."""
        annotator = SolvabilityAnnotator(
            model="gpt-4o-mini",
            api_key="test-key"
        )
        
        assert annotator.model == "gpt-4o-mini"
        assert annotator.api_key == "test-key"
        assert "solvability" in annotator.system_prompt.lower()
    
    def test_tool_schema_generation(self):
        """Test tool schema generation."""
        annotator = SolvabilityAnnotator()
        schema = annotator._get_tool_schema()
        
        assert schema["type"] == "function"
        assert "function" in schema
        assert "analyze_issue_solvability" == schema["function"]["name"]
        
        params = schema["function"]["parameters"]
        assert "properties" in params
        assert "has_clear_problem_statement" in params["properties"]
        assert "has_reproduction_steps" in params["properties"]
        
        # Check that all required fields are present
        required_fields = params["required"]
        assert "has_clear_problem_statement" in required_fields
        assert len(required_fields) == 12  # Should have 12 required prediction fields
    
    def test_parse_result(self):
        """Test parsing LLM result into SolvabilityRubrics."""
        annotator = SolvabilityAnnotator()
        
        # Mock LLM response
        tool_call_args = {
            "has_clear_problem_statement": {
                "detected": True,
                "rationale": "The issue clearly states the problem"
            },
            "has_reproduction_steps": {
                "detected": False,
                "rationale": "No reproduction steps provided"
            },
            "has_expected_behavior": {
                "detected": True,
                "rationale": "Expected behavior is described"
            },
            "has_actual_behavior": {
                "detected": True,
                "rationale": "Actual behavior is described"
            },
            "has_error_messages": {
                "detected": False,
                "rationale": "No error messages included"
            },
            "has_environment_info": {
                "detected": False,
                "rationale": "No environment information"
            },
            "has_version_info": {
                "detected": False,
                "rationale": "No version information"
            },
            "has_code_examples": {
                "detected": False,
                "rationale": "No code examples"
            },
            "has_minimal_example": {
                "detected": False,
                "rationale": "No minimal example"
            },
            "has_scope_definition": {
                "detected": True,
                "rationale": "Scope is well defined"
            },
            "has_impact_description": {
                "detected": False,
                "rationale": "No impact description"
            },
            "shows_investigation_effort": {
                "detected": False,
                "rationale": "No investigation shown"
            },
            "additional_notes": "This is a well-structured issue"
        }
        
        result = annotator._parse_result(tool_call_args)
        
        assert isinstance(result, SolvabilityRubrics)
        assert result.has_clear_problem_statement.detected is True
        assert result.has_reproduction_steps.detected is False
        assert result.additional_notes == "This is a well-structured issue"
        
        # Test detection rate
        assert result.get_detection_rate() == 4/12  # 4 detected out of 12


class TestTrajectoryAnnotator:
    """Tests for TrajectoryAnnotator."""
    
    def test_init(self):
        """Test annotator initialization."""
        annotator = TrajectoryAnnotator(
            model="gpt-4o-mini",
            api_key="test-key"
        )
        
        assert annotator.model == "gpt-4o-mini"
        assert annotator.api_key == "test-key"
        assert "conversation" in annotator.system_prompt.lower()
    
    def test_tool_schema_generation(self):
        """Test tool schema generation."""
        annotator = TrajectoryAnnotator()
        schema = annotator._get_tool_schema()
        
        assert schema["type"] == "function"
        assert "function" in schema
        assert "analyze_conversation_trajectory" == schema["function"]["name"]
        
        params = schema["function"]["parameters"]
        assert "properties" in params
        
        # Check some key properties
        assert "misunderstood_intention" in params["properties"]
        assert "task_completed_successfully" in params["properties"]
        assert "user_requested_clarification" in params["properties"]
        
        # Check optional context fields
        assert "follow_up_timing" in params["properties"]
        assert "task_complexity" in params["properties"]
        assert "overall_quality_score" in params["properties"]
        
        # Check that all required prediction fields are present
        required_fields = params["required"]
        assert len(required_fields) > 40  # Should have many required prediction fields
    
    def test_parse_result_minimal(self):
        """Test parsing minimal LLM result into TrajectoryRubrics."""
        annotator = TrajectoryAnnotator()
        
        # Create minimal tool call args with all required fields
        false_pred = {"detected": False, "rationale": "Not detected"}
        true_pred = {"detected": True, "rationale": "Detected"}
        
        tool_call_args = {}
        
        # Add all required prediction fields
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
        
        for field in prediction_fields:
            if field == "task_completed_successfully":
                tool_call_args[field] = true_pred
            else:
                tool_call_args[field] = false_pred
        
        result = annotator._parse_result(tool_call_args)
        
        assert isinstance(result, TrajectoryRubrics)
        assert result.task_completed_successfully.detected is True
        assert result.misunderstood_intention.detected is False
        
        # Test helper methods
        assert result.get_issue_count() == 0  # No issues detected
        assert result.get_positive_indicators_count() == 1  # One positive indicator
    
    def test_parse_result_with_context(self):
        """Test parsing LLM result with optional context fields."""
        annotator = TrajectoryAnnotator()
        
        # Create tool call args with context
        false_pred = {"detected": False, "rationale": "Not detected"}
        
        tool_call_args = {}
        
        # Add all required prediction fields (simplified)
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
        
        for field in prediction_fields:
            tool_call_args[field] = false_pred
        
        # Add optional context
        tool_call_args.update({
            "follow_up_timing": "immediate",
            "task_complexity": "moderate",
            "task_type": "coding",
            "overall_quality_score": 0.8,
            "additional_notes": "Good conversation overall"
        })
        
        result = annotator._parse_result(tool_call_args)
        
        assert result.follow_up_timing == "immediate"
        assert result.task_complexity == "moderate"
        assert result.task_type == "coding"
        assert result.overall_quality_score == 0.8
        assert result.additional_notes == "Good conversation overall"


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.tool_calls = [Mock()]
    mock_response.choices[0].message.tool_calls[0].function = Mock()
    mock_response.choices[0].message.tool_calls[0].function.arguments = '{"has_clear_problem_statement": {"detected": true, "rationale": "Clear problem"}}'
    return mock_response


def test_annotator_integration():
    """Test basic annotator functionality without actual API calls."""
    # This test would require mocking the LLM calls
    # For now, just test that the annotators can be created
    
    solvability_annotator = SolvabilityAnnotator(model="gpt-4o-mini")
    trajectory_annotator = TrajectoryAnnotator(model="gpt-4o-mini")
    
    assert solvability_annotator.model == "gpt-4o-mini"
    assert trajectory_annotator.model == "gpt-4o-mini"
    
    # Test that tool schemas are valid
    solv_schema = solvability_annotator._get_tool_schema()
    traj_schema = trajectory_annotator._get_tool_schema()
    
    assert "function" in solv_schema
    assert "function" in traj_schema
    assert "parameters" in solv_schema["function"]
    assert "parameters" in traj_schema["function"]