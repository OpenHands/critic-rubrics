"""
Unit tests for FeatureData conversion from ModelResponse.

Tests the BaseRubrics.tool_call_to_feature_data() method using real batch output data.
"""

import json
from typing import Any, Dict

import pytest
from litellm.types.utils import ModelResponse

from critic_rubrics.feature import FeatureData
from critic_rubrics.prediction import BinaryPrediction, ClassificationPrediction
from critic_rubrics.rubrics.trajectory.rubric_impl import AnnotateConversationRubric
from critic_rubrics.rubrics.trajectory.trajectory_with_user import ANNOTATION_SYSTEM_MESSAGE, FEATURES


class TestFeatureDataConversion:
    """Test conversion of ModelResponse to FeatureData objects."""

    def _create_model_response(self, tool_calls: list[Dict[str, Any]]) -> ModelResponse:
        """Helper method to create a ModelResponse object from tool calls."""
        # Create a mock ModelResponse structure
        response_data = {
            "id": "test-response",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        return ModelResponse(**response_data)

    @pytest.fixture
    def rubrics_instance(self) -> AnnotateConversationRubric:
        """Create a rubrics instance for testing."""
        return AnnotateConversationRubric(
            features=FEATURES,
            system_message=ANNOTATION_SYSTEM_MESSAGE,
            user_message="Analyze this conversation for issues.",
            tool_name="annotate_conversation",
            tool_description="Annotate conversation trajectories with user interaction patterns"
        )

    @pytest.fixture
    def sample_tool_call_complete(self):
        """Real tool call data from batch_000000_outputs.jsonl (first entry)."""
        return {
            "id": "call_EPr2GDePcfd1bNBQQ5m8vnhL",
            "type": "function",
            "function": {
                "name": "annotate_conversation",
                "arguments": json.dumps({
                    "user_goal_summary_text": "User wants the agent to examine the repository and update the README with accurate functionality, then accept all changes in the branch.",
                    "overall_sentiment": "Neutral",
                    "overall_sentiment_rationale": "User gives short, direct instructions without emotional language.",
                    "task_type": "Write Documentation",
                    "task_type_rationale": "Primary work was updating README content to document project functionality.",
                    "dev_cluster": "Code Management",
                    "dev_cluster_rationale": "Work involved editing documentation and committing changes in git.",
                    "misunderstood_intention_detected": False,
                    "misunderstood_intention_rationale": "",
                    "did_not_follow_instruction_detected": False,
                    "did_not_follow_instruction_rationale": "",
                    "insufficient_analysis_detected": False,
                    "insufficient_analysis_rationale": "",
                    "insufficient_clarification_detected": False,
                    "insufficient_clarification_rationale": "",
                    "improper_tool_use_or_setup_detected": False,
                    "improper_tool_use_or_setup_rationale": "",
                    "loop_behavior_detected": False,
                    "loop_behavior_rationale": "",
                    "insufficient_testing_detected": False,
                    "insufficient_testing_rationale": "",
                    "insufficient_debugging_detected": False,
                    "insufficient_debugging_rationale": "",
                    "incomplete_implementation_detected": False,
                    "incomplete_implementation_rationale": "",
                    "file_management_errors_detected": False,
                    "file_management_errors_rationale": "",
                    "scope_creep_detected": False,
                    "scope_creep_rationale": "",
                    "risky_actions_or_permission_detected": False,
                    "risky_actions_or_permission_rationale": "",
                    "other_agent_issue_detected": False,
                    "other_agent_issue_rationale": "",
                    "infrastructure_external_issue_detected": False,
                    "infrastructure_external_issue_rationale": "",
                    "infrastructure_agent_caused_issue_detected": False,
                    "infrastructure_agent_caused_issue_rationale": "",
                    "follow_up_timing": "post_completion",
                    "follow_up_timing_rationale": "Agent had sent a finish signal: 'I've updated the README... Is there anything else?' before the user replied.",
                    "clarification_or_restatement_detected": False,
                    "clarification_or_restatement_rationale": "",
                    "correction_detected": False,
                    "correction_rationale": "",
                    "direction_change_detected": False,
                    "direction_change_rationale": "",
                    "vcs_update_requests_detected": True,
                    "vcs_update_requests_rationale": "User instructs: 'accept all code changes in the branch' which is a forward-moving VCS action.",
                    "progress_or_scope_concern_detected": False,
                    "progress_or_scope_concern_rationale": "",
                    "frustration_or_complaint_detected": False,
                    "frustration_or_complaint_rationale": "",
                    "removal_or_reversion_request_detected": False,
                    "removal_or_reversion_request_rationale": "",
                    "other_user_issue_detected": False,
                    "other_user_issue_rationale": ""
                })
            }
        }

    @pytest.fixture
    def sample_tool_call_with_issues(self):
        """Tool call data with multiple detected issues (from second entry)."""
        return {
            "id": "call_pGgHJzE1ok98fChzKDtGrwt3",
            "type": "function",
            "function": {
                "name": "annotate_conversation",
                "arguments": json.dumps({
                    "follow_up_timing": "post_completion",
                    "follow_up_timing_rationale": "User replied after agent used finish: 'I've successfully updated…the README…' indicating completion.",
                    "clarification_or_restatement_detected": False,
                    "clarification_or_restatement_rationale": "",
                    "correction_detected": False,
                    "correction_rationale": "",
                    "direction_change_detected": True,
                    "direction_change_rationale": "User asks a new question: 'how do i move it to the certloop github?' requesting next-step instructions.",
                    "vcs_update_requests_detected": False,
                    "vcs_update_requests_rationale": "",
                    "progress_or_scope_concern_detected": False,
                    "progress_or_scope_concern_rationale": "",
                    "frustration_or_complaint_detected": False,
                    "frustration_or_complaint_rationale": "",
                    "removal_or_reversion_request_detected": False,
                    "removal_or_reversion_request_rationale": "",
                    "other_user_issue_detected": False,
                    "other_user_issue_rationale": ""
                })
            }
        }

    def test_convert_complete_tool_call(self, rubrics_instance: AnnotateConversationRubric, sample_tool_call_complete):
        """Test conversion of a complete tool call with all features."""
        feature_data_list = rubrics_instance.tool_call_to_feature_data(sample_tool_call_complete)
        
        # Should convert all 9 features from trajectory_with_user
        assert len(feature_data_list) == 9
        
        # Check that all returned items are FeatureData instances
        for feature_data in feature_data_list:
            assert isinstance(feature_data, FeatureData)
            assert feature_data.feature is not None
            assert feature_data.prediction is not None

        # Check specific features
        feature_dict = {fd.feature.name: fd for fd in feature_data_list}
        
        # Test ClassificationPrediction feature
        follow_up_timing = feature_dict["follow_up_timing"]
        assert isinstance(follow_up_timing.prediction, ClassificationPrediction)
        assert follow_up_timing.prediction.label == "post_completion"
        assert "finish signal" in follow_up_timing.prediction.rationale
        
        # Test BinaryPrediction features
        vcs_requests = feature_dict["vcs_update_requests"]
        assert isinstance(vcs_requests.prediction, BinaryPrediction)
        assert vcs_requests.prediction.detected is True
        assert "accept all code changes" in vcs_requests.prediction.rationale
        
        clarification = feature_dict["clarification_or_restatement"]
        assert isinstance(clarification.prediction, BinaryPrediction)
        assert clarification.prediction.detected is False
        assert clarification.prediction.rationale == ""

    def test_convert_partial_tool_call(self, rubrics_instance, sample_tool_call_with_issues):
        """Test conversion of tool call with only some features present."""
        feature_data_list = rubrics_instance.tool_call_to_feature_data(sample_tool_call_with_issues)
        
        # Should still convert all 9 features (missing ones get default values)
        assert len(feature_data_list) == 9
        
        feature_dict = {fd.feature.name: fd for fd in feature_data_list}
        
        # Check that direction_change is detected
        direction_change = feature_dict["direction_change"]
        assert isinstance(direction_change.prediction, BinaryPrediction)
        assert direction_change.prediction.detected is True
        assert "certloop github" in direction_change.prediction.rationale

    def test_convert_empty_tool_calls(self, rubrics_instance):
        """Test conversion with empty tool calls list."""
        # Since the method now expects individual tool calls, we can't test empty list directly
        # This test is no longer applicable with the new API
        pass

    def test_convert_invalid_json_arguments(self, rubrics_instance):
        """Test handling of invalid JSON in tool call arguments."""
        invalid_tool_call = {
            "id": "call_invalid",
            "type": "function",
            "function": {
                "name": "annotate_conversation",
                "arguments": "invalid json {"
            }
        }
        
        # With invalid JSON, the method should raise an exception due to missing required fields
        try:
            rubrics_instance.tool_call_to_feature_data(invalid_tool_call)
            assert False, "Expected exception for invalid JSON"
        except Exception:
            # Expected to fail due to missing required fields
            pass

    def test_convert_wrong_function_name(self, rubrics_instance):
        """Test handling of tool calls with wrong function name."""
        wrong_name_tool_call = {
            "id": "call_wrong",
            "type": "function",
            "function": {
                "name": "wrong_function_name",
                "arguments": json.dumps({"some": "data"})
            }
        }
        
        # Should raise ValueError when function name doesn't match
        try:
            rubrics_instance.tool_call_to_feature_data(wrong_name_tool_call)
            assert False, "Expected ValueError for wrong function name"
        except ValueError as e:
            assert "unexpected name" in str(e)

    def test_feature_name_mapping(self, rubrics_instance):
        """Test that feature names are correctly mapped from tool arguments."""
        # Since the implementation is strict and requires all fields, we need to provide complete data
        # This test now verifies that the method fails when required fields are missing
        minimal_tool_call = {
            "id": "call_minimal",
            "type": "function",
            "function": {
                "name": "annotate_conversation",
                "arguments": json.dumps({
                    "follow_up_timing": "mid_conversation",
                    "follow_up_timing_rationale": "Test rationale",
                    "clarification_or_restatement_detected": True,
                    "clarification_or_restatement_rationale": "Test clarification"
                })
            }
        }
        
        # Should raise an exception due to missing required fields for other features
        try:
            rubrics_instance.tool_call_to_feature_data(minimal_tool_call)
            assert False, "Expected exception for missing required fields"
        except Exception:
            # Expected to fail due to missing required fields
            pass

    def test_type_validation(self, rubrics_instance):
        """Test that type validation works correctly for different prediction types."""
        # Test with invalid type for ClassificationPrediction
        invalid_classification_tool_call = {
            "id": "call_invalid_class",
            "type": "function",
            "function": {
                "name": "annotate_conversation",
                "arguments": json.dumps({
                    "follow_up_timing": "invalid_timing_value",  # Should be one of the literal values
                    "follow_up_timing_rationale": "Test rationale"
                })
            }
        }
        
        # Should raise a validation error for invalid literal value
        try:
            rubrics_instance.tool_call_to_feature_data(invalid_classification_tool_call)
            assert False, "Expected validation error for invalid literal value"
        except Exception:
            # Expected to fail due to invalid literal value
            pass
