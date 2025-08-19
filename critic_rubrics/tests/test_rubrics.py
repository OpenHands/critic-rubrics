"""
Tests for rubric dataclasses.
"""

import pytest
from critic_rubrics.core import Prediction
from critic_rubrics.rubrics import SolvabilityRubrics, TrajectoryRubrics, ConversationRubrics


def test_solvability_rubrics_creation():
    """Test SolvabilityRubrics creation."""
    rubrics = SolvabilityRubrics(
        has_clear_problem_statement=Prediction(detected=True, rationale="Clear problem"),
        has_reproduction_steps=Prediction(detected=False, rationale="No steps"),
        has_expected_behavior=Prediction(detected=True, rationale="Expected described"),
        has_actual_behavior=Prediction(detected=True, rationale="Actual described"),
        has_error_messages=Prediction(detected=False, rationale="No errors"),
        has_environment_info=Prediction(detected=True, rationale="Environment given"),
        has_version_info=Prediction(detected=False, rationale="No versions"),
        has_code_examples=Prediction(detected=False, rationale="No code"),
        has_minimal_example=Prediction(detected=False, rationale="No example"),
        has_scope_definition=Prediction(detected=True, rationale="Scope clear"),
        has_impact_description=Prediction(detected=False, rationale="No impact"),
        shows_investigation_effort=Prediction(detected=False, rationale="No investigation"),
    )
    
    assert rubrics.has_clear_problem_statement.detected is True
    assert rubrics.has_reproduction_steps.detected is False
    assert rubrics.get_detection_rate() == 4/12  # 4 detected out of 12


def test_solvability_rubrics_methods():
    """Test SolvabilityRubrics helper methods."""
    rubrics = SolvabilityRubrics(
        has_clear_problem_statement=Prediction(detected=True, rationale="Clear"),
        has_reproduction_steps=Prediction(detected=True, rationale="Steps given"),
        has_expected_behavior=Prediction(detected=False, rationale="Not described"),
        has_actual_behavior=Prediction(detected=False, rationale="Not described"),
        has_error_messages=Prediction(detected=False, rationale="No errors"),
        has_environment_info=Prediction(detected=False, rationale="No env"),
        has_version_info=Prediction(detected=False, rationale="No versions"),
        has_code_examples=Prediction(detected=False, rationale="No code"),
        has_minimal_example=Prediction(detected=False, rationale="No example"),
        has_scope_definition=Prediction(detected=False, rationale="No scope"),
        has_impact_description=Prediction(detected=False, rationale="No impact"),
        shows_investigation_effort=Prediction(detected=False, rationale="No investigation"),
    )
    
    detected = rubrics.get_detected_features()
    missing = rubrics.get_missing_features()
    
    assert "has_clear_problem_statement" in detected
    assert "has_reproduction_steps" in detected
    assert len(detected) == 2
    
    assert "has_expected_behavior" in missing
    assert len(missing) == 10


def test_trajectory_rubrics_creation():
    """Test TrajectoryRubrics creation with minimal required fields."""
    # Create all required prediction fields
    false_pred = Prediction(detected=False, rationale="Not detected")
    true_pred = Prediction(detected=True, rationale="Detected")
    
    rubrics = TrajectoryRubrics(
        # Agent behavioral issues
        misunderstood_intention=false_pred,
        ignored_user_request=false_pred,
        misinterpreted_context=false_pred,
        incomplete_task_execution=false_pred,
        incorrect_approach=false_pred,
        unnecessary_actions=false_pred,
        failed_to_verify=false_pred,
        unclear_communication=false_pred,
        insufficient_explanation=false_pred,
        overly_verbose=false_pred,
        poor_error_handling=false_pred,
        failed_to_recover=false_pred,
        repeated_same_mistake=false_pred,
        
        # User follow-up patterns
        user_requested_clarification=false_pred,
        user_corrected_agent=false_pred,
        user_provided_additional_context=false_pred,
        user_expressed_frustration=false_pred,
        user_requested_different_approach=false_pred,
        user_abandoned_task=false_pred,
        
        # Infrastructure issues
        tool_usage_error=false_pred,
        environment_setup_issue=false_pred,
        permission_or_access_issue=false_pred,
        network_or_connectivity_issue=false_pred,
        file_handling_error=false_pred,
        data_corruption_or_loss=false_pred,
        version_control_issue=false_pred,
        
        # Quality metrics
        conversation_too_long=false_pred,
        multiple_iterations_needed=false_pred,
        backtracking_required=false_pred,
        task_completed_successfully=true_pred,
        user_satisfied_with_result=true_pred,
        efficient_problem_solving=true_pred,
        
        # Behavioral patterns
        agent_learned_from_feedback=false_pred,
        agent_adapted_approach=false_pred,
        agent_asked_clarifying_questions=false_pred,
        agent_anticipated_needs=false_pred,
        agent_provided_alternatives=false_pred,
        agent_offered_additional_help=false_pred,
        
        # Domain-specific
        code_quality_issues=false_pred,
        security_concerns=false_pred,
        performance_issues=false_pred,
        insufficient_documentation=false_pred,
        missing_tests=false_pred,
        incomplete_examples=false_pred,
    )
    
    assert rubrics.task_completed_successfully.detected is True
    assert rubrics.misunderstood_intention.detected is False
    
    # Test helper methods
    assert rubrics.get_issue_count() == 0  # No issues detected
    assert rubrics.get_positive_indicators_count() == 3  # 3 positive indicators
    assert rubrics.get_quality_score() > 0.5  # Should be good quality


def test_trajectory_rubrics_methods():
    """Test TrajectoryRubrics helper methods."""
    false_pred = Prediction(detected=False, rationale="Not detected")
    true_pred = Prediction(detected=True, rationale="Detected")
    
    # Create rubrics with some issues and some positive indicators
    rubrics = TrajectoryRubrics(
        # Some issues
        misunderstood_intention=true_pred,
        ignored_user_request=false_pred,
        misinterpreted_context=false_pred,
        incomplete_task_execution=true_pred,
        incorrect_approach=false_pred,
        unnecessary_actions=false_pred,
        failed_to_verify=false_pred,
        unclear_communication=false_pred,
        insufficient_explanation=false_pred,
        overly_verbose=false_pred,
        poor_error_handling=false_pred,
        failed_to_recover=false_pred,
        repeated_same_mistake=false_pred,
        
        # User follow-up
        user_requested_clarification=true_pred,
        user_corrected_agent=false_pred,
        user_provided_additional_context=false_pred,
        user_expressed_frustration=false_pred,
        user_requested_different_approach=false_pred,
        user_abandoned_task=false_pred,
        
        # Infrastructure (all false for simplicity)
        tool_usage_error=false_pred,
        environment_setup_issue=false_pred,
        permission_or_access_issue=false_pred,
        network_or_connectivity_issue=false_pred,
        file_handling_error=false_pred,
        data_corruption_or_loss=false_pred,
        version_control_issue=false_pred,
        conversation_too_long=false_pred,
        multiple_iterations_needed=false_pred,
        backtracking_required=false_pred,
        code_quality_issues=false_pred,
        security_concerns=false_pred,
        performance_issues=false_pred,
        insufficient_documentation=false_pred,
        missing_tests=false_pred,
        incomplete_examples=false_pred,
        
        # Some positive indicators
        task_completed_successfully=true_pred,
        user_satisfied_with_result=false_pred,
        efficient_problem_solving=false_pred,
        agent_learned_from_feedback=false_pred,
        agent_adapted_approach=false_pred,
        agent_asked_clarifying_questions=false_pred,
        agent_anticipated_needs=false_pred,
        agent_provided_alternatives=false_pred,
        agent_offered_additional_help=false_pred,
    )
    
    assert rubrics.get_issue_count() == 2  # misunderstood_intention + incomplete_task_execution
    assert rubrics.get_positive_indicators_count() == 1  # task_completed_successfully
    
    followup_indicators = rubrics.get_user_followup_indicators()
    assert "user_requested_clarification" in followup_indicators
    assert len(followup_indicators) == 1


def test_conversation_rubrics_creation():
    """Test ConversationRubrics creation."""
    rubrics = ConversationRubrics(
        natural_conversation_flow=Prediction(detected=True, rationale="Flows well"),
        appropriate_turn_taking=Prediction(detected=True, rationale="Good turns"),
        maintains_context=Prediction(detected=True, rationale="Context maintained"),
        clear_communication=Prediction(detected=True, rationale="Very clear"),
        appropriate_tone=Prediction(detected=True, rationale="Professional tone"),
        active_listening=Prediction(detected=True, rationale="Shows listening"),
        mutual_understanding=Prediction(detected=True, rationale="Both understand"),
        clarification_when_needed=Prediction(detected=False, rationale="No clarification needed"),
        acknowledges_responses=Prediction(detected=True, rationale="Good acknowledgment"),
        engaged_participation=Prediction(detected=True, rationale="Both engaged"),
        builds_rapport=Prediction(detected=True, rationale="Good rapport"),
        respectful_interaction=Prediction(detected=True, rationale="Very respectful"),
        achieves_communication_goals=Prediction(detected=True, rationale="Goals achieved"),
        efficient_information_exchange=Prediction(detected=True, rationale="Efficient"),
        productive_outcome=Prediction(detected=True, rationale="Productive"),
        communication_breakdown=Prediction(detected=False, rationale="No breakdown"),
        repetitive_or_circular=Prediction(detected=False, rationale="Not repetitive"),
        off_topic_drift=Prediction(detected=False, rationale="Stayed on topic"),
    )
    
    assert rubrics.natural_conversation_flow.detected is True
    assert rubrics.communication_breakdown.detected is False
    
    # Test quality score (should be high)
    quality = rubrics.get_quality_score()
    assert quality > 0.8  # Should be high quality
    
    # Test methods
    issues = rubrics.get_issues()
    assert len(issues) == 0  # No issues
    
    strengths = rubrics.get_strengths()
    assert len(strengths) > 10  # Many strengths detected