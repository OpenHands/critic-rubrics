"""
Tests for pre-defined rubrics and rubric builders.
"""

import pytest

from unified_rubrics.rubrics import (
    AGENT_BEHAVIORAL_RUBRICS,
    USER_FOLLOWUP_RUBRICS,
    INFRASTRUCTURE_RUBRICS,
    SOLVABILITY_RUBRICS,
    CONVERSATION_RUBRICS,
    create_solvability_rubric,
    create_conversation_rubric,
    create_custom_rubric,
    DEFAULT_SOLVABILITY_FEATURES,
)
from unified_rubrics.core import RubricItem, RubricCategory


class TestPredefinedRubrics:
    def test_agent_behavioral_rubrics(self):
        assert AGENT_BEHAVIORAL_RUBRICS.name == "agent_behavioral"
        assert len(AGENT_BEHAVIORAL_RUBRICS.items) > 0
        
        # Check for key items
        identifiers = AGENT_BEHAVIORAL_RUBRICS.get_item_identifiers()
        assert "misunderstood_intention" in identifiers
        assert "did_not_follow_instruction" in identifiers
        assert "insufficient_analysis" in identifiers
        assert "loop_behavior" in identifiers
    
    def test_user_followup_rubrics(self):
        assert USER_FOLLOWUP_RUBRICS.name == "user_followup"
        assert len(USER_FOLLOWUP_RUBRICS.items) > 0
        assert USER_FOLLOWUP_RUBRICS.mutually_exclusive is True
        assert USER_FOLLOWUP_RUBRICS.max_selections == 1
        
        # Check for key items
        identifiers = USER_FOLLOWUP_RUBRICS.get_item_identifiers()
        assert "clarification_or_restatement" in identifiers
        assert "correction" in identifiers
        assert "direction_change" in identifiers
        assert "vcs_update_requests" in identifiers
    
    def test_infrastructure_rubrics(self):
        assert INFRASTRUCTURE_RUBRICS.name == "infrastructure"
        assert len(INFRASTRUCTURE_RUBRICS.items) > 0
        
        # Check for key items
        identifiers = INFRASTRUCTURE_RUBRICS.get_item_identifiers()
        assert "infrastructure_external_issue" in identifiers
        assert "infrastructure_agent_caused_issue" in identifiers
    
    def test_solvability_rubrics(self):
        assert SOLVABILITY_RUBRICS.name == "solvability"
        assert len(SOLVABILITY_RUBRICS.standalone_items) > 0
        
        # Check for key features
        identifiers = SOLVABILITY_RUBRICS.get_all_identifiers()
        assert "has_clear_requirements" in identifiers
        assert "has_reproduction_steps" in identifiers
        assert "is_bug_report" in identifiers
        assert "is_feature_request" in identifiers
    
    def test_conversation_rubrics(self):
        assert CONVERSATION_RUBRICS.name == "conversation_analysis"
        assert len(CONVERSATION_RUBRICS.categories) > 0
        
        # Should include all major categories
        category_names = [cat.name for cat in CONVERSATION_RUBRICS.categories]
        assert "agent_behavioral" in category_names
        assert "user_followup" in category_names
        assert "infrastructure" in category_names
        
        # Should have additional fields
        assert "follow_up_timing" in CONVERSATION_RUBRICS.additional_fields
        assert "task_type" in CONVERSATION_RUBRICS.additional_fields


class TestRubricBuilders:
    def test_create_solvability_rubric(self):
        custom_features = [
            {"identifier": "has_logs", "description": "Issue includes log files"},
            {"identifier": "has_screenshots", "description": "Issue includes screenshots"},
        ]
        
        rubric = create_solvability_rubric(custom_features)
        
        assert rubric.name == "solvability"
        assert len(rubric.standalone_items) == 2
        
        identifiers = rubric.get_all_identifiers()
        assert "has_logs" in identifiers
        assert "has_screenshots" in identifiers
        
        # Solvability items should not require rationales by default
        for item in rubric.standalone_items:
            assert item.requires_rationale is False
    
    def test_create_conversation_rubric_minimal(self):
        rubric = create_conversation_rubric(
            include_timing=False,
            include_task_type=False
        )
        
        assert rubric.name == "conversation_analysis"
        assert len(rubric.categories) > 0
        assert len(rubric.additional_fields) == 0
    
    def test_create_conversation_rubric_full(self):
        rubric = create_conversation_rubric(
            include_timing=True,
            include_task_type=True
        )
        
        assert rubric.name == "conversation_analysis"
        assert len(rubric.categories) > 0
        assert "follow_up_timing" in rubric.additional_fields
        assert "task_type" in rubric.additional_fields
        
        # Check timing field configuration
        timing_field = rubric.additional_fields["follow_up_timing"]
        assert timing_field["type"] == "string"
        assert "mid_conversation" in timing_field["enum"]
        assert "post_completion" in timing_field["enum"]
        assert "no_follow_up" in timing_field["enum"]
        
        # Check task type field configuration
        task_field = rubric.additional_fields["task_type"]
        assert task_field["type"] == "string"
        assert "coding" in task_field["enum"]
        assert "debugging" in task_field["enum"]
    
    def test_create_custom_rubric_empty(self):
        rubric = create_custom_rubric(
            name="empty_rubric",
            description="An empty rubric for testing"
        )
        
        assert rubric.name == "empty_rubric"
        assert rubric.description == "An empty rubric for testing"
        assert len(rubric.categories) == 0
        assert len(rubric.standalone_items) == 0
        assert len(rubric.additional_fields) == 0
    
    def test_create_custom_rubric_with_items(self):
        items = [
            RubricItem(identifier="item1", description="First item"),
            RubricItem(identifier="item2", description="Second item"),
        ]
        
        rubric = create_custom_rubric(
            name="custom_rubric",
            description="A custom rubric",
            items=items
        )
        
        assert rubric.name == "custom_rubric"
        assert len(rubric.standalone_items) == 2
        assert rubric.get_all_identifiers() == ["item1", "item2"]
    
    def test_create_custom_rubric_with_categories(self):
        category = RubricCategory(name="test_cat", description="Test category")
        category.add_item(RubricItem(identifier="cat_item", description="Category item"))
        
        rubric = create_custom_rubric(
            name="custom_rubric",
            description="A custom rubric",
            categories=[category]
        )
        
        assert rubric.name == "custom_rubric"
        assert len(rubric.categories) == 1
        assert rubric.categories[0].name == "test_cat"
        assert "cat_item" in rubric.get_all_identifiers()
    
    def test_create_custom_rubric_with_additional_fields(self):
        additional_fields = {
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "required": True
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "required": False
            }
        }
        
        rubric = create_custom_rubric(
            name="custom_rubric",
            description="A custom rubric",
            additional_fields=additional_fields
        )
        
        assert rubric.name == "custom_rubric"
        assert len(rubric.additional_fields) == 2
        assert "priority" in rubric.additional_fields
        assert "confidence" in rubric.additional_fields
        
        # Test tool schema generation
        schema = rubric.to_tool_schema()
        properties = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]
        
        assert "priority" in properties
        assert "confidence" in properties
        assert "priority" in required
        assert "confidence" not in required  # Not required


class TestDefaultFeatures:
    def test_default_solvability_features(self):
        assert isinstance(DEFAULT_SOLVABILITY_FEATURES, list)
        assert len(DEFAULT_SOLVABILITY_FEATURES) > 0
        
        # Check structure
        for feature in DEFAULT_SOLVABILITY_FEATURES:
            assert isinstance(feature, dict)
            assert "identifier" in feature
            assert "description" in feature
            assert isinstance(feature["identifier"], str)
            assert isinstance(feature["description"], str)
        
        # Check for key features
        identifiers = [f["identifier"] for f in DEFAULT_SOLVABILITY_FEATURES]
        assert "has_clear_requirements" in identifiers
        assert "has_reproduction_steps" in identifiers
        assert "is_bug_report" in identifiers
        assert "is_feature_request" in identifiers


class TestRubricIntegration:
    def test_solvability_rubric_tool_schema(self):
        """Test that solvability rubric generates valid tool schema."""
        schema = SOLVABILITY_RUBRICS.to_tool_schema()
        
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]
        
        parameters = schema["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "properties" in parameters
        assert "required" in parameters
        
        # Check that all items have detection fields
        for identifier in SOLVABILITY_RUBRICS.get_all_identifiers():
            detection_field = f"{identifier}_detected"
            assert detection_field in parameters["properties"]
            assert detection_field in parameters["required"]
            
            # Solvability items don't require rationales
            rationale_field = f"{identifier}_rationale"
            assert rationale_field not in parameters["properties"]
    
    def test_conversation_rubric_tool_schema(self):
        """Test that conversation rubric generates valid tool schema."""
        schema = CONVERSATION_RUBRICS.to_tool_schema()
        
        assert schema["type"] == "function"
        parameters = schema["function"]["parameters"]
        
        # Check additional fields
        assert "follow_up_timing" in parameters["properties"]
        assert "task_type" in parameters["properties"]
        assert "follow_up_timing" in parameters["required"]
        assert "task_type" in parameters["required"]
        
        # Check that all rubric items have detection and rationale fields
        for identifier in CONVERSATION_RUBRICS.get_all_identifiers():
            detection_field = f"{identifier}_detected"
            rationale_field = f"{identifier}_rationale"
            
            assert detection_field in parameters["properties"]
            assert rationale_field in parameters["properties"]
            assert detection_field in parameters["required"]
            assert rationale_field in parameters["required"]