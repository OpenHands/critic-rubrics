"""
Tests for core rubrics functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
import json

from unified_rubrics.core import (
    RubricItem,
    RubricCategory,
    RubricSet,
    AnnotationResult,
    RubricAnnotator,
    MultiSampleResult,
)


class TestRubricItem:
    def test_basic_creation(self):
        item = RubricItem(
            identifier="test_item",
            description="A test item"
        )
        assert item.identifier == "test_item"
        assert item.description == "A test item"
        assert item.requires_rationale is True
        assert item.category is None
    
    def test_field_names(self):
        item = RubricItem(identifier="test_item", description="Test")
        assert item.detection_field_name == "test_item_detected"
        assert item.rationale_field_name == "test_item_rationale"
    
    def test_tool_schema_properties(self):
        item = RubricItem(
            identifier="test_item",
            description="A test item",
            requires_rationale=True
        )
        
        properties = item.to_tool_schema_properties()
        
        assert "test_item_detected" in properties
        assert "test_item_rationale" in properties
        assert properties["test_item_detected"]["type"] == "boolean"
        assert properties["test_item_detected"]["description"] == "A test item"
        assert properties["test_item_rationale"]["type"] == "string"
    
    def test_tool_schema_no_rationale(self):
        item = RubricItem(
            identifier="test_item",
            description="A test item",
            requires_rationale=False
        )
        
        properties = item.to_tool_schema_properties()
        
        assert "test_item_detected" in properties
        assert "test_item_rationale" not in properties


class TestRubricCategory:
    def test_basic_creation(self):
        category = RubricCategory(
            name="test_category",
            description="A test category"
        )
        assert category.name == "test_category"
        assert category.description == "A test category"
        assert len(category.items) == 0
        assert category.mutually_exclusive is False
    
    def test_add_item(self):
        category = RubricCategory(name="test", description="Test")
        item = RubricItem(identifier="item1", description="Item 1")
        
        category.add_item(item)
        
        assert len(category.items) == 1
        assert category.items[0] == item
        assert item.category == "test"
    
    def test_get_item_identifiers(self):
        category = RubricCategory(name="test", description="Test")
        item1 = RubricItem(identifier="item1", description="Item 1")
        item2 = RubricItem(identifier="item2", description="Item 2")
        
        category.add_item(item1)
        category.add_item(item2)
        
        identifiers = category.get_item_identifiers()
        assert identifiers == ["item1", "item2"]


class TestRubricSet:
    def test_basic_creation(self):
        rubric_set = RubricSet(
            name="test_rubric",
            description="A test rubric set"
        )
        assert rubric_set.name == "test_rubric"
        assert rubric_set.description == "A test rubric set"
        assert len(rubric_set.categories) == 0
        assert len(rubric_set.standalone_items) == 0
    
    def test_add_category(self):
        rubric_set = RubricSet(name="test", description="Test")
        category = RubricCategory(name="cat1", description="Category 1")
        
        rubric_set.add_category(category)
        
        assert len(rubric_set.categories) == 1
        assert rubric_set.categories[0] == category
    
    def test_add_standalone_item(self):
        rubric_set = RubricSet(name="test", description="Test")
        item = RubricItem(identifier="item1", description="Item 1")
        
        rubric_set.add_item(item)
        
        assert len(rubric_set.standalone_items) == 1
        assert rubric_set.standalone_items[0] == item
    
    def test_add_item_to_category(self):
        rubric_set = RubricSet(name="test", description="Test")
        category = RubricCategory(name="cat1", description="Category 1")
        item = RubricItem(identifier="item1", description="Item 1")
        
        rubric_set.add_category(category)
        rubric_set.add_item(item, category_name="cat1")
        
        assert len(category.items) == 1
        assert category.items[0] == item
        assert len(rubric_set.standalone_items) == 0
    
    def test_get_all_items(self):
        rubric_set = RubricSet(name="test", description="Test")
        
        # Add category with items
        category = RubricCategory(name="cat1", description="Category 1")
        cat_item = RubricItem(identifier="cat_item", description="Category Item")
        category.add_item(cat_item)
        rubric_set.add_category(category)
        
        # Add standalone item
        standalone_item = RubricItem(identifier="standalone", description="Standalone")
        rubric_set.add_item(standalone_item)
        
        all_items = rubric_set.get_all_items()
        assert len(all_items) == 2
        assert standalone_item in all_items
        assert cat_item in all_items
    
    def test_to_tool_schema(self):
        rubric_set = RubricSet(name="test", description="Test")
        item = RubricItem(identifier="test_item", description="Test item")
        rubric_set.add_item(item)
        
        schema = rubric_set.to_tool_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "annotate"
        assert "test_item_detected" in schema["function"]["parameters"]["properties"]
        assert "test_item_rationale" in schema["function"]["parameters"]["properties"]
        assert "test_item_detected" in schema["function"]["parameters"]["required"]
        assert "test_item_rationale" in schema["function"]["parameters"]["required"]


class TestAnnotationResult:
    def test_basic_creation(self):
        result = AnnotationResult(
            rubric_set_name="test_rubric",
            detections={"item1": True, "item2": False},
            rationales={"item1": "Evidence found"},
            additional_data={"task_type": "coding"}
        )
        
        assert result.rubric_set_name == "test_rubric"
        assert result.detections["item1"] is True
        assert result.detections["item2"] is False
        assert result.rationales["item1"] == "Evidence found"
        assert result.additional_data["task_type"] == "coding"
    
    def test_get_detected_items(self):
        result = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": True, "item2": False, "item3": True}
        )
        
        detected = result.get_detected_items()
        assert detected == ["item1", "item3"]
    
    def test_get_detection_rate(self):
        result = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": True, "item2": False, "item3": True}
        )
        
        rate = result.get_detection_rate()
        assert rate == 2/3  # 2 out of 3 detected
    
    def test_to_dict(self):
        result = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": True},
            rationales={"item1": "Evidence"},
            additional_data={"task_type": "coding"},
            prompt_tokens=100,
            completion_tokens=50
        )
        
        data = result.to_dict()
        
        assert data["rubric_set_name"] == "test"
        assert data["item1"] is True
        assert data["item1_rationale"] == "Evidence"
        assert data["task_type"] == "coding"
        assert data["prompt_tokens"] == 100
        assert data["completion_tokens"] == 50


class MockAnnotator(RubricAnnotator):
    """Mock annotator for testing."""
    
    def __init__(self, rubric_set, mock_response=None):
        super().__init__(
            rubric_set=rubric_set,
            system_prompt="Test system prompt",
            instruction_prompt="Test instruction prompt"
        )
        self.mock_response = mock_response or self._default_mock_response()
        self.call_count = 0
    
    def _default_mock_response(self):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.tool_calls = [Mock()]
        mock_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "test_item_detected": True,
            "test_item_rationale": "Test rationale"
        })
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        return mock_response
    
    def _call_llm(self, messages, tools, tool_choice, temperature=0.0, **kwargs):
        self.call_count += 1
        return self.mock_response


class TestRubricAnnotator:
    def test_annotate(self):
        # Create test rubric
        rubric_set = RubricSet(name="test", description="Test")
        item = RubricItem(identifier="test_item", description="Test item")
        rubric_set.add_item(item)
        
        # Create mock annotator
        annotator = MockAnnotator(rubric_set)
        
        # Test annotation
        result = annotator.annotate("Test content")
        
        assert annotator.call_count == 1
        assert result.rubric_set_name == "test"
        assert result.detections["test_item"] is True
        assert result.rationales["test_item"] == "Test rationale"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.response_latency > 0
    
    def test_annotate_batch(self):
        # Create test rubric
        rubric_set = RubricSet(name="test", description="Test")
        item = RubricItem(identifier="test_item", description="Test item")
        rubric_set.add_item(item)
        
        # Create mock annotator
        annotator = MockAnnotator(rubric_set)
        
        # Test batch annotation
        contents = ["Content 1", "Content 2", "Content 3"]
        results = annotator.annotate_batch(contents, max_workers=1)
        
        assert len(results) == 3
        assert annotator.call_count == 3
        
        for result in results:
            assert result.rubric_set_name == "test"
            assert result.detections["test_item"] is True


class TestMultiSampleResult:
    def test_get_detection_rates(self):
        # Create sample results
        sample1 = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": True, "item2": False}
        )
        sample2 = AnnotationResult(
            rubric_set_name="test", 
            detections={"item1": False, "item2": True}
        )
        sample3 = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": True, "item2": True}
        )
        
        multi_result = MultiSampleResult(
            rubric_set_name="test",
            samples=[sample1, sample2, sample3]
        )
        
        rates = multi_result.get_detection_rates()
        
        assert rates["item1"] == 2/3  # Detected in 2 out of 3 samples
        assert rates["item2"] == 2/3  # Detected in 2 out of 3 samples
    
    def test_get_detection_entropy(self):
        # Create samples with different detection patterns
        sample1 = AnnotationResult(
            rubric_set_name="test",
            detections={"always_true": True, "always_false": False, "mixed": True}
        )
        sample2 = AnnotationResult(
            rubric_set_name="test",
            detections={"always_true": True, "always_false": False, "mixed": False}
        )
        
        multi_result = MultiSampleResult(
            rubric_set_name="test",
            samples=[sample1, sample2]
        )
        
        entropies = multi_result.get_detection_entropy()
        
        # Always true/false should have 0 entropy (no uncertainty)
        assert entropies["always_true"] == 0.0
        assert entropies["always_false"] == 0.0
        
        # Mixed should have maximum entropy for 2 samples (1.0)
        assert entropies["mixed"] == 1.0
    
    def test_get_consensus_result(self):
        # Create samples
        sample1 = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": True, "item2": False},
            rationales={"item1": "Rationale A"},
            additional_data={"task_type": "coding"}
        )
        sample2 = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": True, "item2": True},
            rationales={"item1": "Rationale A", "item2": "Rationale B"},
            additional_data={"task_type": "coding"}
        )
        sample3 = AnnotationResult(
            rubric_set_name="test",
            detections={"item1": False, "item2": True},
            rationales={"item2": "Rationale B"},
            additional_data={"task_type": "debugging"}
        )
        
        multi_result = MultiSampleResult(
            rubric_set_name="test",
            samples=[sample1, sample2, sample3]
        )
        
        # Test with 60% threshold
        consensus = multi_result.get_consensus_result(threshold=0.6)
        
        # item1: 2/3 = 66.7% > 60% -> True
        # item2: 2/3 = 66.7% > 60% -> True
        assert consensus.detections["item1"] is True
        assert consensus.detections["item2"] is True
        
        # Should have rationales for detected items
        assert "item1" in consensus.rationales
        assert "item2" in consensus.rationales
        
        # Should use most common additional data
        assert consensus.additional_data["task_type"] == "coding"  # 2 out of 3