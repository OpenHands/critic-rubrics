"""
Tests for core classes and functionality.
"""

import pytest
from critic_rubrics.core import Prediction, RubricItem, RubricCategory, RubricSet


def test_prediction_creation():
    """Test Prediction dataclass creation."""
    pred = Prediction(detected=True, rationale="Clear evidence found")
    
    assert pred.detected is True
    assert pred.rationale == "Clear evidence found"


def test_prediction_serialization():
    """Test Prediction serialization."""
    pred = Prediction(detected=False, rationale="No evidence")
    
    data = pred.model_dump()
    assert data == {"detected": False, "rationale": "No evidence"}
    
    # Test deserialization
    new_pred = Prediction(**data)
    assert new_pred.detected == pred.detected
    assert new_pred.rationale == pred.rationale


def test_rubric_item_creation():
    """Test RubricItem creation."""
    item = RubricItem(
        name="test_item",
        description="Test description",
        weight=1.0
    )
    
    assert item.name == "test_item"
    assert item.description == "Test description"
    assert item.weight == 1.0


def test_rubric_category_creation():
    """Test RubricCategory creation."""
    items = [
        RubricItem(name="item1", description="First item"),
        RubricItem(name="item2", description="Second item"),
    ]
    
    category = RubricCategory(
        name="test_category",
        description="Test category",
        items=items
    )
    
    assert category.name == "test_category"
    assert len(category.items) == 2
    assert category.items[0].name == "item1"


def test_rubric_set_creation():
    """Test RubricSet creation."""
    items = [RubricItem(name="item1", description="First item")]
    category = RubricCategory(name="cat1", description="Category", items=items)
    
    rubric_set = RubricSet(
        name="test_set",
        description="Test set",
        categories=[category]
    )
    
    assert rubric_set.name == "test_set"
    assert len(rubric_set.categories) == 1
    assert rubric_set.categories[0].name == "cat1"


def test_rubric_set_tool_schema():
    """Test RubricSet tool schema generation."""
    items = [
        RubricItem(name="has_clear_title", description="Issue has clear title"),
        RubricItem(name="has_description", description="Issue has description"),
    ]
    category = RubricCategory(name="basic", description="Basic info", items=items)
    rubric_set = RubricSet(name="test", description="Test", categories=[category])
    
    schema = rubric_set.to_tool_schema()
    
    assert schema["type"] == "function"
    assert "function" in schema
    assert "name" in schema["function"]
    assert "parameters" in schema["function"]
    
    params = schema["function"]["parameters"]
    assert "properties" in params
    assert "has_clear_title" in params["properties"]
    assert "has_description" in params["properties"]