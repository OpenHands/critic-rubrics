#!/usr/bin/env python3
"""Test the mixin implementation to ensure everything works correctly."""

from typing import Any

from litellm import ChatCompletionRequest

from critic_rubrics.prediction import BinaryPrediction, TextPrediction
from critic_rubrics.rubrics.base import BaseRubrics, Feature


class RubricForTest(BaseRubrics):
    """Test rubric implementation."""
    
    tool_name: str = "test_rubric"
    tool_description: str = "Test rubric for validation"
    system_message: str = "You are a helpful assistant."
    
    features: list[Feature] = [
        Feature(
            name="is_correct",
            prediction_type=BinaryPrediction,
            description="Whether the answer is correct"
        ),
        Feature(
            name="explanation",
            prediction_type=TextPrediction,
            description="Explanation of the answer"
        )
    ]
    
    def create_annotation_request(
        self, inputs: dict[str, Any], model: str = "openai/gpt-4o-mini"
    ) -> ChatCompletionRequest | None:
        """Create a test annotation request."""
        return ChatCompletionRequest(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Evaluate: {inputs.get('text', '')}"}
            ],
            tools=self.tools
        )


def test_basic_functionality():
    """Test basic functionality of the mixin pattern."""
    print("Testing basic functionality...")
    
    # Create an instance
    rubric = RubricForTest()
    
    # Check that mixin methods are available
    assert hasattr(rubric, 'annotate'), "annotate method not found"
    assert hasattr(rubric, 'batch_annotate'), "batch_annotate method not found"
    assert hasattr(rubric, 'get_batch_results'), "get_batch_results method not found"
    
    # Check that tools property works
    tools = rubric.tools
    import rich
    rich.print(tools)
    assert tools is not None, "tools property returned None"
    assert isinstance(tools, list), "tools should be a list"
    assert len(tools) == 1, "tools should have one element"
    assert 'type' in tools[0], "tools[0] missing 'type' field"
    assert tools[0]['type'] == 'function', "tools[0] type is not 'function'"
    assert 'function' in tools[0], "tools[0] missing 'function' field"
    assert 'parameters' in tools[0]['function'], "tools[0] function missing 'parameters'"
    
    print("✓ Basic functionality tests passed")
    
    # Test create_annotation_request
    request = rubric.create_annotation_request({"text": "Test input"})
    assert request is not None, "create_annotation_request returned None"
    assert isinstance(request, dict), "Request should be a dict"
    assert request.get('model') == "openai/gpt-4o-mini", "Model not set correctly"
    assert len(request.get('messages', [])) == 2, "Messages not set correctly"
    assert request.get('tools') is not None, "Tools not set in request"
    
    print("✓ create_annotation_request tests passed")
    

if __name__ == "__main__":
    test_basic_functionality()
