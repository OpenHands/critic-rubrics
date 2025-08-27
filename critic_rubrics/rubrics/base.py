import json
import logging
from abc import abstractmethod
from typing import Any

from litellm import ChatCompletionRequest, ChatCompletionToolChoiceObjectParam, ChatCompletionToolParam
from litellm.types.utils import ChatCompletionMessageToolCall
from pydantic import BaseModel, ValidationError

from critic_rubrics.feature import Feature, FeatureData
from critic_rubrics.prediction import BasePrediction, PredictionMissingFieldError


logger = logging.getLogger(__name__)

def extract_tool_args(tool_call: ChatCompletionMessageToolCall) -> dict[str, Any]:
    assert tool_call.get("type") == "function"
    function_data = tool_call.get("function", {}) or {}

    # Parse the arguments
    arguments_str = function_data.get("arguments", "{}")
    try:
        if isinstance(arguments_str, str):
            return json.loads(arguments_str)
        elif isinstance(arguments_str, dict):
            return arguments_str
        raise ValueError("Unexpected arguments format")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse tool call arguments: {e}")
        return {}


class BaseRubrics(BaseModel):
    tool_name: str
    tool_description: str
    features: list[Feature]
    system_message: str
    user_message: str | None = None  # Optional
    required_all: bool = True
    rationale_description: str = "Brief evidence/quote (<=25 words) explaining why."

    # ============================================================
    # LLM tool schema generation
    # ============================================================

    @property
    def tool_choice(self) -> ChatCompletionToolChoiceObjectParam:
        return ChatCompletionToolChoiceObjectParam(
            type="function",
            function={"name": self.tool_name},
        )

    @property
    def tools(self) -> list[ChatCompletionToolParam]:
        props: dict[str, Any] = {}

        for feature in self.features:
            name = feature.name
            prediction_type = feature.prediction_type
            
            # Validate prediction type
            if not isinstance(prediction_type, type) or not issubclass(prediction_type, BasePrediction):
                logger.warning("Skipping non-Prediction field: %s", name)
                continue

            # Use the feature's description for tool properties
            assert feature.description is not None, f"Field {name} must have a description"
            field_desc = feature.description.strip()

            try:
                props.update(
                    prediction_type.to_tool_properties(
                        field_name=name,
                        field_description=field_desc,
                        rationale_description=self.rationale_description,
                    )
                )
            except Exception as e:
                logger.exception("Failed building tool properties for %s: %s", name, e)

        required = sorted(props.keys()) if self.required_all else []

        return [
            {
                "type": "function",
                "function": {
                    "name": self.tool_name,
                    "description": self.tool_description,
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }
        ]

    # ============================================================
    # Annotation message generation for LLM
    # ============================================================

    @abstractmethod
    def create_annotation_request(
        self,
        inputs: dict[str, Any],
        model: str = "openai/o3-2025-04-16",
    ) -> ChatCompletionRequest | None:
        """Convert the raw inputs dict into an OpenAI-compatible chat completion message.

        inputs: dict[str, Any]
            The raw inputs dict containing all necessary context for the analysis.
            Typically, this is the OpenHands LLM chat completion message list.

        Returns:
            ChatCompletionRequest | None: The formatted message for annotation, or None if formatting fails.
        """
        raise NotImplementedError("Subclasses must implement create_annotation_request")

    def tool_call_to_feature_data(self, tool_call: ChatCompletionMessageToolCall) -> list[FeatureData]:
        """Convert ModelResponse into a list of FeatureData with type checking.
        
        Args:
            response: ModelResponse object from LLM containing tool calls
            
        Returns:
            list[FeatureData]: Parsed and validated feature data
            
        Raises:
            ValueError: If response doesn't contain expected tool calls structure or types
        """
        feature_data_list = []
        
        assert tool_call.get("type") == "function"
        function_data = tool_call.get("function", {}) or {}
        function_name = function_data.get("name")
        if function_name != self.tool_name:
            raise ValueError(f"Tool call with unexpected name: {function_name}")
        tool_args = extract_tool_args(tool_call)

        # Convert each feature
        for feature in self.features:
            try:
                feature_data = FeatureData.from_tool_args(feature, tool_args)
            except ValidationError as e:
                relevant_args = {k: v for k, v in tool_args.items() if feature.name in k}
                logger.warning(f"Validation error for feature {feature.name}: {e}.\nRelevant args: {relevant_args}\n\n")
                continue
            except PredictionMissingFieldError as e:
                logger.warning(f"Missing field for feature {feature.name}: {e}")
                continue
            feature_data_list.append(feature_data)

        return feature_data_list

    def tool_call_match_rubrics(self, tool_call: ChatCompletionMessageToolCall) -> bool:
        """Check if the tool call matches the expected rubric structure."""
        assert tool_call.get("type") == "function"
        function_data = tool_call.get("function", {}) or {}
        function_name = function_data.get("name")
        if function_name != self.tool_name:
            return False
        tool_args = extract_tool_args(tool_call)
        tool_args_set = set(tool_args.keys())
        current_arg_sets = set(self.tools[0]['function']['parameters']["properties"].keys()) # type: ignore

        return tool_args_set == current_arg_sets
