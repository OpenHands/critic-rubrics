import json
import logging
from abc import abstractmethod
from typing import Any

from litellm import ChatCompletionRequest, ChatCompletionToolChoiceObjectParam, ChatCompletionToolParam
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from critic_rubrics.feature import Feature, FeatureData
from critic_rubrics.prediction import BasePrediction


logger = logging.getLogger(__name__)

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

    def response_to_annotation(self, response: ModelResponse) -> list[Feature]:
        """Convert the LLM response (tool calls) into a structured annotation.

        response: ModelResponse
            The raw response from the LLM.

        Returns:
            list[Feature]: The structured annotation.
        """
        raise NotImplementedError("Subclasses must implement response_to_annotation")

    def tool_calls_to_feature_data(self, tool_calls: list[dict[str, Any]]) -> list[FeatureData]:
        """Convert tool calls into a list of FeatureData with type checking.
        
        Args:
            tool_calls: List of tool call dictionaries from LLM response
            
        Returns:
            list[FeatureData]: Parsed and validated feature data
            
        Raises:
            ValueError: If tool calls don't match expected structure or types
        """
        feature_data_list = []
        
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                continue
                
            function_data = tool_call.get("function", {})
            function_name = function_data.get("name")
            
            # Check if this tool call matches our tool name
            if function_name != self.tool_name:
                logger.warning(f"Skipping tool call with unexpected name: {function_name}")
                continue
                
            # Parse the arguments
            arguments_str = function_data.get("arguments", "{}")
            try:
                if isinstance(arguments_str, str):
                    tool_args = json.loads(arguments_str)
                else:
                    tool_args = arguments_str
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call arguments: {e}")
                continue
                
            # Convert each feature
            for feature in self.features:
                try:
                    feature_data = FeatureData.from_tool_args(feature, tool_args)
                    feature_data_list.append(feature_data)
                except ValueError as e:
                    logger.warning(f"Failed to parse feature '{feature.name}': {e}")
                    # Continue with other features rather than failing completely
                    
        return feature_data_list
