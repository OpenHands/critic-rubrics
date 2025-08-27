import logging
from abc import abstractmethod
from typing import Any

from litellm import ChatCompletionRequest, ChatCompletionToolChoiceObjectParam, ChatCompletionToolParam
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from critic_rubrics.feature import Feature
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
