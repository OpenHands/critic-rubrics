import logging
from abc import ABC
from typing import Any, ClassVar

from litellm import ChatCompletionRequest, ChatCompletionToolChoiceObjectParam, ChatCompletionToolParam
from pydantic import BaseModel

from critic_rubrics.prediction import BasePrediction


logger = logging.getLogger(__name__)


class BaseRubrics(BaseModel, ABC):
    TOOL_NAME: ClassVar[str]
    TOOL_DESCRIPTION: ClassVar[str]
    SYSTEM_MESSAGE: ClassVar[str]
    USER_MESSAGE: ClassVar[str | None] = None  # Optional
    REQUIRED_ALL: ClassVar[bool] = True
    RATIONALE_DESCRIPTION: ClassVar[str] = "Brief evidence/quote (≤25 words) explaining why."

    # ============================================================
    # LLM tool schema generation
    # ============================================================

    @classmethod
    def tool_choice(cls) -> ChatCompletionToolChoiceObjectParam:
        return ChatCompletionToolChoiceObjectParam(
            type="function",
            function={"name": cls.TOOL_NAME},
        )

    @classmethod
    def tools(cls) -> list[ChatCompletionToolParam]:
        props: dict[str, Any] = {}

        for name, field in cls.model_fields.items():  # pydantic v2
            ann = field.annotation
            if not isinstance(ann, type) or not issubclass(ann, BasePrediction):
                logger.warning("Skipping non-Prediction field: %s", name)
                continue

            # use the rubric field's description for the *_detected or *_label text
            assert field.description is not None, f"Field {name} must have a description"
            field_desc = field.description.strip()

            try:
                props.update(
                    ann.to_tool_properties(
                        field_name=name,
                        field_description=field_desc,
                        rationale_description=cls.RATIONALE_DESCRIPTION,
                    )
                )
            except Exception as e:
                logger.exception("Failed building tool properties for %s: %s", name, e)

        required = sorted(props.keys()) if cls.REQUIRED_ALL else []

        return [
            {
                "type": "function",
                "function": {
                    "name": cls.TOOL_NAME,
                    "description": cls.TOOL_DESCRIPTION,
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }
        ]

    # ============================================================
    # Annotation message generation for LLM
    # ============================================================

    @classmethod
    def create_annotation_request(
        cls,
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
