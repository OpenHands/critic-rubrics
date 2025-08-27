from typing import Any

from litellm import ChatCompletionRequest

from ..base import BaseRubrics
from .converter import transform_for_annotator


class AnnotateConversationRubric(BaseRubrics):
    def create_annotation_request(
        self,
        inputs: dict[str, Any],
        model: str = "openai/o3-2025-04-16",
    ) -> ChatCompletionRequest | None:
        assert self.user_message is not None, "user_message must be defined for this rubrics"
        messages = transform_for_annotator(
            inputs,
            system_message=self.system_message,
            annotation_instruction_message=self.user_message,
        )
        if messages is None:
            return None
        return ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=0.0,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )
