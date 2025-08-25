from typing import Any, ClassVar
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class Prediction(BaseModel):
    """Represents a prediction with detection result and rationale."""
    detected: bool = Field(
        description="Set TRUE only with specific evidence."
    )
    rationale: str = Field(
        description="Brief evidence/quote (≤25 words) explaining why."
    )


class BaseRubrics(BaseModel):
    TOOL_NAME: ClassVar[str] = "annotate_conversation"
    TOOL_DESCRIPTION: ClassVar[str] = "Annotate agent conversation."
    REQUIRED_ALL: ClassVar[bool] = Field(
        default=True,
        description="If TRUE, all <feature>_detected and <feature>_rationale fields are required.",
    )
    # One canonical description for ALL <feature>_rationale fields
    RATIONALE_DESCRIPTION: ClassVar[str] = (
        "Quote evidence concisely (≤25 words) and explain in a sentence."
    )

    def extra_tool_properties(self) -> dict[str, Any]:
        return {}

    @property
    def tool_choice(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.TOOL_NAME}}

    @property
    def tool_description(self) -> dict[str, Any]:
        props: dict[str, Any] = {}

        for name, field in self.model_fields.items():  # pydantic v2
            ann = field.annotation
            is_prediction = isinstance(ann, type) and issubclass(ann, Prediction)
            if not is_prediction:
                logger.warning(
                    f"Field '{name}' is not of type 'Prediction'. Skipping."
                )
                continue
            props[f"{name}_detected"] = {"type": "boolean", "description": field.description}
            props[f"{name}_rationale"] = {"type": "string", "description": self.RATIONALE_DESCRIPTION}

        props.update(self.extra_tool_properties() or {})
        required = sorted(props.keys()) if self.REQUIRED_ALL else []

        return {
            "type": "function",
            "function": {
                "name": self.TOOL_NAME,
                "description": self.TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }
