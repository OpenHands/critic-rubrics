# base_rubrics.py
from typing import Any, ClassVar
import logging
from pydantic import BaseModel

from critic_rubrics.prediction import BasePrediction

logger = logging.getLogger(__name__)

class BaseRubrics(BaseModel):
    TOOL_NAME: ClassVar[str] = "annotate_conversation"
    TOOL_DESCRIPTION: ClassVar[str] = "Annotate agent conversation."
    REQUIRED_ALL: ClassVar[bool] = True
    RATIONALE_DESCRIPTION: ClassVar[str] = "Brief evidence/quote (≤25 words) explaining why."

    @property
    def tool_choice(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.TOOL_NAME}}

    @property
    def tool_description(self) -> dict[str, Any]:
        props: dict[str, Any] = {}

        for name, field in self.model_fields.items():  # pydantic v2
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
                        rationale_description=self.RATIONALE_DESCRIPTION,
                    )
                )
            except Exception as e:
                logger.exception("Failed building tool properties for %s: %s", name, e)

        required = sorted(props.keys()) if self.REQUIRED_ALL else []

        return {
            "type": "function",
            "function": {
                "name": self.TOOL_NAME,
                "description": self.TOOL_DESCRIPTION,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }
