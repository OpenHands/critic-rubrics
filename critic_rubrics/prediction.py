from typing import Any, Generic, Literal, TypeVar, get_args, get_origin

from pydantic import BaseModel, Field


class BasePrediction(BaseModel):
    
    @classmethod
    def to_tool_properties(
        cls,
        field_name: str,
        field_description: str,
        rationale_description: str,
    ) -> dict[str, Any]:
        """Return flattened tool 'properties' entries for this field."""
        raise NotImplementedError
    

    @classmethod
    def from_tool_args(
        cls,
        feature_name: str,
        tool_args: dict[str, Any]
    ) -> "BasePrediction":
        """Create a prediction instance from tool arguments."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to serializable dict with type information."""
        data = self.model_dump()
        data["type"] = self._get_type_name()
        return data
    
    def _get_type_name(self) -> str:
        """Get the type name for this prediction class."""
        # Convert "BinaryPrediction" -> "binary", "TextPrediction" -> "text", etc.
        class_name = self.__class__.__name__
        class_name = class_name.removesuffix("Prediction").lower()
        return class_name



class BinaryPrediction(BasePrediction):
    """Boolean detection + rationale (flattened as <name>_detected / <name>_rationale)."""

    detected: bool = Field(description="Set TRUE only with specific evidence.")
    rationale: str = Field(description="Brief evidence/quote (<=25 words) explaining why.")

    @classmethod
    def to_tool_properties(
        cls,
        field_name: str,
        field_description: str,
        rationale_description: str,
    ) -> dict[str, Any]:
        return {
            f"{field_name}_detected": {"type": "boolean", "description": field_description},
            f"{field_name}_rationale": {"type": "string", "description": rationale_description},
        }
    
    @classmethod
    def from_tool_args(
        cls,
        feature_name: str,
        tool_args: dict[str, Any]
    ) -> "BinaryPrediction":
        detected = tool_args.get(f"{feature_name}_detected")
        if detected is None:
            raise ValueError(f"Missing required field '{feature_name}_detected'")
        rationale = tool_args.get(f"{feature_name}_rationale", "")
        return cls(
            detected=detected,
            rationale=rationale,
        )

class TextPrediction(BasePrediction):
    """Free text output (flattened as <name>_text)."""

    text: str

    @classmethod
    def to_tool_properties(
        cls,
        field_name: str,
        field_description: str,
        rationale_description: str,  # unused
    ) -> dict[str, Any]:
        return {
            f"{field_name}_text": {"type": "string", "description": field_description},
        }
    

    @classmethod
    def from_tool_args(
        cls,
        feature_name: str,
        tool_args: dict[str, Any]
    ) -> "TextPrediction":
        text = tool_args.get(f"{feature_name}_text")
        if text is None:
            raise ValueError(f"Missing required field '{feature_name}_text'")
        return cls(
            text=text,
        )


L = TypeVar("L", bound=str)


class ClassificationPrediction(BasePrediction, Generic[L]):
    """Single-label classification + rationale (flattened as <name>_label / <name>_rationale)."""

    label: L = Field(description="Choose one label from the allowed set.")
    rationale: str = Field(description="Brief evidence/quote (<=25 words) explaining why.")

    @classmethod
    def to_tool_properties(
        cls,
        field_name: str,
        field_description: str,
        rationale_description: str,
    ) -> dict[str, Any]:
        # Extract Literal[...] values from the specialized model's 'label' annotation
        labels: list[str] = []
        label_field = cls.model_fields.get("label")
        if label_field is not None:
            ann = label_field.annotation
            if get_origin(ann) is Literal:
                labels = [str(v) for v in get_args(ann)]

        label_schema: dict[str, Any] = {"type": "string", "description": field_description}
        if labels:
            label_schema["enum"] = labels

        return {
            f"{field_name}": label_schema,
            f"{field_name}_rationale": {"type": "string", "description": rationale_description},
        }

    @classmethod
    def from_tool_args(
        cls,
        feature_name: str,
        tool_args: dict[str, Any]
    ) -> "ClassificationPrediction":
        label = tool_args.get(feature_name)
        if label is None:
            raise ValueError(f"Missing required field '{feature_name}'")
        rationale = tool_args.get(f"{feature_name}_rationale", "")
        return cls(
            label=label,
            rationale=rationale,
        )
