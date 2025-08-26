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
    def from_tool_response(
        cls,
        field_name: str,
        args: dict[str, Any],
    ) -> "BasePrediction":
        """Reconstruct a prediction from tool response arguments."""
        raise NotImplementedError

    # NOTE: we should add the validation/error auto conversion in Prediction

class BinaryPrediction(BasePrediction):
    """Boolean detection + rationale (flattened as <name>_detected / <name>_rationale)."""

    detected: bool = Field(description="Set TRUE only with specific evidence.")
    rationale: str = Field(description="Brief evidence/quote (≤25 words) explaining why.")

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
    def from_tool_response(
        cls,
        field_name: str,
        args: dict[str, Any],
    ) -> "BinaryPrediction":
        """Reconstruct a BinaryPrediction from tool response arguments."""
        return cls(
            detected=args.get(f"{field_name}_detected", False),
            rationale=args.get(f"{field_name}_rationale", ""),
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
    def from_tool_response(
        cls,
        field_name: str,
        args: dict[str, Any],
    ) -> "TextPrediction":
        """Reconstruct a TextPrediction from tool response arguments."""
        return cls(
            text=args.get(f"{field_name}_text", ""),
        )


L = TypeVar("L", bound=str)


class ClassificationPrediction(BasePrediction, Generic[L]):
    """Single-label classification + rationale (flattened as <name>_label / <name>_rationale)."""

    label: L = Field(description="Choose one label from the allowed set.")
    rationale: str = Field(description="Brief evidence/quote (≤25 words) explaining why.")

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
    def from_tool_response(
        cls,
        field_name: str,
        args: dict[str, Any],
    ) -> "ClassificationPrediction[L]":
        """Reconstruct a ClassificationPrediction from tool response arguments."""
        return cls(
            label=args.get(f"{field_name}", ""),  # type: ignore
            rationale=args.get(f"{field_name}_rationale", ""),
        )
