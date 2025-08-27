from typing import Any

from pydantic import BaseModel

from critic_rubrics.prediction import BasePrediction


class Feature(BaseModel):
    name: str
    description: str
    prediction_type: type[BasePrediction]


class FeatureData(BaseModel):
    """Represents a feature with its actual prediction data."""
    feature: Feature
    prediction: BasePrediction
    
    @classmethod
    def from_tool_args(cls, feature: Feature, tool_args: dict[str, Any]) -> "FeatureData":
        """Create FeatureData from tool call arguments.
        
        Args:
            feature: The feature definition
            tool_args: Flattened tool call arguments
            
        Returns:
            FeatureData with the parsed prediction
            
        Raises:
            ValueError: If tool_args don't match the expected prediction type structure
        """
        prediction_type = feature.prediction_type
        try:
            prediction = prediction_type.from_tool_args(feature.name, tool_args)
        except Exception as e:
            raise ValueError(
                f"Failed to create {prediction_type.__name__} for feature '{feature.name}': {e}"
            ) from e
            
        return cls(feature=feature, prediction=prediction)
    