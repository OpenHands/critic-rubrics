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
        
        # Extract the relevant arguments for this feature from the flattened tool_args
        feature_args = cls._extract_feature_args(feature.name, tool_args, prediction_type)
        
        # Create the prediction instance with type validation
        try:
            prediction = prediction_type(**feature_args)
        except Exception as e:
            raise ValueError(
                f"Failed to create {prediction_type.__name__} for feature '{feature.name}': {e}"
            ) from e
            
        return cls(feature=feature, prediction=prediction)
    
    @staticmethod
    def _extract_feature_args(
        feature_name: str, 
        tool_args: dict[str, Any], 
        prediction_type: type[BasePrediction]
    ) -> dict[str, Any]:
        """Extract arguments for a specific feature from flattened tool arguments."""
        from critic_rubrics.prediction import BinaryPrediction, ClassificationPrediction, TextPrediction
        
        if prediction_type == BinaryPrediction:
            return {
                "detected": tool_args.get(f"{feature_name}_detected"),
                "rationale": tool_args.get(f"{feature_name}_rationale", "")
            }
        elif prediction_type == TextPrediction:
            return {
                "text": tool_args.get(f"{feature_name}_text", "")
            }
        elif issubclass(prediction_type, ClassificationPrediction):
            return {
                "label": tool_args.get(feature_name),
                "rationale": tool_args.get(f"{feature_name}_rationale", "")
            }
        else:
            # For custom prediction types, try to infer from the prediction type's fields
            feature_args = {}
            for field_name in prediction_type.model_fields:
                # Try different naming patterns
                possible_keys = [
                    f"{feature_name}_{field_name}",
                    f"{feature_name}" if field_name == "label" else f"{feature_name}_{field_name}",
                    field_name
                ]
                
                for key in possible_keys:
                    if key in tool_args:
                        feature_args[field_name] = tool_args[key]
                        break
                        
            return feature_args