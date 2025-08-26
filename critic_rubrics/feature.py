from pydantic import BaseModel

from critic_rubrics.prediction import BasePrediction


class Feature(BaseModel):
    name: str
    description: str
    prediction_type: type[BasePrediction]

