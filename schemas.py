from pydantic import BaseModel
from decimal import Decimal
from typing import List


class Diagnosis(BaseModel):
    title: str
    value: Decimal
    pathologies: List[str]


class PredictModel(BaseModel):
    predict: List[Diagnosis]
