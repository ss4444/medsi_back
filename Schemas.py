from pydantic import BaseModel
from decimal import Decimal
from typing import List


class Diagnosis(BaseModel):
    title: str
    value: Decimal


class PredictModel(BaseModel):
    predict: List[Diagnosis]


class Pathologies(BaseModel):
    pathologies: List[str]
