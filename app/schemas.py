from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Any


class CustomerData(BaseModel):
    data: dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float = Field(ge=0, le=1)
    risk_level: str


class BatchPredictionRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(min_length=1, max_length=100)

    @field_validator("customers")
    @classmethod
    def validate_customers(cls, customers):
        if not customers:
            raise ValueError("At least one customer is required.")
        return customers


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

