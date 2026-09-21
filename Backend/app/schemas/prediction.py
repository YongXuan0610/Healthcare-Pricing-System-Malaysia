"""Strict request schemas for separate NSS-primary and US-benchmark models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from model_training.nss80.production_practical15 import canonical_category, validate_consistency


class NSS80CostPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age_years: int = Field(ge=0, le=120)
    gender: str
    chronic_ailment: str
    pregnant: str | None = None
    communicable_disease: str
    other_ailment_last_15_days: str
    number_of_hospitalisations: int = Field(ge=1, le=24)
    length_of_stay_days: int = Field(ge=1, le=365)
    ailment_nature: str
    hospitalisation_treatment_nature: str
    medical_institution_type: str
    ward_type: str
    place_of_hospitalisation: str
    surgery: str
    medicine: str

    @field_validator(
        "gender",
        "chronic_ailment",
        "pregnant",
        "communicable_disease",
        "other_ailment_last_15_days",
        "ailment_nature",
        "hospitalisation_treatment_nature",
        "medical_institution_type",
        "ward_type",
        "place_of_hospitalisation",
        "surgery",
        "medicine",
        mode="before",
    )
    @classmethod
    def canonicalize_category(cls, value: object, info: ValidationInfo) -> object:
        normalized = canonical_category(info.field_name, value)
        if normalized is None and info.field_name != "pregnant":
            raise ValueError(f"{info.field_name} is required")
        return normalized

    @model_validator(mode="after")
    def validate_nss_relationships(self) -> "NSS80CostPredictionInput":
        validate_consistency(self.model_dump())
        return self


class USBenchmarkPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(ge=18, le=64, description="Age within the training dataset range")
    sex: Literal["female", "male"]
    bmi: float = Field(ge=15.96, le=53.13, description="BMI within the training dataset range")
    children: int = Field(ge=0, le=5)
    smoker: Literal["no", "yes"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]

    @field_validator("sex", "smoker", "region", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


MLCostPredictionInput = USBenchmarkPredictionInput
