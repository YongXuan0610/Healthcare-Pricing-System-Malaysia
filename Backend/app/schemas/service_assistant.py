"""Request and response schemas for the Healthcare Service Assistant."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ServiceAssistantStatus = Literal["matched", "ambiguous", "unmatched", "urgent"]


class ServiceAssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=500)


class ServiceSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str
    mapped_service: str
    service_name: str | None = None
    category: str
    confidence: float = Field(ge=0, le=1)
    message: str


class ServiceAssistantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ServiceAssistantStatus
    service: str | None = None
    mapped_service: str | None = None
    service_name: str | None = None
    category: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    message: str
    suggestions: list[ServiceSuggestion] = Field(default_factory=list)
