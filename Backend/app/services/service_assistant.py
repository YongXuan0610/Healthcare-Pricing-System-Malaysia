"""Deterministic navigation helper for available public healthcare services."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from app.schemas.service_assistant import (
    ServiceAssistantResponse,
    ServiceSuggestion,
)


WARD_CHARGES_CATEGORY = "Ward Charges"
MINIMUM_MATCH_SCORE = 4.0
AMBIGUITY_RATIO = 0.65


@dataclass(frozen=True)
class ServiceDefinition:
    display_name: str
    mapped_service: str
    service_name: str | None
    category: str
    message: str
    weighted_terms: tuple[tuple[str, float], ...]


# Each mapping targets a normalized category returned by /public/categories.
# Guided selections do not use this matcher; it remains the deterministic fallback.
SERVICE_MAPPINGS: dict[str, ServiceDefinition] = {
    "physiotherapy": ServiceDefinition(
        display_name="Physiotherapy",
        mapped_service="Physiotherapy",
        service_name=None,
        category="Physiotherapy",
        message=(
            "Official published physiotherapy charges are available in the "
            "Public Hospital pricing section."
        ),
        weighted_terms=(
            ("physiotherapy", 6.0),
            ("physical therapy", 6.0),
            ("rehabilitation", 5.0),
            ("rehab", 5.0),
            ("difficulty walking", 4.0),
            ("movement problem", 4.0),
            ("movement problems", 4.0),
            ("mobility problem", 4.0),
            ("mobility problems", 4.0),
            ("sports injury", 4.0),
            ("knee pain", 4.0),
            ("back pain", 4.0),
            ("muscle pain", 4.0),
            ("joint pain", 4.0),
            ("mobility", 2.5),
            ("movement", 2.0),
            ("knee", 2.0),
            ("walking", 1.5),
            ("injury", 1.5),
        ),
    ),
    "nephrology": ServiceDefinition(
        display_name="Nephrology",
        mapped_service="Nephrology",
        service_name=None,
        category="Nephrology",
        message=(
            "Official published nephrology charges are available in the Public "
            "Hospital pricing section."
        ),
        weighted_terms=(
            ("nephrology", 6.0),
            ("kidney problem", 6.0),
            ("kidney care", 6.0),
            ("renal problem", 6.0),
            ("renal care", 6.0),
            ("dialysis", 6.0),
            ("kidney", 4.5),
            ("renal", 4.5),
        ),
    ),
    "dietetics": ServiceDefinition(
        display_name="Dietetics / Nutrition",
        mapped_service="Dietetics",
        service_name=None,
        category="Dietetics",
        message=(
            "Official published dietetics charges are available in the Public "
            "Hospital pricing section."
        ),
        weighted_terms=(
            ("dietetics", 6.0),
            ("dietitian", 6.0),
            ("nutrition advice", 6.0),
            ("nutrition assessment", 6.0),
            ("dietary advice", 5.0),
            ("meal planning", 5.0),
            ("weight management", 5.0),
            ("nutrition", 5.0),
            ("diet", 4.0),
            ("dietary", 4.0),
        ),
    ),
    "oncology": ServiceDefinition(
        display_name="Radiotherapy / Oncology",
        mapped_service="Radiotherapy / Oncology",
        service_name=None,
        category="Radiotherapy / Oncology",
        message=(
            "Official published radiotherapy and oncology charges are available "
            "in the Public Hospital pricing section."
        ),
        weighted_terms=(
            ("radiotherapy", 6.0),
            ("radiation therapy", 6.0),
            ("oncology", 6.0),
            ("oncologist", 6.0),
            ("cancer related", 6.0),
            ("cancer care", 6.0),
            ("cancer", 5.0),
        ),
    ),
    "diagnostic_imaging": ServiceDefinition(
        display_name="Radiology / Imaging",
        mapped_service="Radiology / Imaging",
        service_name=None,
        category="Radiology / Imaging",
        message=(
            "Official published radiology and imaging charges are available in "
            "the Public Hospital pricing section."
        ),
        weighted_terms=(
            ("x ray", 6.0),
            ("radiology", 6.0),
            ("diagnostic imaging", 6.0),
            ("ct scan", 6.0),
            ("mri scan", 6.0),
            ("ultrasound", 5.0),
            ("imaging", 5.0),
            ("scan", 3.5),
        ),
    ),
    "laboratory_testing": ServiceDefinition(
        display_name="Laboratory",
        mapped_service="Laboratory",
        service_name=None,
        category="Laboratory",
        message=(
            "Official published laboratory charges are available in the Public "
            "Hospital pricing section."
        ),
        weighted_terms=(
            ("blood test", 6.0),
            ("urine test", 6.0),
            ("laboratory test", 6.0),
            ("lab test", 6.0),
            ("diagnostic test", 5.0),
            ("diagnostic testing", 5.0),
            ("blood work", 5.0),
            ("laboratory", 4.5),
            ("investigation", 4.0),
        ),
    ),
    "ward_admission": ServiceDefinition(
        display_name="Ward / Admission",
        mapped_service=WARD_CHARGES_CATEGORY,
        service_name=None,
        category=WARD_CHARGES_CATEGORY,
        message=(
            "Official published ward charges are available in the Public Hospital "
            "pricing section."
        ),
        weighted_terms=(
            ("ward charges", 6.0),
            ("hospital room", 6.0),
            ("room charges", 6.0),
            ("hospital admission", 6.0),
            ("overnight stay", 5.0),
            ("ward", 5.0),
            ("admission", 5.0),
            ("inpatient room", 5.0),
            ("bed charges", 5.0),
        ),
    ),
}


EMERGENCY_PHRASES = (
    "severe difficulty breathing",
    "struggling to breathe",
    "cannot breathe",
    "can t breathe",
    "not breathing",
    "unconscious",
    "unresponsive",
    "not waking up",
    "uncontrolled bleeding",
    "severe bleeding",
    "bleeding will not stop",
    "bleeding won t stop",
    "severe chest pain",
    "crushing chest pain",
    "stroke symptoms",
    "signs of stroke",
    "having a seizure",
    "active seizure",
    "seizure now",
    "severe traumatic injury",
    "major trauma",
    "badly injured in an accident",
)

EMERGENCY_TERM_COMBINATIONS = (
    ("face drooping", "arm weakness"),
    ("heavy bleeding", "not stopping"),
    ("major accident", "severe injury"),
)

URGENT_MESSAGE = (
    "Your description may involve an urgent health concern. The Healthcare "
    "Service Assistant is designed only to help users locate healthcare pricing "
    "categories and should not be used for emergency assessment. Please seek "
    "urgent medical attention where appropriate."
)

UNMATCHED_MESSAGE = (
    "I am not confident which pricing category matches your description. Please "
    "provide a little more information or use the guided category list. You could "
    "mention rehabilitation, imaging, laboratory testing, nutrition, oncology, "
    "kidney services, or ward pricing."
)

AMBIGUOUS_MESSAGE = (
    "More than one available service may match your description. Select the one "
    "that best reflects the pricing information you want to view."
)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    return f" {normalized_phrase} " in f" {normalized_text} "


def _has_emergency_signal(normalized_text: str) -> bool:
    if any(_contains_phrase(normalized_text, phrase) for phrase in EMERGENCY_PHRASES):
        return True

    return any(
        all(_contains_phrase(normalized_text, term) for term in terms)
        for terms in EMERGENCY_TERM_COMBINATIONS
    )


def _score_service(normalized_text: str, definition: ServiceDefinition) -> float:
    return sum(
        weight
        for term, weight in definition.weighted_terms
        if _contains_phrase(normalized_text, term)
    )


def _confidence(score: float, competing_score: float) -> float:
    evidence_component = min(score, 8.0) / 20.0
    separation_component = min(max(score - competing_score, 0.0), 4.0) * 0.025
    return round(min(0.98, 0.48 + evidence_component + separation_component), 2)


class HealthcareServiceAssistant:
    """Recommend only services represented by the controlled mapping above."""

    def recommend(self, message: str) -> ServiceAssistantResponse:
        normalized_text = normalize_text(message)

        if _has_emergency_signal(normalized_text):
            return ServiceAssistantResponse(
                status="urgent",
                message=URGENT_MESSAGE,
            )

        ranked_matches = sorted(
            (
                (_score_service(normalized_text, definition), mapping_key, definition)
                for mapping_key, definition in SERVICE_MAPPINGS.items()
            ),
            key=lambda match: (-match[0], match[1]),
        )

        top_score, _, top_definition = ranked_matches[0]
        second_score = ranked_matches[1][0]

        if top_score < MINIMUM_MATCH_SCORE:
            return ServiceAssistantResponse(
                status="unmatched",
                message=UNMATCHED_MESSAGE,
            )

        is_ambiguous = (
            second_score >= MINIMUM_MATCH_SCORE
            and second_score >= top_score * AMBIGUITY_RATIO
        )

        if is_ambiguous:
            suggestions: list[ServiceSuggestion] = []
            for index, (score, _, definition) in enumerate(ranked_matches[:2]):
                competing_score = ranked_matches[1 - index][0]
                suggestions.append(
                    self._build_suggestion(definition, score, competing_score)
                )

            return ServiceAssistantResponse(
                status="ambiguous",
                message=AMBIGUOUS_MESSAGE,
                suggestions=suggestions,
            )

        suggestion = self._build_suggestion(
            top_definition,
            top_score,
            second_score,
        )
        return ServiceAssistantResponse(
            status="matched",
            service=suggestion.service,
            mapped_service=suggestion.mapped_service,
            service_name=suggestion.service_name,
            category=suggestion.category,
            confidence=suggestion.confidence,
            message=suggestion.message,
        )

    @staticmethod
    def _build_suggestion(
        definition: ServiceDefinition,
        score: float,
        competing_score: float,
    ) -> ServiceSuggestion:
        return ServiceSuggestion(
            service=definition.display_name,
            mapped_service=definition.mapped_service,
            service_name=definition.service_name,
            category=definition.category,
            confidence=_confidence(score, competing_score),
            message=definition.message,
        )
