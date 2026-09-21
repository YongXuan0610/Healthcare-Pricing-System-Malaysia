import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.prediction import NSS80CostPredictionInput, USBenchmarkPredictionInput
from app.schemas.service_assistant import (
    ServiceAssistantRequest,
    ServiceAssistantResponse,
)
from app.services.liam_pricing_service import LIAMPricingReferenceService
from app.services.nss80_prediction_service import NSS80PredictionService
from app.services.prediction_service import USKaggleBenchmarkService
from app.services.public_pricing_service import HOSPITALS, PublicPricingService
from app.services.service_assistant import HealthcareServiceAssistant
from app.services.supabase_pricing_service import (
    PricingDataStoreError,
    SupabasePricingDataAPI,
    SupabasePrivatePricingService,
    SupabasePublicPricingService,
)
from pydantic import BaseModel, Field

BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def parse_frontend_origins(raw_value: str | None = None) -> list[str]:
    """Return explicit browser origins for CORS without permitting wildcards."""

    configured_value = (
        os.getenv("FRONTEND_ORIGINS", "")
        if raw_value is None
        else raw_value
    )
    candidates = configured_value.split(",") if configured_value.strip() else []
    if not candidates:
        candidates = list(DEFAULT_FRONTEND_ORIGINS)

    origins: list[str] = []
    for candidate in candidates:
        origin = candidate.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ValueError(
                "FRONTEND_ORIGINS must contain explicit origins, not '*'."
            )

        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "FRONTEND_ORIGINS entries must be valid HTTP(S) origins."
            )

        if origin not in origins:
            origins.append(origin)

    if not origins:
        return list(DEFAULT_FRONTEND_ORIGINS)
    return origins


frontend_origins = parse_frontend_origins()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

pricing_api = SupabasePricingDataAPI()
public_pricing_service = SupabasePublicPricingService(pricing_api)
private_pricing_service = SupabasePrivatePricingService(pricing_api)
liam_pricing_service = LIAMPricingReferenceService()
service_assistant = HealthcareServiceAssistant()

_nss80_prediction_service: NSS80PredictionService | None = None
_nss80_loading_attempted = False
_us_benchmark_service: USKaggleBenchmarkService | None = None
_us_benchmark_loading_attempted = False


def get_nss80_prediction_service() -> NSS80PredictionService | None:
    """Load the NSS primary artifacts once without blocking pricing endpoints."""

    global _nss80_loading_attempted, _nss80_prediction_service
    if _nss80_loading_attempted:
        return _nss80_prediction_service

    _nss80_loading_attempted = True
    try:
        _nss80_prediction_service = NSS80PredictionService()
    except Exception:
        logger.exception("NSS 80 primary-model artifacts could not be loaded")
        _nss80_prediction_service = None
    return _nss80_prediction_service


def get_us_benchmark_service() -> USKaggleBenchmarkService | None:
    """Load the isolated US Kaggle benchmark artifacts on demand."""

    global _us_benchmark_loading_attempted, _us_benchmark_service
    if _us_benchmark_loading_attempted:
        return _us_benchmark_service

    _us_benchmark_loading_attempted = True
    try:
        _us_benchmark_service = USKaggleBenchmarkService()
    except Exception:
        logger.exception("US Kaggle benchmark artifacts could not be loaded")
        _us_benchmark_service = None
    return _us_benchmark_service


get_ml_prediction_service = get_nss80_prediction_service


class PublicPredictionInput(BaseModel):
    category: str
    state: str | None = None
    citizenship: str
    hospital: str = Field(default="all", max_length=10)
    service_name: str | None = Field(default=None, max_length=120)


class PrivatePredictionInput(BaseModel):
    hospital: str
    package_name: str
    ward_type: str
    nights: int = Field(ge=0, le=30)


def map_citizenship_to_patient_class(citizenship: str) -> str:
    normalized = citizenship.strip().lower()
    mappings = {
        "malaysian": "citizen",
        "citizen": "citizen",
        "non-malaysian": "foreigner",
        "foreigner": "foreigner",
        "foreign": "foreigner",
    }
    patient_class = mappings.get(normalized)
    if patient_class is None:
        raise HTTPException(
            status_code=422,
            detail="Citizenship must be Malaysian or Non-Malaysian.",
        )
    return patient_class


def pricing_database_error(exc: PricingDataStoreError) -> HTTPException:
    logger.exception("Supabase pricing database error")
    return HTTPException(
        status_code=503,
        detail=(
            "The pricing database is temporarily unavailable. "
            "Please verify the backend Supabase configuration and try again."
        ),
    )


def get_system_settings_record() -> dict:
    """Read the single public system settings row from Supabase."""

    try:
        rows = pricing_api.select(
            "system_settings",
            select=(
                "id,system_name,contact_email,"
                "allow_guest_predictions,maintenance_mode"
            ),
            filters={"id": "eq.1"},
            limit=1,
        )
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    if not rows:
        raise HTTPException(
            status_code=503,
            detail="System settings are unavailable.",
        )

    return rows[0]


def get_authenticated_identity(
    authorization: str | None,
) -> dict | None:
    """Resolve a Supabase bearer token into the current user and profile role."""

    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="A valid Supabase bearer token is required.",
        )

    try:
        pricing_api._ensure_configured()
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    headers = {
        "apikey": pricing_api.publishable_key,
        "Authorization": f"Bearer {token.strip()}",
        "Accept": "application/json",
    }

    try:
        user_response = pricing_api.session.get(
            f"{pricing_api.supabase_url}/auth/v1/user",
            headers=headers,
            timeout=pricing_api.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Authentication verification is temporarily unavailable.",
        ) from exc

    if user_response.status_code in {401, 403}:
        raise HTTPException(
            status_code=401,
            detail="Your login session is invalid or has expired.",
        )

    if not user_response.ok:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify the current login session.",
        )

    try:
        user_payload = user_response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Authentication verification returned an invalid response.",
        ) from exc

    user_id = user_payload.get("id")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Unable to identify the current user.",
        )

    try:
        profile_response = pricing_api.session.get(
            f"{pricing_api.supabase_url}/rest/v1/profiles",
            headers=headers,
            params={
                "select": "role",
                "id": f"eq.{user_id}",
                "limit": "1",
            },
            timeout=pricing_api.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify the current user role.",
        ) from exc

    if not profile_response.ok:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify the current user role.",
        )

    try:
        profiles = profile_response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="User role verification returned an invalid response.",
        ) from exc

    role = "user"
    if isinstance(profiles, list) and profiles:
        role = str(profiles[0].get("role") or "user")

    return {
        "id": user_id,
        "email": user_payload.get("email"),
        "role": role,
    }


def enforce_pricing_prediction_access(
    authorization: str | None,
) -> None:
    """Apply maintenance and guest-prediction settings server-side."""

    settings = get_system_settings_record()
    maintenance_mode = bool(settings.get("maintenance_mode"))
    allow_guest_predictions = bool(
        settings.get("allow_guest_predictions", True)
    )

    identity: dict | None = None

    if authorization:
        identity = get_authenticated_identity(authorization)

    if maintenance_mode:
        if not identity or identity.get("role") != "admin":
            raise HTTPException(
                status_code=503,
                detail=(
                    "The pricing system is currently in maintenance mode. "
                    "Please try again later."
                ),
            )

    if not allow_guest_predictions and identity is None:
        raise HTTPException(
            status_code=401,
            detail="Please sign in to use the healthcare pricing tools.",
        )


@app.get("/")
def root():
    return {
        "message": "Backend is running",
        "components": {
            "primary_model": "NSS 80th Round India inpatient medical expenditure",
            "secondary_benchmark": "US Kaggle medical charges",
            "malaysia_reference": "LIAM published private healthcare price ranges",
            "pricing_database": "Supabase PostgreSQL",
        },
    }


@app.get("/system/settings")
def get_system_settings():
    settings = get_system_settings_record()

    return {
        "system_name": settings.get("system_name") or "MyCareCost",
        "contact_email": settings.get("contact_email") or "",
        "allow_guest_predictions": bool(
            settings.get("allow_guest_predictions", True)
        ),
        "maintenance_mode": bool(settings.get("maintenance_mode")),
    }


@app.get("/pricing/health")
def pricing_health():
    try:
        private_hospitals = private_pricing_service.list_hospitals()
        public_categories = public_pricing_service.list_categories()
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    return {
        "status": "ok",
        "data_store": "supabase_postgresql",
        "private_hospitals": len(private_hospitals),
        "public_categories": len(public_categories),
    }


@app.get("/private/hospitals")
def get_private_hospitals():
    try:
        hospitals = private_pricing_service.list_hospitals()
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    return {"hospitals": hospitals}


@app.get("/public/categories")
def get_public_categories():
    try:
        categories = public_pricing_service.list_categories()
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    return {"categories": categories}


@app.get("/public/hospitals")
def get_public_hospitals():
    try:
        hospitals = public_pricing_service.list_hospitals()
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc
    return {"hospitals": hospitals}


@app.post(
    "/assistant/recommend-service",
    response_model=ServiceAssistantResponse,
)
def recommend_healthcare_service(
    data: ServiceAssistantRequest,
) -> ServiceAssistantResponse:
    try:
        return service_assistant.recommend(data.message)
    except Exception as exc:
        logger.exception("Healthcare Service Assistant failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "The Healthcare Service Assistant is temporarily unavailable. "
                "Please select a service manually."
            ),
        ) from exc


@app.get("/private/packages/{hospital}")
def get_private_hospital_packages(hospital: str):
    try:
        packages = private_pricing_service.list_packages(hospital)
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    return {
        "hospital": hospital,
        "packages": packages,
    }


@app.get("/private/wards/{hospital}")
def get_private_hospital_wards(hospital: str):
    try:
        ward_rates = private_pricing_service.list_wards(hospital)
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    return {
        "hospital": hospital,
        "ward_rates": [
            {
                "id": "no-stay",
                "name": "No Stay",
                "dailyRate": 0,
                "description": "Outpatient treatment with no overnight stay",
                "rateBasis": "No stay",
                "notes": "",
            },
            *ward_rates,
        ],
    }


@app.post("/predict/public")
def predict_public(
    data: PublicPredictionInput,
    authorization: str | None = Header(default=None),
):
    enforce_pricing_prediction_access(authorization)
    patient_class = map_citizenship_to_patient_class(data.citizenship)

    try:
        pricing_reference, matched_charges = (
            public_pricing_service.get_pricing_reference(
                category=data.category,
                service_name=data.service_name,
                patient_class=patient_class,
                hospital_code=data.hospital,
            )
        )
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    available_codes = {item["hospital_code"] for item in HOSPITALS}
    selected_hospital = data.hospital.strip().upper()
    if selected_hospital not in {"ALL", "*"} and selected_hospital not in available_codes:
        raise HTTPException(status_code=422, detail="Select a valid public hospital or All Public Hospitals.")

    hospitals = PublicPricingService.build_hospital_comparison(
        matched_charges,
        hospital_code=data.hospital,
    )
    national_references = PublicPricingService.build_national_references(
        matched_charges
    )
    estimate_available = pricing_reference["estimate_available"]
    backward_compatible_cost = pricing_reference["typical_estimate"]
    summary_hospital_code = next(
        (
            hospital["hospital_code"]
            for hospital in hospitals
            if hospital["statistics"]["estimate_available"]
        ),
        None,
    )

    return {
        "type": "public",
        "estimate_available": estimate_available,
        "pricing_type": pricing_reference["pricing_type"],
        "predicted_cost": backward_compatible_cost,
        "published_cost": pricing_reference["published_cost"],
        "lower_estimate": pricing_reference["lower_estimate"],
        "typical_estimate": pricing_reference["typical_estimate"],
        "upper_estimate": pricing_reference["upper_estimate"],
        "records_used": pricing_reference["records_used"],
        "range_method": pricing_reference["range_method"],
        "currency": "MYR",
        "input_received": data.model_dump(),
        "patient_class": patient_class,
        "pricing_source": (
            "official_public_dataset" if estimate_available else "unavailable"
        ),
        "summary_hospital_code": summary_hospital_code,
        "data_store": "supabase_postgresql",
        "message": (
            None
            if estimate_available
            else (
                "No matching published public hospital price is currently "
                "available for this selection."
            )
        ),
        "state_adjusted": False,
        "hospital_filter": data.hospital,
        "hospital_sources_checked": (
            len(HOSPITALS)
            if selected_hospital in {"ALL", "*"}
            else 1
        ),
        "pricing_scope": (
            "Public pricing references are based on five official hospital datasets "
            "and separate Ministry of Health Malaysia national schedules. Published "
            "charges are not adjusted mathematically by state."
        ),
        "methodology": (
            "One record is shown as an exact published charge; two to four records use "
            "the published minimum and maximum; five or more records use P25, median, "
            "and P75 from matching values within a single hospital. Hospitals are not "
            "pooled into a market statistic, and MOH schedules remain separate."
        ),
        "hospitals": hospitals,
        "national_references": national_references,
        "matched_public_charges": matched_charges[
            matched_charges["source_type"].eq("hospital")
            if "source_type" in matched_charges.columns
            else [True] * len(matched_charges)
        ].head(100).to_dict(
            orient="records"
        ),
    }


@app.post("/predict/private")
def predict_private(
    data: PrivatePredictionInput,
    authorization: str | None = Header(default=None),
):
    enforce_pricing_prediction_access(authorization)

    try:
        package = private_pricing_service.get_package(
            hospital=data.hospital,
            package_name=data.package_name,
        )

        if package is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No pricing package named '{data.package_name}' was found "
                    f"for {data.hospital}."
                ),
            )

        package_price = float(package["price"])

        if data.ward_type == "No Stay":
            ward_price_per_night = 0.0
        else:
            ward = private_pricing_service.get_ward(
                hospital_key=data.hospital,
                ward_type=data.ward_type,
            )

            if ward is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No ward rate named '{data.ward_type}' was found "
                        f"for {data.hospital}."
                    ),
                )

            ward_price_per_night = float(ward["dailyRate"])

        matching_packages = private_pricing_service.list_packages(
            data.hospital,
        )
    except PricingDataStoreError as exc:
        raise pricing_database_error(exc) from exc

    ward_cost = ward_price_per_night * data.nights
    total_cost = round(package_price + ward_cost, 2)

    exact_and_related_matches = [
        row
        for row in matching_packages
        if data.package_name.casefold() in row["name"].casefold()
    ][:3]

    return {
        "type": "private",
        "predicted_cost": total_cost,
        "currency": "MYR",
        "input_received": data.model_dump(),
        "breakdown": {
            "package_price": package_price,
            "ward_price_per_night": ward_price_per_night,
            "ward_cost": ward_cost,
            "surgeon_fee": None,
            "misc_fee": None,
        },
        "matching_private_packages": exact_and_related_matches,
        "additional_charges_included": False,
        "data_store": "supabase_postgresql",
        "message": (
            "This reference includes only published package and ward charges available "
            "in the database. Additional medical, professional, medication, diagnostic, "
            "implant, and miscellaneous charges may apply."
        ),
    }


@app.post("/predict/model")
@app.post("/predict/nss80")
def predict_with_model(
    data: NSS80CostPredictionInput,
    authorization: str | None = Header(default=None),
):
    enforce_pricing_prediction_access(authorization)

    nss80_service = get_nss80_prediction_service()
    if nss80_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The NSS 80 primary research model is temporarily unavailable. "
                "Public and private pricing references remain available."
            ),
        )
    try:
        prediction = nss80_service.predict(data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"type": "machine_learning_primary", **prediction}


@app.get("/predict/nss80/options")
def get_nss80_prediction_options():
    nss80_service = get_nss80_prediction_service()
    if nss80_service is None:
        raise HTTPException(
            status_code=503,
            detail="The NSS 80 practical model options are temporarily unavailable.",
        )
    return nss80_service.prediction_options()


@app.post("/predict/benchmark/us")
def predict_us_benchmark(
    data: USBenchmarkPredictionInput,
    authorization: str | None = Header(default=None),
):
    enforce_pricing_prediction_access(authorization)

    benchmark_service = get_us_benchmark_service()
    if benchmark_service is None:
        raise HTTPException(
            status_code=503,
            detail="The secondary US Kaggle benchmark is temporarily unavailable.",
        )
    return {
        "type": "machine_learning_benchmark",
        **benchmark_service.predict(data.model_dump()),
    }


@app.get("/pricing/reference/liam/procedures")
def list_liam_procedures():
    return {
        "component": "malaysia_liam_pricing_reference",
        "research_role": "pricing_reference_only",
        "procedures": liam_pricing_service.list_procedures(),
    }


@app.get("/pricing/reference/liam")
def get_liam_pricing_reference(
    procedure_code: str = Query(min_length=1),
    care_setting: str | None = None,
    segmentation_type: str = "Overall",
    segment: str = "All",
):
    try:
        return liam_pricing_service.get_reference(
            procedure_code=procedure_code,
            care_setting=care_setting,
            segmentation_type=segmentation_type,
            segment=segment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
