"""Self-contained production contract and HTTP tests for Practical 15."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.schemas.prediction import NSS80CostPredictionInput
from app.services.nss80_prediction_service import NSS80PredictionService
from model_training.nss80.practical15_intervals import bounds, calibration_adjustment
from model_training.nss80.production_practical15 import (
    FEATURES,
    OPTIONAL_FIELDS,
    SERVICE_RECEIPT_FEATURES,
    model_safe_values,
)
from model_training.nss80.production_v4 import FEATURES as OLD_FEATURES
from model_training.nss80.safe_improvement_v4 import PROHIBITED_FEATURES
from scripts import main


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "reports/nss80/practical_15_promotion/sample_api_request.json"


class TestClient:
    __test__ = False

    def __init__(self, app):
        self.app = app

    def request(self, method, path, body=None):
        async def exchange():
            payload = json.dumps(body).encode() if body is not None else b""
            messages = []
            delivered = False

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": payload, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                messages.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "root_path": "",
                "query_string": b"",
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 12345),
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"host", b"testserver"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
            await self.app(scope, receive, send)
            status = next(
                message["status"]
                for message in messages
                if message["type"] == "http.response.start"
            )
            text = b"".join(
                message.get("body", b"")
                for message in messages
                if message["type"] == "http.response.body"
            ).decode()
            return SimpleNamespace(
                status_code=status,
                text=text,
                json=lambda: json.loads(text),
            )

        return asyncio.run(exchange())

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, json):
        return self.request("POST", path, json)


@pytest.fixture(scope="module")
def service():
    return NSS80PredictionService()


@pytest.fixture(scope="module")
def request_values():
    return json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))


def test_frozen_production_through_http(service, request_values):
    request = NSS80CostPredictionInput(**request_values)
    transformed = model_safe_values(request.model_dump())
    frame = pd.DataFrame([transformed], columns=FEATURES)
    direct = round(max(0.0, float(service.estimator.predict(frame)[0])), 2)

    with patch.object(
        main,
        "get_system_settings_record",
        return_value={"allow_guest_predictions": True, "maintenance_mode": False},
    ):
        client = TestClient(main.app)
        options = client.get("/predict/nss80/options")
        response = client.post("/predict/nss80", request.model_dump())

    assert options.status_code == 200
    assert options.json()["field_order"] == FEATURES
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == "nss80_practical_15"
    assert body["currency"] == "INR"
    assert body["predicted_medical_expenditure"] == pytest.approx(direct)
    assert body["lower_bound"] <= direct <= body["upper_bound"]

    metadata = json.loads((ROOT / "models/nss80/metadata.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / "models/nss80/primary_model.joblib").read_bytes()).hexdigest()
    assert digest == metadata["model_sha256"]


def test_exact_contract_missing_removed_and_leakage(service, request_values):
    base = NSS80CostPredictionInput(**request_values).model_dump()
    assert list(NSS80CostPredictionInput.model_fields) == FEATURES
    assert len(FEATURES) == 15
    assert not set(FEATURES) & PROHIBITED_FEATURES
    for feature in set(FEATURES) - set(OPTIONAL_FIELDS):
        with pytest.raises(ValidationError):
            NSS80CostPredictionInput(**{key: value for key, value in base.items() if key != feature})
    for feature in (set(OLD_FEATURES) - set(FEATURES)) | PROHIBITED_FEATURES:
        with pytest.raises(ValidationError):
            NSS80CostPredictionInput(**{**base, feature: 0})


def test_service_encodings_are_binary(service, request_values):
    base = NSS80CostPredictionInput(**request_values).model_dump()
    for feature in SERVICE_RECEIPT_FEATURES:
        assert service.categories[feature] == ["Not received", "Received"]
        for label, expected in [("Not received", 0), ("Received", 1)]:
            assert model_safe_values({**base, feature: label})[feature] == expected
        with pytest.raises(ValueError):
            service.validate_input({**base, feature: "Paid"})


def test_person_calibration_and_interval_enlargement(service, request_values):
    adjustment, rank, count = calibration_adjustment(
        np.zeros(10), np.ones(10), np.arange(10.0), np.arange(10)
    )
    assert (adjustment, rank, count) == (8.0, 10, 10)
    frame = pd.DataFrame(
        [model_safe_values(NSS80CostPredictionInput(**request_values).model_dump())],
        columns=FEATURES,
    )
    lower, upper = bounds(service.interval, frame)
    expanded_lower, expanded_upper = bounds(service.interval, frame, np.array([1e8]))
    assert (expanded_lower <= lower).all()
    assert (expanded_upper >= upper).all()


def test_http_access_controls_unchanged(request_values):
    client = TestClient(main.app)
    for settings in [
        {"allow_guest_predictions": False, "maintenance_mode": False},
        {"allow_guest_predictions": True, "maintenance_mode": True},
    ]:
        with patch.object(main, "get_system_settings_record", return_value=settings):
            result = client.post("/predict/nss80", request_values)
            assert result.status_code in [401, 403, 503]
            assert client.get("/predict/nss80/options").status_code == 200
