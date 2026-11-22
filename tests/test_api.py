from unittest import mock

import pytest
from fastapi.testclient import TestClient

from src import config

if not config.MODEL_PATH.exists():
    pytest.skip("run python -m src.train first", allow_module_level=True)

from serving import app as app_module          # noqa: E402  (needs the trained artifact)

client = TestClient(app_module.app)

BASE = {"encounter_id": 1, "race": "Caucasian", "gender": "Female", "age": "[70-80)", "admission_type_id": 1,
        "discharge_disposition_id": 1, "admission_source_id": 7, "time_in_hospital": 5, "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 16, "number_outpatient": 0, "number_emergency": 1,
        "number_inpatient": 2, "diag_1": "428", "diag_2": "250.01", "diag_3": "401", "number_diagnoses": 9,
        "A1Cresult": ">8", "insulin": "Up", "change": "Ch", "diabetesMed": "Yes"}


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["capacity"] == config.CAPACITY


def test_valid_prediction_and_determinism():
    a = client.post("/predict", json=BASE).json(); b = client.post("/predict", json=BASE).json()
    assert a["source"] == "model" and a["eligible"] and 0 <= a["risk_score"] <= 1
    assert a["flag_for_follow_up"] == (a["risk_score"] >= a["threshold"]) and a["risk_score"] == b["risk_score"]


@pytest.mark.parametrize("change", [{"num_medications": -1}, {"time_in_hospital": "three"}, {"insulin": "Lots"},
                                    {"age": "seventy"}, {"gender": "X"}, {"num_medicatons": 5}])
def test_malformed_payloads_are_rejected(change):
    assert client.post("/predict", json={**BASE, **change}).status_code == 422


def test_missing_required_field_is_rejected():
    assert client.post("/predict", json={k: v for k, v in BASE.items() if k != "age"}).status_code == 422


def test_unseen_category_and_outlier_are_handled_with_warnings():
    body = client.post("/predict", json={**BASE, "medical_specialty": "Space-Medicine", "num_lab_procedures": 900}).json()
    assert body["source"] == "model" and len(body["warnings"]) == 2


def test_optional_fields_may_be_missing():
    payload = {k: v for k, v in BASE.items() if k not in ("race", "medical_specialty", "diag_2", "diag_3")}
    assert client.post("/predict", json=payload).json()["source"] == "model"


def test_death_or_hospice_is_not_scored():
    body = client.post("/predict", json={**BASE, "discharge_disposition_id": 11}).json()
    assert body["eligible"] is False and body["source"] == "not_applicable" and body["flag_for_follow_up"] is None


def test_fallback_rule_when_model_fails():
    est = app_module.service.model.named_steps["model"]
    with mock.patch.object(est, "predict_proba", side_effect=RuntimeError("boom")):
        body = client.post("/predict", json=BASE).json()
    assert body["source"] == "fallback_rule"
    assert body["flag_for_follow_up"] == (BASE["number_inpatient"] >= app_module.service.fallback_k)
