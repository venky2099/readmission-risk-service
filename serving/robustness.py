"""Robustness and edge-case checks through the real API (FastAPI TestClient, no network needed).

Run:  python -m serving.robustness
Each case sends one modified payload and records the HTTP status, the source of the answer and any warnings.
The last case simulates a scoring failure to show the fallback path. Writes reports/robustness.json.
"""
import json
from unittest import mock

from fastapi.testclient import TestClient

from serving import app as app_module
from serving.payloads import row_to_payload
from src import config
from src.train import load_parts


def cases(base):
    def drop(*keys):
        return {k: v for k, v in base.items() if k not in keys}
    return [
        ("valid encounter (test set row)", base, "scored"),
        ("optional fields missing (race, specialty, diagnoses)", drop("race", "medical_specialty", "diag_1", "diag_2", "diag_3"), "scored"),
        ("unseen medical specialty", {**base, "medical_specialty": "Space-Medicine"}, "scored with warning"),
        ("unseen ICD-9 code format", {**base, "diag_1": "X12.9"}, "scored, grouped as Other"),
        ("extreme outlier: 900 lab procedures", {**base, "num_lab_procedures": 900}, "clipped, warning"),
        ("extreme outlier: 450 prior inpatient stays", {**base, "number_inpatient": 450}, "clipped, warning"),
        ("negative count", {**base, "num_medications": -3}, "rejected 422"),
        ("wrong type: text in a number field", {**base, "time_in_hospital": "three"}, "rejected 422"),
        ("value outside closed vocabulary (insulin='Lots')", {**base, "insulin": "Lots"}, "rejected 422"),
        ("required field missing (age)", drop("age"), "rejected 422"),
        ("unknown field name (typo)", {**base, "num_medicatons": 5}, "rejected 422"),
        ("malformed age bracket", {**base, "age": "seventy"}, "rejected 422"),
        ("patient died at discharge (disposition 11)", {**base, "discharge_disposition_id": 11}, "not applicable"),
    ]


def main():
    te = load_parts(("test",))["test"]
    base = row_to_payload(te.iloc[0])
    client = TestClient(app_module.app)
    rows = []
    for name, payload, expected in cases(base):
        r = client.post("/predict", json=payload)
        body = r.json()
        rows.append({"case": name, "expected": expected, "status": r.status_code,
                     "source": body.get("source") if r.status_code == 200 else None,
                     "risk_score": body.get("risk_score") if r.status_code == 200 else None,
                     "flag": body.get("flag_for_follow_up") if r.status_code == 200 else None,
                     "detail": (body.get("warnings") if r.status_code == 200 else
                                [f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in body.get("detail", [])])})
    estimator = app_module.service.model.named_steps["model"]        # shared by the full pipeline and its slices
    with mock.patch.object(estimator, "predict_proba", side_effect=RuntimeError("simulated failure")):
        r = client.post("/predict", json=base)
        body = r.json()
        rows.append({"case": "model raises during scoring (simulated)", "expected": "fallback rule", "status": r.status_code,
                     "source": body.get("source"), "flag": body.get("flag_for_follow_up"), "detail": body.get("warnings")})
    rows[0]["payload_preview"] = {k: base[k] for k in ("encounter_id", "gender", "age", "admission_source_id", "time_in_hospital",
                                                       "number_inpatient", "diag_1")}
    (config.REPORTS / "robustness.json").write_text(json.dumps(rows, indent=1))
    for x in rows:
        print(f"{x['status']}  {x['source'] or '-':14s} {x['case']}")


if __name__ == "__main__":
    main()
