# 30-day readmission risk service

A mini production ML system that flags diabetic patients at discharge for a follow-up call, built for the M.Sc. Data
Science & AI course *Machine Learning Model Engineering* at BITS Pilani (Graded Assignment 1). Author: S Venkatesh
(2025EM1200045). This is coursework on a public dataset; it is not a clinical device.

**The decision it supports.** When a discharge order is signed, the hospital system calls `POST /predict` for that
patient. The follow-up team can call 20 of every 100 eligible discharges, so the model flags the riskiest 20% (the
threshold is the 80th percentile of validation scores). Patients who died or went to hospice are not scored.

**Data.** Diabetes 130-US Hospitals for Years 1999-2008 (Clore, Cios, DeShazo and Strack, 2014), UCI Machine Learning
Repository, doi:10.24432/C5230J, CC BY 4.0: 101,766 encounters of 71,518 patients. Target: readmitted within 30 days.

## Quickstart (Python 3.11)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python run_all.py               # fetch, split, train, evaluate once on test, drift, robustness, benchmark, tests (~10 min)
```

Serve the trained model:

```bash
uvicorn serving.app:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"gender\": \"Female\", \"age\": \"[70-80)\", \"admission_type_id\": 1, \"discharge_disposition_id\": 1, \"admission_source_id\": 7, \"time_in_hospital\": 5, \"num_lab_procedures\": 45, \"num_procedures\": 1, \"num_medications\": 16, \"number_inpatient\": 2, \"number_diagnoses\": 9, \"diag_1\": \"428\", \"insulin\": \"Up\"}"
curl http://127.0.0.1:8000/metrics
```

Interactive API documentation: http://127.0.0.1:8000/docs

## What each step does

| Step | File | Output |
|---|---|---|
| Download from UCI | `scripts/fetch_data.py` | `data/raw/diabetic_data.csv` (not committed) |
| Clean, exclude death/hospice, patient-grouped stratified split 70/15/15 | `src/data_pipeline.py` | `data/processed/split.csv`, `reports/data_audit.json` |
| Baselines, tuning on validation, capacity threshold, export | `src/train.py` | `artifacts/*.joblib`, `artifacts/metadata.json`, `reports/validation.json` |
| One test evaluation with patient-grouped bootstrap intervals | `src/evaluate.py` | `reports/test_metrics.json`, `reports/figures/` |
| Drift check (PSI, KS, score PSI, flagged share) | `src/monitor.py` | `reports/drift_report.json` |
| API: `/health`, `/predict`, `/metrics`, Pydantic validation, fallback rule | `serving/app.py`, `serving/schemas.py` | |
| Edge cases through the API | `serving/robustness.py` | `reports/robustness.json` |
| Latency p50/p95/p99 and throughput, 1 and 4 workers | `serving/benchmark.py` | `reports/benchmark.json` |
| Unit tests (schemas, inference, split, no leakage) | `tests/` | `pytest -q` |

## Design rules

* **No leakage.** Every encounter of a patient sits in one part only. Imputers, scaler and encoder are fitted on the
  training part inside one scikit-learn `Pipeline`; a unit test checks that the saved scaler's means equal the
  training means. Feature engineering (`src/features.py`) is stateless.
* **Test set used once.** `src/train.py` never loads it; `src/evaluate.py` refuses to run twice unless asked.
* **Imbalance.** About 11% are readmitted. Class-weighted loss, a capacity-based threshold set on validation, and
  metrics that ignore the easy majority: PR-AUC, recall and precision at 20% flagged, lift over random calling.
* **Graceful degradation.** Malformed payloads get HTTP 422 with the reason; unseen categories and out-of-range
  numbers are handled with warnings; if the model is missing or fails, the service answers with the transparent
  fallback rule (prior inpatient stays) and says so in `source`.
