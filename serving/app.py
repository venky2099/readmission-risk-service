"""FastAPI service: GET /health, POST /predict, GET /metrics.

Run:  uvicorn serving.app:app --host 127.0.0.1 --port 8000
The model and its metadata are loaded once at start-up. If the model cannot be loaded or fails while scoring,
the service does not go down: it answers with the transparent fallback rule (prior inpatient stays) and says so
in the response ("source": "fallback_rule"), and /health reports "degraded".
"""
import json
import logging
import os
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request

from serving.schemas import Encounter, Health, Prediction
from src import config
from src.features import RAW_NUMERIC

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("readmission-service")


class ModelService:
    def __init__(self, model_path=None, metadata_path=None):
        self.model, self.error = None, None
        self.meta = json.loads(Path(metadata_path or config.METADATA_PATH).read_text())
        try:
            self.model = joblib.load(model_path or os.environ.get("MODEL_PATH", config.MODEL_PATH))
            onehot = self.model.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
            cat_cols = self.model.named_steps["prep"].transformers_[1][2]
            self.known = {c: set(map(str, cats)) for c, cats in zip(cat_cols, onehot.categories_)}
        except Exception as exc:                                   # missing or corrupt artifact
            self.error = f"{type(exc).__name__}: {exc}"
            self.known = {}
            log.error("model not loaded, serving fallback rule only: %s", self.error)
        self.threshold = float(self.meta["threshold"])
        self.capacity = float(self.meta["capacity"])
        self.ranges = self.meta["train_numeric_ranges"]
        self.fallback_k = float(self.meta["fallback_rule"]["flag_if_at_least"])

    def _soft_checks(self, row):
        """Clip numeric values above the training range, engineer features once, and note unseen categories."""
        warnings = []
        for c in RAW_NUMERIC:
            hi = self.ranges[c][1]
            if row[c] is not None and row[c] > hi:
                warnings.append(f"{c}={row[c]} is above the training maximum {hi:g}; clipped to {hi:g}")
                row[c] = hi
        eng = None
        if self.model is not None:
            eng = self.model.named_steps["engineer"].transform(pd.DataFrame([row]))
            first = eng.iloc[0]
            for c, known in self.known.items():
                if str(first[c]) not in known:
                    warnings.append(f"{c}='{first[c]}' was not seen in training; treated as an infrequent category")
        return row, eng, warnings

    def predict(self, enc: Encounter) -> Prediction:
        row = enc.model_dump(by_alias=True)
        base = dict(encounter_id=enc.encounter_id, threshold=self.threshold, capacity=self.capacity,
                    model_version=self.meta["model_version"])
        if enc.discharge_disposition_id in config.EXCLUDED_DISPOSITIONS:
            return Prediction(**base, eligible=False, risk_score=None, flag_for_follow_up=None, source="not_applicable",
                              warnings=["discharge disposition is death or hospice: no readmission follow-up applies"])
        row, eng, warnings = self._soft_checks(row)
        if self.model is not None:
            try:
                score = float(self.model[1:].predict_proba(eng)[0, 1])     # features already engineered above
                if not np.isfinite(score):
                    raise ValueError("non-finite score")
                return Prediction(**base, eligible=True, risk_score=round(score, 6), flag_for_follow_up=score >= self.threshold,
                                  source="model", warnings=warnings)
            except Exception as exc:
                log.error("model scoring failed (%s: %s); fallback rule used", type(exc).__name__, exc)
                warnings.append(f"model scoring failed ({type(exc).__name__}); fallback rule used")
        else:
            warnings.append("model not loaded; fallback rule used")
        flag = row["number_inpatient"] >= self.fallback_k
        return Prediction(**base, eligible=True, risk_score=None, flag_for_follow_up=bool(flag), source="fallback_rule",
                          warnings=warnings)


app = FastAPI(title="30-day readmission risk service", version=config.MODEL_VERSION,
              description="Flags discharges for a follow-up call; capacity 20 of every 100 eligible discharges.")
service = ModelService()
STATS = {"latency_ms": deque(maxlen=5000), "status": Counter(), "source": Counter(), "flags": Counter()}


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    if request.url.path == "/predict":
        STATS["latency_ms"].append(ms)
        STATS["status"][response.status_code] += 1
    response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
    return response


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok" if service.model is not None else "degraded", model_loaded=service.model is not None,
                  model_version=service.meta["model_version"], threshold=service.threshold, capacity=service.capacity)


@app.post("/predict", response_model=Prediction)
def predict(encounter: Encounter):
    pred = service.predict(encounter)
    STATS["source"][pred.source] += 1
    if pred.flag_for_follow_up is not None:
        STATS["flags"][bool(pred.flag_for_follow_up)] += 1
    log.info("encounter=%s source=%s flag=%s warnings=%d", pred.encounter_id, pred.source, pred.flag_for_follow_up,
             len(pred.warnings))
    return pred


@app.get("/metrics")
def metrics():
    lat = np.array(STATS["latency_ms"]) if STATS["latency_ms"] else np.array([np.nan])
    n_flag = STATS["flags"][True] + STATS["flags"][False]
    total = sum(STATS["status"].values())
    return {"requests": total, "status_counts": dict(STATS["status"]), "source_counts": dict(STATS["source"]),
            "error_rate": (total - STATS["status"][200]) / total if total else 0.0,
            "flagged_share": STATS["flags"][True] / n_flag if n_flag else None,
            "latency_ms_p50": float(np.nanpercentile(lat, 50)), "latency_ms_p95": float(np.nanpercentile(lat, 95)),
            "latency_ms_p99": float(np.nanpercentile(lat, 99))}
