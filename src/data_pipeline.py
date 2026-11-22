"""Ingestion, cleaning and the leak-free split.

Run:  python -m src.data_pipeline
Writes data/processed/split.csv (encounter_id, patient_nbr, y, split) and prints an audit.
The split is grouped by patient: every encounter of one patient falls in the same part, so the model is never
tested on a patient it has already seen (many patients have several encounters).
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src import config


def load_raw(path=config.RAW_CSV):
    # '?' marks missing values. 'None' in A1Cresult / max_glu_serum means "test not done" and must stay a category,
    # so pandas' default NA strings are switched off.
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False,
                       dtype={"payer_code": str, "medical_specialty": str, "diag_1": str, "diag_2": str, "diag_3": str})


def audit(df):
    return {"rows": int(len(df)), "patients": int(df.patient_nbr.nunique()),
            "readmitted_lt30": int((df.readmitted == "<30").sum()),
            "md5": hashlib.md5(config.RAW_CSV.read_bytes()).hexdigest()}


def clean(df):
    """Remove encounters for which the follow-up decision makes no sense, and build the target."""
    before = len(df)
    df = df[~df.discharge_disposition_id.isin(config.EXCLUDED_DISPOSITIONS)].copy()
    df["y"] = (df.readmitted == "<30").astype(int)
    return df, before - len(df)


def split(df, seed=config.SEED):
    """Stratified by the target, grouped by patient: 20 folds -> 14 train, 3 validation, 3 test."""
    sgkf = StratifiedGroupKFold(n_splits=config.N_SPLIT_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df.y, groups=df.patient_nbr)):
        fold[idx] = k
    part = np.where(np.isin(fold, config.TEST_FOLDS), "test", np.where(np.isin(fold, config.VAL_FOLDS), "val", "train"))
    return pd.DataFrame({"encounter_id": df.encounter_id.values, "patient_nbr": df.patient_nbr.values,
                         "y": df.y.values, "split": part})


def main():
    raw = load_raw()
    a = audit(raw)
    assert (a["rows"], a["patients"], a["readmitted_lt30"]) == (config.EXPECTED_ROWS, config.EXPECTED_PATIENTS,
                                                                 config.EXPECTED_LT30), f"unexpected raw data: {a}"
    df, removed = clean(raw)
    s = split(df)
    groups = {p: set(s.loc[s.split == p, "patient_nbr"]) for p in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    config.SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(config.SPLIT_CSV, index=False)
    summary = s.groupby("split").agg(encounters=("y", "size"), patients=("patient_nbr", "nunique"), readmit_rate=("y", "mean"))
    out = {"raw": a, "excluded_expired_or_hospice": int(removed), "eligible_encounters": int(len(df)),
           "split": summary.reindex(["train", "val", "test"]).reset_index().to_dict("records"),
           "missing_share": raw.isna().mean().round(4)[lambda x: x > 0].to_dict()}
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "data_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
