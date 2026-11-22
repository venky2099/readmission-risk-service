"""Feature engineering shared by training and serving.

FeatureEngineer turns the raw encounter fields (as they appear in the dataset and in API payloads) into model features.
It is the first step of the saved scikit-learn Pipeline, so the API applies exactly the same logic as training.
It learns nothing from data, so it cannot leak information between splits.
"""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MED_COLUMNS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
               "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
               "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
               "metformin-rosiglitazone", "metformin-pioglitazone"]
# The most common medicines stay as their own categorical features; the rest are summarised by the two counts below.
KEY_MEDS = ["metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride", "insulin"]

RAW_NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
               "number_emergency", "number_inpatient", "number_diagnoses"]
RAW_CATEGORICAL = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
                   "medical_specialty", "max_glu_serum", "A1Cresult", "change", "diabetesMed"]

NUMERIC_FEATURES = ["age_mid"] + RAW_NUMERIC + ["total_prior_visits", "n_meds_active", "n_meds_changed"]
CATEGORICAL_FEATURES = ([c for c in RAW_CATEGORICAL] + ["diag_1_group", "diag_2_group", "diag_3_group"]
                        + KEY_MEDS)


def icd9_group(code):
    """Group an ICD-9 code into the nine chapters used by Strack et al. (2014). Unknown formats become 'Other'."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if s in ("", "?", "nan", "None"):
        return "Missing"
    if s[0] in "VvEe":
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 250 <= v < 251:
        return "Diabetes"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


_AGE = re.compile(r"^\[(\d{1,3})-(\d{1,3})\)$")


def age_midpoint(bracket):
    """'[70-80)' -> 75.0. Anything else -> NaN (imputed later)."""
    m = _AGE.match(str(bracket)) if bracket is not None else None
    return (int(m.group(1)) + int(m.group(2))) / 2 if m else np.nan


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw encounter columns in, model feature columns out. Stateless."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        out["age_mid"] = X["age"].map(age_midpoint) if "age" in X else np.nan
        for c in RAW_NUMERIC:
            out[c] = pd.to_numeric(X[c], errors="coerce") if c in X else np.nan
        visits = out[["number_outpatient", "number_emergency", "number_inpatient"]]
        out["total_prior_visits"] = visits.sum(axis=1, min_count=1)
        meds = pd.DataFrame({c: (X[c] if c in X else "No") for c in MED_COLUMNS}, index=X.index).fillna("No")
        out["n_meds_active"] = (meds != "No").sum(axis=1).astype(float)
        out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)
        for c in RAW_CATEGORICAL:
            col = X[c] if c in X else pd.Series(np.nan, index=X.index)
            out[c] = col.astype(object).where(col.notna(), "Missing").astype(str)
        for i in (1, 2, 3):
            src = X[f"diag_{i}"] if f"diag_{i}" in X else pd.Series(np.nan, index=X.index)
            out[f"diag_{i}_group"] = src.map(icd9_group)
        for c in KEY_MEDS:
            out[c] = meds[c].astype(str)
        return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
