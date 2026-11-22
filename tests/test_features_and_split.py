import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import FeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES, age_midpoint, icd9_group


@pytest.mark.parametrize("code,group", [("250.83", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"),
                                        ("486", "Respiratory"), ("V57", "Other"), ("E888", "Other"), ("996", "Injury"),
                                        (None, "Missing"), ("?", "Missing"), ("not-a-code", "Other")])
def test_icd9_groups(code, group):
    assert icd9_group(code) == group


def test_age_midpoint():
    assert age_midpoint("[0-10)") == 5 and age_midpoint("[70-80)") == 75 and np.isnan(age_midpoint("old"))


def test_feature_engineer_is_stateless_and_tolerates_missing_columns():
    fe = FeatureEngineer()
    out = fe.fit_transform(pd.DataFrame([{"age": "[60-70)", "gender": "Male", "time_in_hospital": 3}]))
    assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert out.loc[0, "age_mid"] == 65 and out.loc[0, "diag_1_group"] == "Missing" and out.loc[0, "insulin"] == "No"
    assert not hasattr(fe, "n_features_in_")      # nothing learned from data, so nothing can leak


@pytest.fixture(scope="module")
def split():
    if not config.SPLIT_CSV.exists():
        pytest.skip("run python -m src.data_pipeline first")
    return pd.read_csv(config.SPLIT_CSV)


def test_no_patient_in_two_parts(split):
    sets = {p: set(split.loc[split.split == p, "patient_nbr"]) for p in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])


def test_split_sizes_and_balance(split):
    share = split.split.value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.02 and abs(share["val"] - 0.15) < 0.02 and abs(share["test"] - 0.15) < 0.02
    rates = split.groupby("split").y.mean()
    assert rates.max() - rates.min() < 0.01        # stratified: same readmission rate in every part


def test_scaler_learned_from_training_rows_only(split):
    joblib = pytest.importorskip("joblib")
    if not config.MODEL_PATH.exists():
        pytest.skip("run python -m src.train first")
    from src.train import load_parts
    model = joblib.load(config.MODEL_PATH)
    scaler = model.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    parts = load_parts(("train", "test"))
    feats_tr = FeatureEngineer().transform(parts["train"])[NUMERIC_FEATURES]
    feats_all = FeatureEngineer().transform(pd.concat([parts["train"], parts["test"]]))[NUMERIC_FEATURES]
    assert np.allclose(scaler.mean_, feats_tr.mean().values, rtol=1e-9)
    assert not np.allclose(scaler.mean_, feats_all.mean().values, rtol=1e-9)
