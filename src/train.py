"""Model fitting, tuning on validation, capacity threshold, and export.

Run:  python -m src.train
Uses only the train and validation parts of data/processed/split.csv. The test part is not loaded here.
Writes artifacts/model.joblib, artifacts/metadata.json, artifacts/drift_reference.json, reports/validation.json.
"""
import hashlib
import itertools
import json
import platform
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config
from src.data_pipeline import load_raw
from src.features import FeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RAW_NUMERIC


def load_parts(parts=("train", "val")):
    raw = load_raw()
    s = pd.read_csv(config.SPLIT_CSV)
    assert set(parts) <= {"train", "val", "test"}
    df = raw.merge(s[["encounter_id", "y", "split"]], on="encounter_id", validate="one_to_one")
    return {p: df[df.split == p].reset_index(drop=True) for p in parts}


def make_pipeline(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                                     sparse_output=False))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("engineer", FeatureEngineer()), ("prep", prep), ("model", model)])


def capacity_threshold(scores, capacity=config.CAPACITY):
    """The score above which the top `capacity` share of encounters falls."""
    return float(np.quantile(scores, 1 - capacity))


def at_threshold(y, s, t):
    flag = s >= t
    tp = int((flag & (y == 1)).sum()); base = y.mean()
    prec = tp / max(1, flag.sum()); rec = tp / max(1, (y == 1).sum())
    return {"flagged_share": float(flag.mean()), "recall": rec, "precision": prec, "lift": prec / base,
            "f1_macro": float(f1_score(y, flag.astype(int), average="macro"))}


def score_report(y, s, t):
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            **at_threshold(y, s, t), "threshold": float(t)}


def heuristic_score(df):
    """Interpretable rule: the more inpatient stays in the past year, the higher the risk."""
    return df["number_inpatient"].astype(float).values


def heuristic_threshold(scores, capacity=config.CAPACITY):
    """Smallest whole number of prior stays whose flagged share does not exceed the capacity (ties are flagged together)."""
    for k in range(0, 50):
        if (scores >= k).mean() <= capacity:
            return float(k)
    return 50.0


def main():
    t_start = time.time()
    parts = load_parts(("train", "val"))
    tr, va = parts["train"], parts["val"]
    Xtr, ytr, Xva, yva = tr, tr.y.values, va, va.y.values
    results, tuning = {}, []

    # Baseline 0: prior only (every encounter gets the training readmission rate)
    dummy = make_pipeline(DummyClassifier(strategy="prior")).fit(Xtr, ytr)
    s = dummy.predict_proba(Xva)[:, 1]
    results["B0 prior (dummy)"] = {"pr_auc": float(average_precision_score(yva, s)), "roc_auc": 0.5,
                                   "note": "constant score: cannot rank, so no capacity list can be formed"}

    # Baseline 1: heuristic rule on prior inpatient stays
    k = heuristic_threshold(heuristic_score(tr))
    s = heuristic_score(va)
    results["B1 rule: prior inpatient stays"] = {**score_report(yva, s, k), "rule": f"flag if number_inpatient >= {int(k)}"}

    # Baseline 2: logistic regression, class-weighted loss, C tuned on validation
    best_lr = None
    for C in (0.01, 0.1, 1.0):
        m = make_pipeline(LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(Xtr, ytr)
        s = m.predict_proba(Xva)[:, 1]
        r = score_report(yva, s, capacity_threshold(s)); tuning.append({"model": "logistic", "C": C, **r})
        if best_lr is None or r["pr_auc"] > best_lr[1]["pr_auc"]:
            best_lr = (C, r, m)
    results["B2 logistic regression"] = {**best_lr[1], "params": {"C": best_lr[0], "class_weight": "balanced"}}

    # Main model: histogram gradient boosting, class-weighted loss, small grid on validation
    grid = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "min_samples_leaf": [50, 200], "max_iter": [200, 400]}
    best = None
    for lr_, leaves, msl, iters in itertools.product(*grid.values()):
        params = dict(learning_rate=lr_, max_leaf_nodes=leaves, min_samples_leaf=msl, max_iter=iters)
        t0 = time.time()
        m = make_pipeline(HistGradientBoostingClassifier(**params, l2_regularization=1.0, class_weight="balanced",
                                                         early_stopping=False, random_state=config.SEED)).fit(Xtr, ytr)
        s = m.predict_proba(Xva)[:, 1]
        r = score_report(yva, s, capacity_threshold(s)); r["fit_seconds"] = round(time.time() - t0, 1)
        tuning.append({"model": "hist_gradient_boosting", **params, **r})
        print(f"HGB {params} -> val PR-AUC {r['pr_auc']:.4f}, recall@20% {r['recall']:.3f}")
        if best is None or r["pr_auc"] > best[1]["pr_auc"]:
            best = (params, r, m)
    params, _, model = best

    # Ablation: the same settings without class weights (does the weighted loss help ranking?)
    m_unw = make_pipeline(HistGradientBoostingClassifier(**params, l2_regularization=1.0, early_stopping=False,
                                                         random_state=config.SEED)).fit(Xtr, ytr)
    s_unw = m_unw.predict_proba(Xva)[:, 1]
    results["M1 gradient boosting, unweighted (ablation)"] = {**score_report(yva, s_unw, capacity_threshold(s_unw)), "params": params}

    s_main = model.predict_proba(Xva)[:, 1]
    threshold = capacity_threshold(s_main)
    results["M2 gradient boosting, class-weighted (selected)"] = {**score_report(yva, s_main, threshold),
                                                                  "params": {**params, "l2_regularization": 1.0,
                                                                             "class_weight": "balanced"}}

    # Export (the baselines are saved too, so the one test evaluation can compare all models on equal terms)
    config.ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump({"model": best_lr[2], "threshold": best_lr[1]["threshold"]}, config.ARTIFACTS / "baseline_logistic.joblib")
    joblib.dump({"model": m_unw, "threshold": capacity_threshold(s_unw)}, config.ARTIFACTS / "ablation_unweighted.joblib")
    feats = model.named_steps["engineer"].transform(Xtr)
    train_ranges = {c: [float(tr[c].min()), float(tr[c].max())] for c in RAW_NUMERIC}
    meta = {
        "model_version": config.MODEL_VERSION, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file_sha256": hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest(),
        "estimator": "HistGradientBoostingClassifier", "params": results["M2 gradient boosting, class-weighted (selected)"]["params"],
        "capacity": config.CAPACITY, "threshold": threshold, "threshold_set_on": "validation (80th percentile of scores)",
        "fallback_rule": {"feature": "number_inpatient", "flag_if_at_least": k},
        "train_numeric_ranges": train_ranges, "excluded_dispositions": sorted(config.EXCLUDED_DISPOSITIONS),
        "n_train": int(len(tr)), "n_val": int(len(va)), "train_readmit_rate": float(ytr.mean()),
        "test_set_used": False, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "n_model_features": int(model.named_steps["prep"].transform(feats.head(5)).shape[1]),
    }
    config.METADATA_PATH.write_text(json.dumps(meta, indent=1))

    # Drift reference: training distributions of every engineered feature, plus validation score deciles
    def binned(values, edges):
        inner = np.asarray(edges)[1:-1]
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        return (counts / counts.sum()).tolist()
    score_edges = np.unique(np.quantile(s_main, np.linspace(0, 1, 11)))
    ref = {"numeric": {}, "categorical": {},
           "score": {"edges": score_edges.tolist(), "props": binned(s_main, score_edges)}}   # validation scores
    for c in NUMERIC_FEATURES:
        v = feats[c].dropna().values
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))      # deciles; collapse for count features
        ref["numeric"][c] = {"edges": edges.tolist(), "props": binned(v, edges),
                             "values_sample": np.random.default_rng(0).choice(v, 5000).tolist()}
    for c in CATEGORICAL_FEATURES:
        ref["categorical"][c] = feats[c].value_counts(normalize=True).to_dict()
    config.REFERENCE_PATH.write_text(json.dumps(ref))

    config.REPORTS.mkdir(exist_ok=True)
    out = {"validation": results, "tuning": tuning, "selected_params": params, "threshold": threshold,
           "fallback_threshold": k, "train_seconds_total": round(time.time() - t_start, 1)}
    (config.REPORTS / "validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k_: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in r.items()} for k_, r in results.items()}, indent=1))
    print(f"Saved {config.MODEL_PATH.name} (threshold {threshold:.4f}); test set not used.")


if __name__ == "__main__":
    main()
