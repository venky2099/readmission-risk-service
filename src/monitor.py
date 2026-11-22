"""Drift monitoring: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) against the training reference.

Run:  python -m src.monitor
Compares two incoming batches with the training distributions saved by src.train:
  1. "test month": the held-out test encounters, standing in for a normal month of new discharges;
  2. "new partner hospital": a simulated shift, older and mostly emergency admissions, built by resampling test rows.
Alert rules (applied per batch):
  feature PSI >= 0.25 -> alert, 0.10 to 0.25 -> watch; score PSI >= 0.25 -> alert;
  flagged share outside 15% to 25% (capacity 20%) -> alert, because the follow-up list would no longer fit the team.
Writes reports/drift_report.json.
"""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src import config
from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.train import load_parts

EPS = 1e-4
PSI_WATCH, PSI_ALERT = 0.10, 0.25
SHARE_BAND = (0.15, 0.25)


def psi(expected, actual):
    e = np.clip(np.asarray(expected, float), EPS, None); a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def binned_psi(ref_bins, values):
    """PSI over the training bins: ref_bins holds the bin edges and the training share in each bin."""
    inner = np.asarray(ref_bins["edges"], float)[1:-1]
    act = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
    return psi(ref_bins["props"], act)


def categorical_psi(ref_props, values):
    vc = pd.Series(values).value_counts(normalize=True)
    cats = sorted(set(ref_props) | set(vc.index))
    return psi([ref_props.get(c, 0) for c in cats], [vc.get(c, 0) for c in cats])


def check_batch(name, df, model, ref, meta):
    feats = model.named_steps["engineer"].transform(df)
    rows = []
    for c in NUMERIC_FEATURES:
        v = feats[c].dropna().values
        rows.append({"feature": c, "psi": binned_psi(ref["numeric"][c], v),
                     "ks_p": float(ks_2samp(ref["numeric"][c]["values_sample"], v, method="asymp").pvalue)})
    for c in CATEGORICAL_FEATURES:
        rows.append({"feature": c, "psi": categorical_psi(ref["categorical"][c], feats[c].values), "ks_p": None})
    t = pd.DataFrame(rows)
    t["status"] = np.where(t.psi >= PSI_ALERT, "alert", np.where(t.psi >= PSI_WATCH, "watch", "stable"))
    scores = model.predict_proba(df)[:, 1]
    score_psi = binned_psi(ref["score"], scores)
    share = float((scores >= meta["threshold"]).mean())
    alerts = [f"feature drift: {r.feature} (PSI {r.psi:.2f})" for r in t.itertuples() if r.status == "alert"]
    if score_psi >= PSI_ALERT:
        alerts.append(f"score drift (PSI {score_psi:.2f})")
    if not SHARE_BAND[0] <= share <= SHARE_BAND[1]:
        alerts.append(f"flagged share {share:.1%} outside {SHARE_BAND[0]:.0%} to {SHARE_BAND[1]:.0%}")
    return {"batch": name, "encounters": int(len(df)), "score_psi": score_psi, "flagged_share": share,
            "n_alert": int((t.status == "alert").sum()), "n_watch": int((t.status == "watch").sum()),
            "top_features": t.sort_values("psi", ascending=False).head(6).round(4).to_dict("records"),
            "alerts": alerts, "decision": "investigate and consider retraining" if alerts else "no action"}


def simulated_partner_hospital(te, n=5000, seed=config.SEED):
    """Older patients, mostly admitted through the emergency room (admission_source_id 7)."""
    rng = np.random.default_rng(seed)
    w = np.where(te.age.isin(["[70-80)", "[80-90)", "[90-100)"]), 4.0, 1.0) * np.where(te.admission_source_id == 7, 4.0, 1.0)
    idx = rng.choice(len(te), size=n, replace=True, p=w / w.sum())
    return te.iloc[idx].reset_index(drop=True)


def main():
    meta = json.loads(config.METADATA_PATH.read_text())
    ref = json.loads(config.REFERENCE_PATH.read_text())
    model = joblib.load(config.MODEL_PATH)
    te = load_parts(("test",))["test"]
    batches = [check_batch("test month (no shift expected)", te, model, ref, meta),
               check_batch("new partner hospital (simulated shift)", simulated_partner_hospital(te), model, ref, meta)]
    out = {"rules": {"psi_watch": PSI_WATCH, "psi_alert": PSI_ALERT, "flagged_share_band": SHARE_BAND}, "batches": batches}
    (config.REPORTS / "drift_report.json").write_text(json.dumps(out, indent=1))
    for b in batches:
        print(f"{b['batch']}: score PSI {b['score_psi']:.3f}, flagged {b['flagged_share']:.1%}, alerts: {b['alerts'] or 'none'}")


if __name__ == "__main__":
    main()
