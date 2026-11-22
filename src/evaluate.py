"""The single test evaluation, plus validation-only interpretability.

Run:  python -m src.evaluate      (after src.train)
Every model is scored on the test part at the threshold fixed on validation. Nothing is re-tuned here.
Writes reports/test_metrics.json and figures in reports/figures/.
"""
import json
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, brier_score_loss
from src import config
from src.train import load_parts, score_report, heuristic_score, at_threshold


def grouped_bootstrap(y, s, t, groups, B=1000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); idx_of = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    rows = []
    for _ in range(B):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug), replace=True)])
        yb, sb = y[idx], s[idx]
        r = at_threshold(yb, sb, t)
        rows.append({"pr_auc": average_precision_score(yb, sb), "roc_auc": roc_auc_score(yb, sb),
                     "recall": r["recall"], "precision": r["precision"], "lift": r["lift"]})
    b = pd.DataFrame(rows)
    return {m: [float(b[m].quantile(.025)), float(b[m].quantile(.975))] for m in b.columns}


def capacity_curve(y, s, shares):
    order = np.argsort(-s); ys = y[order]; out = []
    for q in shares:
        n = int(round(q * len(y))); out.append(ys[:n].sum() / ys.sum())
    return np.array(out)


def main():
    meta = json.loads(config.METADATA_PATH.read_text())
    if meta["test_set_used"] and "--again" not in sys.argv:
        raise SystemExit("The test set was already used for this model. Re-run training first, or pass --again "
                         "to repeat the identical evaluation (nothing is re-tuned either way).")
    parts = load_parts(("val", "test"))
    va, te = parts["val"], parts["test"]
    yte, groups = te.y.values, te.patient_nbr.values
    main_model = joblib.load(config.MODEL_PATH)
    lr = joblib.load(config.ARTIFACTS / "baseline_logistic.joblib")
    unw = joblib.load(config.ARTIFACTS / "ablation_unweighted.joblib")

    scores = {"B1 rule: prior inpatient stays": (heuristic_score(te), meta["fallback_rule"]["flag_if_at_least"]),
              "B2 logistic regression": (lr["model"].predict_proba(te)[:, 1], lr["threshold"]),
              "M1 gradient boosting, unweighted (ablation)": (unw["model"].predict_proba(te)[:, 1], unw["threshold"]),
              "M2 gradient boosting, class-weighted (selected)": (main_model.predict_proba(te)[:, 1], meta["threshold"])}
    table = {name: score_report(yte, s, t) for name, (s, t) in scores.items()}
    table["B0 prior (dummy)"] = {"pr_auc": float(yte.mean()), "roc_auc": 0.5, "note": "cannot rank"}
    s_main, t_main = scores["M2 gradient boosting, class-weighted (selected)"]
    ci = grouped_bootstrap(yte, s_main, t_main, groups)

    # Calibration: the weighted model's scores are not probabilities; the unweighted model's are closer to them.
    s_unw = scores["M1 gradient boosting, unweighted (ablation)"][0]
    brier = {"M2 class-weighted": float(brier_score_loss(yte, s_main)), "M1 unweighted": float(brier_score_loss(yte, s_unw))}

    # Who gets flagged: share flagged and recall by age band and sex (test, selected model, fixed threshold)
    flag = s_main >= t_main
    te2 = te.assign(flag=flag, age_band=pd.cut(te.age.str.extract(r"\[(\d+)")[0].astype(int), [0, 50, 70, 80, 100], right=False,
                                                labels=["under 50", "50 to 69", "70 to 79", "80 and over"]))
    sub = []
    for col in ["age_band", "gender"]:
        for g, d in te2.groupby(col, observed=True):
            if len(d) < 50:
                continue
            pos = d.y == 1
            sub.append({"group": f"{col}: {g}", "encounters": int(len(d)), "readmit_rate": float(pos.mean()),
                        "flagged_share": float(d.flag.mean()), "recall": float((d.flag & pos).sum() / max(1, pos.sum()))})

    # Interpretability on VALIDATION (not test): permutation importance of raw inputs, measured as the drop in PR-AUC
    raw_cols = ["number_inpatient", "number_emergency", "number_outpatient", "discharge_disposition_id", "admission_source_id",
                "time_in_hospital", "num_medications", "num_lab_procedures", "number_diagnoses", "age", "diag_1", "diag_2",
                "diag_3", "medical_specialty", "insulin", "A1Cresult", "admission_type_id", "race", "gender"]
    rng = np.random.default_rng(config.SEED)
    base_ap = average_precision_score(va.y, main_model.predict_proba(va)[:, 1]); imp = []
    for c in raw_cols:
        drops = []
        for _ in range(3):
            v = va.copy(); v[c] = rng.permutation(v[c].values)
            drops.append(base_ap - average_precision_score(va.y, main_model.predict_proba(v)[:, 1]))
        imp.append({"input": c, "pr_auc_drop": float(np.mean(drops))})
    imp = sorted(imp, key=lambda r: -r["pr_auc_drop"])

    # Figures
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    shares = np.linspace(0.01, 1, 100)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, (s, _) in scores.items():
        if name.startswith("M1"):
            continue
        lab = name.split(" (")[0]
        ax[0].plot(shares * 100, capacity_curve(yte, s, shares) * 100, label=lab)
        p, r, _ = precision_recall_curve(yte, s); ax[1].plot(r, p, label=f"{lab} (PR-AUC {average_precision_score(yte, s):.3f})")
    ax[0].plot([0, 100], [0, 100], "k--", lw=.8, label="random calling")
    ax[0].axvline(config.CAPACITY * 100, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("share of discharges called (%)"); ax[0].set_ylabel("share of readmissions reached (%)")
    ax[0].set_title("Capacity curve (test)"); ax[0].legend(fontsize=7)
    ax[1].axhline(yte.mean(), color="k", ls="--", lw=.8); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-recall (test)"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(config.FIGURES / "test_capacity_pr.png", dpi=160); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for lab, s in [("M2 class-weighted", s_main), ("M1 unweighted", s_unw)]:
        fp, mp = calibration_curve(yte, s, n_bins=10, strategy="quantile"); ax[0].plot(mp, fp, marker="o", label=lab)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("mean score"); ax[0].set_ylabel("observed readmission rate")
    ax[0].set_title("Calibration (test)"); ax[0].legend(fontsize=8)
    top = imp[:10][::-1]
    ax[1].barh([r["input"] for r in top], [r["pr_auc_drop"] for r in top]); ax[1].set_xlabel("drop in PR-AUC when shuffled")
    ax[1].set_title("Most influential inputs (validation)")
    plt.tight_layout(); plt.savefig(config.FIGURES / "calibration_importance.png", dpi=160); plt.close()

    out = {"test_rows": int(len(te)), "test_patients": int(te.patient_nbr.nunique()), "test_readmit_rate": float(yte.mean()),
           "table": table, "selected_ci95": ci, "brier": brier, "subgroups": sub, "importance_validation": imp,
           "flagged_share_on_test": float(flag.mean())}
    (config.REPORTS / "test_metrics.json").write_text(json.dumps(out, indent=1))
    meta["test_set_used"] = True
    config.METADATA_PATH.write_text(json.dumps(meta, indent=1))
    show = pd.DataFrame(table).T[["pr_auc", "roc_auc", "flagged_share", "recall", "precision", "lift"]].astype(float).round(3)
    print(show.to_string()); print("selected model 95% CI:", {k: [round(a, 3), round(b, 3)] for k, (a, b) in ci.items()})


if __name__ == "__main__":
    main()
