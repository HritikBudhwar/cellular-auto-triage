"""
nettriage — ML auto-triage for cellular (LTE/5G) network measurement logs.

Pipeline
  1. Parse raw per-second measurement logs (G-NetTrack style CSV traces).
  2. Engineer rolling-window time-series features per trace (radio KPIs + handovers).
  3. Supervised: predict an upcoming throughput drop (failure prediction), RandomForest + GridSearchCV.
  4. Regression: predict near-future downlink throughput from radio metrics.
  5. Unsupervised: IsolationForest anomaly detection on radio KPIs.
  6. Categorize detected failures (KMeans) into coverage / interference / congestion / mobility.
  7. Auto-generate a Markdown report with Matplotlib charts + metrics.json.

Works on the public UCC 5G dataset (github.com/uccmisl/5Gdataset) or any CSV with the same columns.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

RADIO = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
NUMERIC = ["Speed", *RADIO, "DL_bitrate", "UL_bitrate"]
WINDOW = 5      # seconds of history used for features
HORIZON = 5     # seconds ahead we predict
BASELINE = 10   # seconds used for the "normal" throughput baseline
DROP_RATIO = 0.5
MIN_BASELINE_KBPS = 200


# ----------------------------------------------------------------------------- 1. parse
def load_traces(root: str) -> pd.DataFrame:
    """Read every CSV trace under `root` into one DataFrame with trace metadata."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    frames = []
    for i, path in enumerate(files):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not {"RSRP", "SNR", "DL_bitrate"}.issubset(df.columns):
            continue
        parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
        df["trace_id"] = i
        df["mobility"] = "Driving" if "driving" in parts else ("Static" if "static" in parts else "Unknown")
        df["source_file"] = os.path.relpath(path, root)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No usable CSV traces found under {root!r}")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce '-' placeholders to NaN, fill short gaps within each trace, drop unusable rows."""
    df = df.copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
        else:
            df[c] = np.nan
    df["row_in_trace"] = df.groupby("trace_id").cumcount()
    g = df.groupby("trace_id")
    for c in ["Speed", "RSRQ", "CQI", "RSSI", "UL_bitrate"]:
        df[c] = g[c].transform(lambda s: s.ffill(limit=3).bfill(limit=3))
    if "State" in df:  # keep only samples where the device was actively downloading, if labelled
        active = df["State"].astype(str).str.upper().eq("D")
        if active.mean() > 0.2:
            df = df[active]
    df = df.dropna(subset=["RSRP", "SNR", "DL_bitrate"])
    for c in ["Speed", "RSRQ", "CQI", "RSSI", "UL_bitrate"]:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    return df.sort_values(["trace_id", "row_in_trace"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- 2. features + labels
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-window time-series features and the future-throughput labels."""
    out = []
    for _, t in df.groupby("trace_id", sort=False):
        t = t.copy()
        for c in ["RSRP", "RSRQ", "SNR", "CQI"]:
            r = t[c].rolling(WINDOW, min_periods=WINDOW)
            t[f"{c}_mean"] = r.mean()
            t[f"{c}_std"] = r.std()
            t[f"{c}_min"] = r.min()
            t[f"{c}_slope"] = t[c].diff(WINDOW - 1) / (WINDOW - 1)
        t["dl_mean"] = t["DL_bitrate"].rolling(WINDOW, min_periods=WINDOW).mean()
        t["dl_baseline"] = t["DL_bitrate"].rolling(BASELINE, min_periods=BASELINE).median()
        if "CellID" in t:
            changed = t["CellID"].astype(str).ne(t["CellID"].astype(str).shift()).astype(int)
            changed.iloc[0] = 0
            t["handovers_recent"] = changed.rolling(WINDOW, min_periods=1).sum()
        else:
            t["handovers_recent"] = 0
        fut = t["DL_bitrate"][::-1].rolling(HORIZON, min_periods=HORIZON).mean()[::-1].shift(-1)
        t["dl_future"] = fut
        out.append(t)
    f = pd.concat(out)
    f = f.dropna(subset=["RSRP_std", "dl_baseline", "dl_future"])
    f = f[f["dl_baseline"] >= MIN_BASELINE_KBPS]
    f["drop_next"] = (f["dl_future"] < DROP_RATIO * f["dl_baseline"]).astype(int)
    return f.reset_index(drop=True)


def feature_columns(f: pd.DataFrame) -> list[str]:
    cols = [c for c in f.columns if c.endswith(("_mean", "_std", "_min", "_slope"))]
    return cols + ["RSRP", "RSRQ", "SNR", "CQI", "Speed", "handovers_recent"]


# ----------------------------------------------------------------------------- 3-6. models
def train_failure_classifier(f, cols, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

    X, y, groups = f[cols].values, f["drop_next"].values, f["trace_id"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        {"n_estimators": [200, 300], "max_depth": [10, 20], "min_samples_leaf": [1, 5]},
        scoring="f1", cv=GroupKFold(n_splits=min(3, len(np.unique(groups[tr])))), n_jobs=-1,
    )
    grid.fit(X[tr], y[tr], groups=groups[tr])
    model = grid.best_estimator_
    proba = model.predict_proba(X[te])[:, 1]
    pred = (proba >= 0.5).astype(int)

    # naive rule baseline: "signal is weak right now"
    rule = ((f["RSRP"].values[te] < -105) | (f["SNR"].values[te] < 0)).astype(int)
    metrics = {
        "test_rows": int(len(te)), "test_traces": int(len(np.unique(groups[te]))),
        "positive_rate": float(y[te].mean()),
        "best_params": grid.best_params_,
        "f1": float(f1_score(y[te], pred)), "precision": float(precision_score(y[te], pred, zero_division=0)),
        "recall": float(recall_score(y[te], pred)), "roc_auc": float(roc_auc_score(y[te], proba)),
        "pr_auc": float(average_precision_score(y[te], proba)),
        "rule_baseline_f1": float(f1_score(y[te], rule)),
        "confusion_matrix": confusion_matrix(y[te], pred).tolist(),
        "report": classification_report(y[te], pred, output_dict=True, zero_division=0),
    }
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    return model, metrics, importances, (te, y[te], proba)


def train_throughput_regressor(f, cols, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit

    X, y, groups = f[cols].values, np.log1p(f["dl_future"].values), f["trace_id"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed).fit(X[tr], y[tr])
    pred_kbps, true_kbps = np.expm1(reg.predict(X[te])), np.expm1(y[te])
    naive = f["dl_mean"].values[te]  # "throughput stays the same" baseline
    return reg, {
        "mae_mbps": float(mean_absolute_error(true_kbps, pred_kbps) / 1000),
        "naive_mae_mbps": float(mean_absolute_error(true_kbps, naive) / 1000),
        "r2_log": float(r2_score(y[te], reg.predict(X[te]))),
    }


def detect_anomalies(f, contamination=0.05, seed=42):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    cols = ["RSRP", "RSRQ", "SNR", "CQI", "RSRP_std", "SNR_std", "RSRP_slope", "SNR_slope", "handovers_recent"]
    Z = StandardScaler().fit_transform(f[cols].values)
    iso = IsolationForest(contamination=contamination, random_state=seed).fit(Z)
    flag = iso.predict(Z) == -1
    base = f["drop_next"].mean()
    hit = f.loc[flag, "drop_next"].mean() if flag.any() else 0.0
    return flag, {
        "contamination": contamination, "flagged_rows": int(flag.sum()),
        "drop_rate_in_flagged": float(hit), "drop_rate_overall": float(base),
        "lift": float(hit / base) if base else None,
        "recall_of_drops": float(f.loc[flag, "drop_next"].sum() / max(f["drop_next"].sum(), 1)),
    }


def categorize_failures(f, k=4, seed=42):
    """Cluster drop events and name each cluster from its centroid (relative to normal samples)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    cols = ["RSRP_mean", "SNR_mean", "CQI_mean", "Speed", "handovers_recent"]
    ev = f[f["drop_next"] == 1].copy()
    if len(ev) < k:
        ev["category"] = "Uncategorized"
        return ev
    sc = StandardScaler().fit(f[cols].values)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sc.transform(ev[cols].values))
    cent = pd.DataFrame(km.cluster_centers_, columns=cols)
    names = {}
    for i, c in cent.iterrows():
        if c["handovers_recent"] > 0.75 or c["Speed"] > 1.0:
            names[i] = "Mobility / handover"
        elif c["RSRP_mean"] < -0.5:
            names[i] = "Weak coverage"
        elif c["SNR_mean"] < -0.5:
            names[i] = "Interference (low SNR)"
        else:
            names[i] = "Congestion / scheduling"
    ev["category"] = [names[c] for c in km.labels_]
    return ev


# ----------------------------------------------------------------------------- 7. report
def write_report(out_dir, f, cls_metrics, importances, eval_data, reg_metrics, anom_flag, anom_metrics, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ax = importances.head(12)[::-1].plot.barh(figsize=(7, 4.5), color="#3b6ea5")
    ax.set_title("Top features for predicting a throughput drop"); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=150); plt.close()

    _, y_true, proba = eval_data
    p, r, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(5, 4)); plt.plot(r, p, color="#3b6ea5")
    plt.axhline(y_true.mean(), ls="--", color="grey", label="base rate")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Failure prediction — PR curve"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "pr_curve.png"), dpi=150); plt.close()

    tid = f["trace_id"].value_counts().index[0]
    t = f[f["trace_id"] == tid]
    m = anom_flag[f["trace_id"].values == tid]
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    axs[0].plot(t["row_in_trace"], t["DL_bitrate"] / 1000, lw=0.8); axs[0].set_ylabel("DL Mbps")
    axs[1].plot(t["row_in_trace"], t["RSRP"], lw=0.8); axs[1].set_ylabel("RSRP dBm")
    axs[2].plot(t["row_in_trace"], t["SNR"], lw=0.8); axs[2].set_ylabel("SNR dB")
    for a in axs:
        a.scatter(t["row_in_trace"][m], a.lines[0].get_ydata()[m], color="crimson", s=8, zorder=3)
    axs[0].set_title(f"Trace {tid}: anomalies flagged by IsolationForest (red)"); axs[2].set_xlabel("seconds")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "anomaly_timeline.png"), dpi=150); plt.close()

    counts = events["category"].value_counts()
    ax = counts.plot.bar(figsize=(6, 4), color="#c0504d", rot=15)
    ax.set_title("Throughput-drop events by failure category"); ax.set_ylabel("events")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "failure_categories.png"), dpi=150); plt.close()

    summary = {
        "dataset": {"rows": int(len(f)), "traces": int(f["trace_id"].nunique()),
                    "drop_events": int(f["drop_next"].sum())},
        "failure_prediction": {k: v for k, v in cls_metrics.items() if k != "report"},
        "throughput_regression": reg_metrics,
        "anomaly_detection": anom_metrics,
        "failure_categories": counts.to_dict(),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    c, rm, am = cls_metrics, reg_metrics, anom_metrics
    md = f"""# Cellular Network Auto-Triage Report

**Data:** {summary['dataset']['rows']:,} one-second samples from {summary['dataset']['traces']} traces,
{summary['dataset']['drop_events']:,} upcoming throughput-drop events (≥{int((1-DROP_RATIO)*100)}% fall within {HORIZON}s).

## 1. Failure prediction (supervised, RandomForest + GridSearchCV)
| Metric | Model | Rule baseline (RSRP<-105 or SNR<0) |
|---|---|---|
| F1 | {c['f1']:.3f} | {c['rule_baseline_f1']:.3f} |
| Precision | {c['precision']:.3f} | |
| Recall | {c['recall']:.3f} | |
| ROC-AUC | {c['roc_auc']:.3f} | |
| PR-AUC | {c['pr_auc']:.3f} | base rate {c['positive_rate']:.3f} |

Held-out: {c['test_rows']:,} samples from {c['test_traces']} unseen traces. Best params: `{c['best_params']}`

![](figures/pr_curve.png)
![](figures/feature_importance.png)

## 2. Throughput forecasting (regression)
MAE **{rm['mae_mbps']:.2f} Mbps** vs {rm['naive_mae_mbps']:.2f} Mbps for a "no change" baseline (R² on log scale {rm['r2_log']:.3f}).

## 3. Anomaly detection (unsupervised, IsolationForest)
Flagged {am['flagged_rows']:,} samples ({am['contamination']:.0%}); drop rate among flagged = {am['drop_rate_in_flagged']:.3f}
vs {am['drop_rate_overall']:.3f} overall → **{(am['lift'] or 0):.1f}× lift**, catching {am['recall_of_drops']:.1%} of drops.

![](figures/anomaly_timeline.png)

## 4. Failure categorization (KMeans on drop events)
{chr(10).join(f'- **{k}**: {v}' for k, v in counts.items())}

![](figures/failure_categories.png)
"""
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(md)
    return summary


def run(data_root: str, out_dir: str = "output") -> dict:
    raw = load_traces(data_root)
    df = clean(raw)
    f = build_features(df)
    cols = feature_columns(f)
    _, cls_m, imp, ev = train_failure_classifier(f, cols)
    _, reg_m = train_throughput_regressor(f, cols)
    flag, an_m = detect_anomalies(f)
    events = categorize_failures(f)
    return write_report(out_dir, f, cls_m, imp, ev, reg_m, flag, an_m, events)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Auto-triage cellular network measurement logs")
    ap.add_argument("data_root", help="folder containing CSV traces (searched recursively)")
    ap.add_argument("--out", default="output")
    s = run(ap.parse_args().data_root, ap.parse_args().out)
    print(json.dumps(s, indent=2, default=str))
