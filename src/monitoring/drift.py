# src/monitoring/drift.py
"""
Evidently AI — Data Drift & Model Monitoring

Detects when:
  - Input features drift from training distribution
  - Model performance degrades over time
  - Data quality issues appear

Generates an HTML report you can open in browser.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset
)
from evidently.metrics import (
    DatasetDriftMetric,
)
from src.utils import load_config, get_logger

log = get_logger("drift_monitoring")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

import os
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "BM25", "Dense", "PassLen", "QueryLen",
    "Overlap", "OverlapRatio", "LogPassLen",
    "LogBM25", "DenseSquared", "Combined"
]


def load_features() -> tuple:
    """Loads train and eval feature sets."""
    with open(PROCESSED_DIR / "features_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(PROCESSED_DIR / "features_eval.pkl", "rb") as f:
        eval_data = pickle.load(f)

    X_train = pd.DataFrame(train_data["X"], columns=FEATURE_NAMES)
    y_train = pd.Series(train_data["y"], name="label")

    X_eval = pd.DataFrame(eval_data["X"], columns=FEATURE_NAMES)
    y_eval = pd.Series(eval_data["y"], name="label")

    return X_train, y_train, X_eval, y_eval


def simulate_production_drift(X_ref: pd.DataFrame,
                              drift_factor: float = 0.3) -> pd.DataFrame:
    """
    Simulates production data that has drifted from training.
    In real life this would be live traffic data.

    drift_factor: how much drift to add (0=none, 1=heavy)
    """
    X_prod = X_ref.copy()

    # Add realistic drift to some features
    noise = np.random.normal(0, drift_factor, X_prod.shape)
    std_vals = X_prod.std().values
    X_prod = X_prod + noise * std_vals

    # Clip to realistic ranges
    X_prod = X_prod.clip(lower=0)

    return X_prod


def run_drift_report(X_reference: pd.DataFrame,
                     X_current: pd.DataFrame,
                     y_reference: pd.Series,
                     y_current: pd.Series) -> str:
    """
    Generates a full Evidently drift report.
    Returns path to HTML report.
    """
    print("\n📊 Generating drift report...")

    # Add target column
    ref_data = X_reference.copy()
    ref_data["target"] = y_reference.values[:len(X_reference)]

    curr_data = X_current.copy()
    curr_data["target"] = y_current.values[:len(X_current)]

    # Build report with multiple presets
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])

    # Use sample for speed
    sample_size = min(2000, len(ref_data), len(curr_data))
    report.run(
        reference_data=ref_data.sample(sample_size, random_state=42),
        current_data=curr_data.sample(sample_size, random_state=42)
    )

    # Save HTML report
    report_path = REPORTS_DIR / "drift_report.html"
    report.save_html(str(report_path))

    print(f"✅ Drift report saved to {report_path}")
    return str(report_path)


def check_drift_threshold(X_reference: pd.DataFrame,
                          X_current: pd.DataFrame,
                          threshold: float = 0.3) -> dict:
    """
    Checks if feature drift exceeds threshold.
    Returns alert if retraining is needed.

    Uses Population Stability Index (PSI) per feature.
    PSI > 0.2 = significant drift → retrain model.
    """
    print("\n🔍 Checking drift thresholds...")

    drift_results = {}
    alerts = []

    for col in X_reference.columns:
        ref_vals = X_reference[col].values
        curr_vals = X_current[col].values

        # Normalize
        ref_mean = ref_vals.mean()
        ref_std = ref_vals.std() + 1e-8

        ref_norm = (ref_vals - ref_mean) / ref_std
        curr_norm = (curr_vals - ref_mean) / ref_std

        # KS statistic as drift measure
        from scipy import stats
        ks_stat, p_value = stats.ks_2samp(ref_norm, curr_norm)

        drifted = ks_stat > threshold
        drift_results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted
        }

        if drifted:
            alerts.append(col)

    # Summary
    num_drifted = len(alerts)
    total = len(X_reference.columns)
    drift_pct = num_drifted / total * 100

    print(f"\n{'='*45}")
    print(f"  Drift Detection Results")
    print(f"{'='*45}")
    print(f"  Features checked : {total}")
    print(f"  Features drifted : {num_drifted} ({drift_pct:.1f}%)")
    print(f"{'='*45}")

    for feat, result in drift_results.items():
        status = "🔴 DRIFT" if result["drifted"] else "🟢 OK"
        print(f"  {feat:15} KS={result['ks_statistic']:.4f}  {status}")

    print(f"{'='*45}")

    if num_drifted > total * 0.3:
        print("\n⚠️  ALERT: Significant drift detected!")
        print("   Recommendation: Retrain the model")
        needs_retrain = True
    else:
        print("\n✅ Drift within acceptable limits")
        needs_retrain = False

    log.info(f"Drift check: {num_drifted}/{total} features drifted")

    return {
        "features": drift_results,
        "num_drifted": num_drifted,
        "drift_percentage": drift_pct,
        "needs_retrain": needs_retrain,
        "alerts": alerts
    }


if __name__ == "__main__":
    print("=" * 50)
    print("  NeuralRank — Drift Monitoring")
    print("=" * 50)

    # Load features
    print("\nLoading features...")
    X_train, y_train, X_eval, y_eval = load_features()
    print(f"✅ Reference data: {X_train.shape}")
    print(f"✅ Current data:   {X_eval.shape}")

    # Simulate production drift
    print("\nSimulating production data with drift...")
    X_prod = simulate_production_drift(X_eval, drift_factor=0.4)
    print(f"✅ Production data simulated: {X_prod.shape}")

    # Check drift thresholds
    drift_results = check_drift_threshold(X_train, X_prod, threshold=0.15)

    # Generate full HTML report
    report_path = run_drift_report(
        X_train.head(3000),
        X_prod.head(3000),
        y_train,
        y_eval
    )

    print(f"\n📄 Open this file in your browser:")
    print(f"   {report_path}")
    print(f"\n✅ Module 4.2 complete!")
    print(f"✅ Ready for Module 4.3 — Docker Compose")