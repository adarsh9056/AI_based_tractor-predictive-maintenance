#!/usr/bin/env python3
"""Train RF + XGBoost on AI4I-style data; persist best model and artifacts to backend/."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
MODELS = ROOT / "models"
DATA_CSV = ROOT / "data" / "ai4i2020.csv"

sys.path.insert(0, str(BACKEND))
from feature_engineering import FEATURE_LIST, add_engineered_columns  # noqa: E402

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # noqa: BLE001
    XGBClassifier = None  # type: ignore[misc, assignment]
    _HAS_XGB = False


def ensure_data() -> None:
    if DATA_CSV.exists():
        return
    print("ai4i2020.csv not found; generating synthetic bootstrap data...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bootstrap_data

    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_data.generate(10_000).to_csv(DATA_CSV, index=False)


def failure_type(row: pd.Series) -> str:
    if int(row.get("HDF", 0)) == 1:
        return "Heat Dissipation Failure"
    if int(row.get("PWF", 0)) == 1:
        return "Power Failure"
    if int(row.get("OSF", 0)) == 1:
        return "Overstrain Failure"
    if int(row.get("TWF", 0)) == 1:
        return "Tool Wear Failure"
    if int(row.get("RNF", 0)) == 1:
        return "Random Failure"
    return "Healthy"


def train() -> None:
    ensure_data()
    df = pd.read_csv(DATA_CSV)
    df.columns = [c.strip() for c in df.columns]
    for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if c not in df.columns:
            df[c] = 0
    df = add_engineered_columns(df)
    df["failure_type"] = df.apply(failure_type, axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df["failure_type"])
    X = df[FEATURE_LIST].values.astype(np.float64)

    counts = Counter(y)
    strat = y if all(counts[i] >= 2 for i in counts) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Cross-validation on original (imbalanced) training fold — honest macro-F1 story
    min_per_class = min(Counter(y_train).values())
    n_splits = min(5, min_per_class)
    if n_splits < 2:
        cv_scores = {"note": "Too few samples per class for stratified CV; skipped.", "skipped": True}
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        rf_cv = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        scores = cross_val_score(rf_cv, X_train_s, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        cv_scores = {
            "cv_f1_macro_mean": float(np.mean(scores)),
            "cv_f1_macro_std": float(np.std(scores)),
            "n_splits": int(n_splits),
        }

    k = min(5, min(Counter(y_train).values()) - 1)
    if k >= 1:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_tr, y_tr = smote.fit_resample(X_train_s, y_train)
    else:
        X_tr, y_tr = X_train_s, y_train

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_test_s)

    def score(name: str, pred: np.ndarray) -> dict:
        return {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        }

    metrics = [score("random_forest", rf_pred)]
    xgb = None
    xgb_pred = rf_pred
    if _HAS_XGB and XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb.fit(X_tr, y_tr)
        xgb_pred = xgb.predict(X_test_s)
        metrics.append(score("xgboost", xgb_pred))
    else:
        print("XGBoost not available (install libomp on macOS); using Random Forest only.")

    # Prefer higher macro-F1 (better for imbalance) over accuracy alone
    best = rf
    best_name = "random_forest"
    best_pred = rf_pred
    if xgb is not None and len(metrics) > 1 and metrics[1]["f1_macro"] >= metrics[0]["f1_macro"]:
        best = xgb
        best_name = "xgboost"
        best_pred = xgb_pred

    cm = confusion_matrix(y_test, best_pred)
    cm_labels = [str(c) for c in le.classes_]
    per_class = classification_report(
        y_test, best_pred, target_names=cm_labels, zero_division=0, output_dict=True
    )

    report_txt = classification_report(y_test, best_pred, target_names=cm_labels, zero_division=0)

    payload = {
        "compare": metrics,
        "chosen": best_name,
        "imbalance_note": (
            "Accuracy can look high when the majority class is Healthy; rely on macro-F1, "
            "per-class recall, confusion matrix, and CV on the held-out training split."
        ),
        "cross_validation_train_fold": cv_scores,
        "confusion_matrix": {"labels": cm_labels, "matrix": cm.tolist()},
        "per_class_report": per_class,
        "report_text": report_txt,
    }

    BACKEND.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, BACKEND / "model.pkl")
    joblib.dump(scaler, BACKEND / "scaler.pkl")
    joblib.dump(le, BACKEND / "label_encoder.pkl")
    joblib.dump(FEATURE_LIST, BACKEND / "feature_list.pkl")
    joblib.dump(rf, MODELS / "random_forest.pkl")
    if xgb is not None:
        joblib.dump(xgb, MODELS / "xgboost.pkl")

    (MODELS / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Chosen:", best_name)
    print(report_txt)


if __name__ == "__main__":
    train()
