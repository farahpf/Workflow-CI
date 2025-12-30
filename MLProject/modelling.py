# modelling.py (CI / MLflow Project)
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path ke folder preprocessing (X_train.npz, X_test.npz, y_train.csv, y_test.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # ===== Validasi file =====
    required_files = [
        "X_train.npz",
        "X_test.npz",
        "y_train.csv",
        "y_test.csv",
    ]

    for f in required_files:
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"File tidak ditemukan: {data_dir / f}")

    # ===== Load data =====
    X_train = sparse.load_npz(data_dir / "X_train.npz")
    X_test = sparse.load_npz(data_dir / "X_test.npz")
    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0].values
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0].values

    # ===== MLflow Project handle run otomatis =====
    mlflow.sklearn.autolog(log_models=True)

    # ===== Model =====
    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ===== Evaluasi =====
    y_pred = model.predict(X_test)

    auc = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ===== Manual log tambahan (boleh walau autolog aktif) =====
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)
    mlflow.log_metric("test_f1", f1)
    if auc is not None:
        mlflow.log_metric("test_roc_auc", auc)

    print("✅ CI Training selesai")
    print("Metrics:", {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    })


if __name__ == "__main__":
    main()
