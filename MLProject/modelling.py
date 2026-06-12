import argparse
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TARGET_COLUMN = "Churn"


def load_dataset(data_file: str):
    df = pd.read_csv(data_file)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def train_model(data_file: str, n_estimators: int, max_depth: int, min_samples_split: int):
    # Paksa MLflow memakai tracking folder relatif agar aman di Windows dan GitHub Actions.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Telco Churn CI Training")

    X, y = load_dataset(data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight="balanced"
    )
# Gunakan run yang sudah dibuat MLflow Project
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("data_file", data_file)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / "random_forest_telco_churn.joblib"
        joblib.dump(model, model_path)

        report_path = artifacts_dir / "classification_report.txt"
        report = classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
        report_path.write_text(report, encoding="utf-8")

        cm_path = artifacts_dir / "confusion_matrix.csv"
        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(
            cm,
            index=["actual_no_churn", "actual_churn"],
            columns=["pred_no_churn", "pred_churn"]
        ).to_csv(cm_path, index=True)

        # Log artifact folder dengan path relatif, bukan path Windows absolut.
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="training_artifacts")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5)
        )

    print("=" * 60)
    print("Training CI selesai")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Artifacts berhasil di-log ke MLflow.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", type=str, default="telco_churn_preprocessed.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)

    args = parser.parse_args()

    train_model(
        data_file=args.data_file,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )
