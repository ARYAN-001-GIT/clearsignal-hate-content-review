#!/usr/bin/env python3
"""Train a separate HASOC English/Indic/code-mixed text baseline."""

from __future__ import annotations

import json
import os
import sys
from getpass import getpass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import build_pipeline
from src.multilingual import HASOCPasswordRequired, load_hasoc


def main() -> None:
    print("Loading official HASOC English, Hindi, Marathi and English-Hindi code-mixed training data...")
    try:
        frame = load_hasoc()
    except HASOCPasswordRequired:
        password = os.environ.get("HASOC_DATASET_PASSWORD") or getpass(
            "Enter the HASOC dataset password from your registration (not an API key): "
        )
        if not password:
            raise HASOCPasswordRequired("No HASOC dataset password was entered. Register with HASOC and run this command again.")
        frame = load_hasoc(password=password)
    if frame["label"].value_counts().min() < 2:
        raise ValueError("Each binary class needs at least two examples for a training/validation split.")
    train, validation = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["label"],
    )
    model = build_pipeline()
    model.fit(train["text"], train["label"])
    predicted = model.predict(validation["text"])

    per_language: dict[str, dict[str, float | int]] = {}
    for language, group in validation.assign(prediction=predicted).groupby("language"):
        per_language[str(language)] = {
            "examples": int(len(group)),
            "accuracy": round(float(accuracy_score(group["label"], group["prediction"])), 4),
            "macro_f1": round(float(f1_score(group["label"], group["prediction"], labels=["not_harmful", "harmful"], average="macro", zero_division=0)), 4),
        }

    metrics = {
        "dataset": "HASOC 2021 multilingual tasks",
        "task": "Binary harmful-content screening; original HASOC task labels are collapsed and are not equivalent to HateXplain's hate class.",
        "model": "Slang-normalized word+character TF-IDF and class-balanced logistic regression",
        "seed": 42,
        "split_method": "Stratified 80/20 split of the downloaded official training archives; this is a local validation split, not the official test set.",
        "train_examples": int(len(train)),
        "validation_examples": int(len(validation)),
        "source_counts": {str(key): int(value) for key, value in frame["language"].value_counts().items()},
        "validation_language_metrics": per_language,
        "accuracy": round(float(accuracy_score(validation["label"], predicted)), 4),
        "macro_f1": round(float(f1_score(validation["label"], predicted, labels=["not_harmful", "harmful"], average="macro", zero_division=0)), 4),
        "classes": ["not_harmful", "harmful"],
        "confusion_matrix": confusion_matrix(validation["label"], predicted, labels=["not_harmful", "harmful"]).tolist(),
        "classification_report": classification_report(validation["label"], predicted, labels=["not_harmful", "harmful"], output_dict=True, zero_division=0),
        "note": "Research baseline only. Language detection in the app is a lightweight hint; do not treat predictions as moderation decisions.",
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(model, artifacts / "hasoc_multilingual_model.joblib")
    (artifacts / "hasoc_multilingual_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("train_examples", "validation_examples", "accuracy", "macro_f1", "validation_language_metrics")}, indent=2))
    print(f"\nSaved the separate multilingual model and report under {artifacts}")


if __name__ == "__main__":
    main()
