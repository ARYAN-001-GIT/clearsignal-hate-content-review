#!/usr/bin/env python3
"""Train the baseline and save a held-out benchmark report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_hatexplain
from src.feedback import apply_feedback, read_feedback
from src.model import train_and_evaluate


def target_slices(predictions: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[bool]] = {}
    for _, row in predictions.iterrows():
        for target in row["targets"] or []:
            groups.setdefault(str(target), []).append(bool(row["correct"]))
    return {
        target: {
            "examples": len(values),
            "accuracy": round(float(sum(values) / len(values)), 4),
        }
        for target, values in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def main() -> None:
    print("Loading the official HateXplain benchmark...")
    benchmark = load_hatexplain()
    feedback = read_feedback()
    training_frame, weights = apply_feedback(benchmark, feedback)
    model, metrics, predictions = train_and_evaluate(training_frame, weights)
    metrics["target_slices"] = target_slices(predictions)
    metrics["accuracy"] = round(float(accuracy_score(predictions["label"], predictions["prediction"])), 4)
    metrics["feedback_weight"] = 2.0

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(model, artifacts / "clearsignal_model.joblib")
    (artifacts / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = []
    for name, values in metrics["classification_report"].items():
        rows.append({"label": name, **values} if isinstance(values, dict) else {"label": name, "accuracy": values})
    pd.DataFrame(rows).to_csv(artifacts / "classification_report.csv", index=False)
    print(json.dumps({key: metrics[key] for key in ["train_examples", "validation_examples", "test_examples", "feedback_examples", "accuracy", "macro_f1"]}, indent=2))
    print(f"\nSaved model and report under {artifacts}")
    print("Feedback corrections are included in training only; the official test IDs remain held out.")


if __name__ == "__main__":
    main()
