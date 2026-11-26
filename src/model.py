"""TF-IDF baseline, benchmark metrics and evidence extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.text_utils import normalize_documents

CLASSES = ["normal", "offensive", "hate"]
PIPELINE_VERSION = 2


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True)),
        ]
    )
    classifier = LogisticRegression(max_iter=1200, class_weight="balanced", C=2.0, random_state=42)
    return Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_documents, validate=False)),
            ("features", features),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(frame: pd.DataFrame, weights: list[float] | None = None) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    train = frame[frame["split"].isin(["train", "feedback_train"])].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or test.empty or train["label"].nunique() < 2:
        raise ValueError("Training and test splits need at least two label classes.")
    if weights is None:
        weights = [1.0] * len(train)
    pipeline = build_pipeline()
    pipeline.fit(train["text"], train["label"], classifier__sample_weight=np.asarray(weights))

    test_pred = pipeline.predict(test["text"])
    report = classification_report(test["label"], test_pred, labels=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        "dataset": "HateXplain",
        "pipeline_version": PIPELINE_VERSION,
        "model": "slang-normalized word+character TF-IDF and class-balanced logistic regression",
        "seed": 42,
        "train_examples": int(len(train)),
        "validation_examples": int(len(validation)),
        "test_examples": int(len(test)),
        "feedback_examples": int((train["split"] == "feedback_train").sum()),
        "macro_f1": float(f1_score(test["label"], test_pred, labels=CLASSES, average="macro", zero_division=0)),
        "classes": CLASSES,
        "confusion_matrix": confusion_matrix(test["label"], test_pred, labels=CLASSES).tolist(),
        "classification_report": report,
        "note": "Held-out official HateXplain test split; not an LLM and not a deployment safety estimate.",
    }
    predictions = test[["id", "text", "label", "targets", "rationales"]].copy()
    predictions["prediction"] = test_pred
    predictions["confidence"] = pipeline.predict_proba(test["text"]).max(axis=1)
    predictions["correct"] = predictions["label"] == predictions["prediction"]
    return pipeline, metrics, predictions


def top_evidence(pipeline: Pipeline, text: str, limit: int = 8) -> tuple[str, list[dict[str, float]], dict[str, float]]:
    probabilities = pipeline.predict_proba([text])[0]
    classes = list(pipeline.named_steps["classifier"].classes_)
    winner_idx = int(np.argmax(probabilities))
    label = classes[winner_idx]
    transformed = pipeline.named_steps["features"].transform(normalize_documents([text]))
    names = pipeline.named_steps["features"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    coef = classifier.coef_[winner_idx]
    sparse_row = transformed.tocsr()
    contributions: dict[str, float] = {}
    for col, value in zip(sparse_row.indices, sparse_row.data):
        name = str(names[col])
        if name.startswith("word__"):
            feature = name.removeprefix("word__")
            token = feature.split()
            if len(token) == 1 and feature not in {"", "_"}:
                contributions[feature] = contributions.get(feature, 0.0) + float(value * coef[col])
    evidence = [
        {"token": token, "score": round(score, 4)}
        for token, score in sorted(contributions.items(), key=lambda pair: pair[1], reverse=True)[:limit]
        if score > 0
    ]
    scores = {class_name: round(float(prob), 4) for class_name, prob in zip(classes, probabilities)}
    return label, evidence, scores


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
