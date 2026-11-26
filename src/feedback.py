"""Local reviewer corrections used as weighted training examples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "data" / "feedback.csv"
COLUMNS = ["text", "label", "reviewer_note", "created_at"]


def read_feedback(path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path).reindex(columns=COLUMNS)
    frame = frame.dropna(subset=["text", "label"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    return frame[(frame["text"] != "") & frame["label"].isin(["normal", "offensive", "hate"])]


def save_feedback(text: str, label: str, reviewer_note: str = "", path: Path = FEEDBACK_PATH) -> None:
    clean_text = " ".join(text.split())
    label = label.strip().lower()
    if len(clean_text) < 3:
        raise ValueError("Please enter a little more text before saving a correction.")
    if label not in {"normal", "offensive", "hate"}:
        raise ValueError("Choose one of the three supported labels.")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [{"text": clean_text, "label": label, "reviewer_note": reviewer_note.strip(), "created_at": pd.Timestamp.now(tz="UTC").isoformat()}]
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def apply_feedback(train: pd.DataFrame, feedback: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Append reviewer-labeled items to train only, with a modest, visible weight."""
    training_rows = train[train["split"] == "train"].copy()
    holdout_rows = train[train["split"] != "train"].copy()
    if feedback.empty:
        return train, [1.0] * len(training_rows)
    additions = pd.DataFrame(
        {
            "id": [f"review-{i}" for i in range(len(feedback))],
            "text": feedback["text"].tolist(),
            "label": feedback["label"].tolist(),
            "split": "feedback_train",
            "rationales": [[] for _ in range(len(feedback))],
            "targets": [[] for _ in range(len(feedback))],
        }
    )
    combined = pd.concat([training_rows, additions, holdout_rows], ignore_index=True)
    weights = [1.0] * len(training_rows) + [2.0] * len(additions)
    return combined, weights
