"""Download and normalize the official HateXplain research files."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_PATH = RAW_DIR / "dataset.json"
SPLITS_PATH = RAW_DIR / "post_id_divisions.json"

FILES = {
    DATASET_PATH: "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
    SPLITS_PATH: "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json",
}

LABELS = {"normal": "normal", "offensive": "offensive", "hatespeech": "hate", "hate": "hate"}


def download_sources(force: bool = False) -> None:
    """Fetch the two public source files, preserving a local copy for reproducibility."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in FILES.items():
        if destination.exists() and not force:
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "ClearSignal-student-research/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {destination.name}. Check your internet connection or follow "
                "DATASET_CARD.md to place the source files in data/raw/."
            ) from exc


def _majority(values: list[str], default: str = "") -> str:
    cleaned = [str(value).strip().lower() for value in values if value]
    if not cleaned:
        return default
    counts = pd.Series(cleaned).value_counts()
    # Stable tie-break: alphabetic order, so reruns always make the same choice.
    winners = sorted(counts[counts == counts.max()].index.tolist())
    return winners[0]


def _rationale_tokens(record: dict[str, Any], tokens: list[str]) -> list[str]:
    annotations = record.get("rationales") or []
    if not annotations or not tokens:
        return []
    votes = [0] * len(tokens)
    for rationale in annotations:
        for index, selected in enumerate(rationale[: len(tokens)]):
            votes[index] += int(bool(selected))
    threshold = max(1, (len(annotations) + 1) // 2)
    return [token for token, count in zip(tokens, votes) if count >= threshold]


def load_hatexplain() -> pd.DataFrame:
    """Return a normalized frame with official split labels and human annotations."""
    if not DATASET_PATH.exists() or not SPLITS_PATH.exists():
        download_sources()

    with DATASET_PATH.open(encoding="utf-8") as stream:
        posts = json.load(stream)
    with SPLITS_PATH.open(encoding="utf-8") as stream:
        split_ids = json.load(stream)

    split_by_id: dict[str, str] = {}
    for split, ids in split_ids.items():
        normalized_split = {"valid": "validation", "val": "validation"}.get(split.lower(), split.lower())
        for post_id in ids:
            split_by_id[str(post_id)] = normalized_split

    rows: list[dict[str, Any]] = []
    for post_id, record in posts.items():
        annotators = record.get("annotators") or []
        label_values = [LABELS.get(str(a.get("label", "")).lower()) for a in annotators]
        label = _majority([x for x in label_values if x])
        if not label:
            continue
        tokens = [str(token) for token in (record.get("post_tokens") or [])]
        if not tokens:
            continue
        targets = sorted({str(target) for a in annotators for target in (a.get("target") or []) if target})
        rows.append(
            {
                "id": str(post_id),
                "text": " ".join(tokens),
                "label": label,
                "split": split_by_id.get(str(post_id), "unassigned"),
                "rationales": _rationale_tokens(record, tokens),
                "targets": targets,
                "annotator_labels": [x for x in label_values if x],
            }
        )
    frame = pd.DataFrame(rows)
    required = {"train", "validation", "test"}
    if not required.issubset(set(frame["split"].unique())):
        raise ValueError("The official split file did not provide train, validation and test partitions.")
    return frame
