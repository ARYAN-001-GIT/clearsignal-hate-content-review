"""Download and read separate HASOC multilingual training archives."""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HASOC_DIR = ROOT / "data" / "raw" / "hasoc"

# Official HASOC 2021 training archives. Each source uses its own task labels.
HASOC_SOURCES = {
    "English": "https://hasocfire.github.io/hasoc/2021/files/en_Hasoc2021_train.zip",
    "Hindi": "https://hasocfire.github.io/hasoc/2021/files/hi_Hasoc2021_train.zip",
    "Marathi": "https://hasocfire.github.io/hasoc/2021/files/mr_Hasoc2021_train.zip",
    "English-Hindi code-mixed": "https://hasocfire.github.io/hasoc/2021/files/ICHCL_train.zip",
}


class HASOCPasswordRequired(ValueError):
    """Raised when an official HASOC ZIP needs the organizer-provided password."""


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ClearSignal-student-research/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {path.name}. Check internet access and try again.") from exc


def _column_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_column(columns: list[object], choices: tuple[str, ...]) -> object | None:
    normalized = {_column_key(column): column for column in columns}
    for choice in choices:
        if choice in normalized:
            return normalized[choice]
    for key, column in normalized.items():
        if any(choice in key for choice in choices):
            return column
    return None


def _map_label(value: object) -> str | None:
    label = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if label in {"not", "none", "non_hate", "non_hateful", "not_hateful", "not_offensive", "normal", "clean", "0", "0.0"}:
        return "not_harmful"
    if label in {"hof", "hate", "hateful", "offensive", "harmful", "1", "1.0", "yes"}:
        return "harmful"
    return None


def _read_table(payload: bytes, name: str) -> pd.DataFrame | None:
    if not name.lower().endswith((".csv", ".tsv", ".txt")):
        return None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(payload), sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
            continue
    return None


def load_hasoc(download: bool = True, password: str | None = None) -> pd.DataFrame:
    """Return training rows with a conservative binary harmful/not-harmful label.

    HASOC's original labels include offensive-content categories, not just hate
    speech. They are intentionally collapsed to a broader content label for
    this separate experiment and are never mixed into the HateXplain classes.
    """
    HASOC_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for language, url in HASOC_SOURCES.items():
        archive_path = HASOC_DIR / f"{_column_key(language)}.zip"
        if not archive_path.exists() and download:
            _download(url, archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing {archive_path.name}; run with downloads enabled or fetch it from the official HASOC page.")
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith((".csv", ".tsv", ".txt"))]
                for info in members:
                    try:
                        payload = archive.read(info, pwd=password.encode("utf-8") if password else None)
                    except RuntimeError as exc:
                        message = str(exc).lower()
                        if "password" in message or "decrypt" in message:
                            raise HASOCPasswordRequired(
                                "HASOC archives are password-protected. Register on the official HASOC dataset page, then rerun and enter the dataset password. This password is separate from any API key."
                            ) from exc
                        raise
                    table = _read_table(payload, info.filename)
                    if table is None or table.empty:
                        continue
                    text_column = _find_column(list(table.columns), ("text", "tweet", "post", "comment", "sentence", "content", "message"))
                    label_column = _find_column(list(table.columns), ("task1", "label", "class", "category", "target"))
                    if text_column is None or label_column is None:
                        continue
                    for text, raw_label in zip(table[text_column], table[label_column]):
                        clean_text = str(text).strip()
                        mapped = _map_label(raw_label)
                        if clean_text and clean_text.lower() != "nan" and mapped:
                            rows.append({"text": clean_text, "label": mapped, "language": language, "source_file": f"{archive_path.name}:{info.filename}"})
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{archive_path.name} is not a readable ZIP file. Delete it and run the training command again.") from exc

    frame = pd.DataFrame(rows).drop_duplicates(subset=["text", "label", "language"]).reset_index(drop=True)
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError("HASOC files were downloaded, but usable text and both label classes were not found. Check the source archive format.")
    return frame
