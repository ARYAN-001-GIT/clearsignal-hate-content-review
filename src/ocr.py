"""Local text extraction for meme images; this module never calls an API."""

from __future__ import annotations

import io
import sys
from typing import Any

from PIL import Image


VISION_LANGUAGES = {
    "English": ("en",),
    "Hindi": ("hi",),
    "Marathi": ("mr",),
    "English-Hindi code-mixed": ("en", "hi"),
    "Gujarati": ("gu",),
    "Bodo": ("brx",),
}
TESSERACT_LANGUAGE = {
    "English": "eng",
    "Hindi": "hin",
    "Marathi": "mar",
    "English-Hindi code-mixed": "eng+hin",
    "Gujarati": "guj",
    "Bodo": "brx",
}


def _vision_ocr(image_bytes: bytes, language: str) -> tuple[str, str] | None:
    """Use macOS Vision when its installed OCR model supports the chosen language."""
    if sys.platform != "darwin":
        return None
    try:
        import Vision
        from Foundation import NSData
    except ImportError:
        return None

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    try:
        supported, error = request.supportedRecognitionLanguagesAndReturnError_(None)
        if error:
            supported = []
    except (AttributeError, TypeError):
        supported = []

    prefixes = VISION_LANGUAGES.get(language, ())
    selected_languages = [
        next((code for code in supported if str(code).lower().startswith(prefix + "-")), "")
        for prefix in prefixes
    ]
    selected_languages = [code for code in selected_languages if code]
    if selected_languages:
        request.setRecognitionLanguages_(selected_languages)
    else:
        try:
            request.setAutomaticallyDetectsLanguage_(True)
        except AttributeError:
            pass

    payload = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
    # Pass nil for options: PyObjC bridges an empty Python dict as a key/value
    # sequence here, which Vision rejects with “key does not exist”.
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(payload, None)
    success, error = handler.performRequests_error_([request], None)
    if error:
        raise RuntimeError(str(error))
    if not success:
        raise RuntimeError("macOS Vision could not read the image.")

    lines: list[str] = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if candidates:
            lines.append(str(candidates[0].string()))
    note = "The selected OCR language is unavailable on this Mac; automatic OCR was used." if prefixes and len(selected_languages) < len(prefixes) else ""
    return "\n".join(lines).strip(), note


def _tesseract_ocr(image_bytes: bytes, language: str) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("Install the project requirements to enable local OCR, or type the meme text yourself.") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return pytesseract.image_to_string(image, lang=TESSERACT_LANGUAGE.get(language, "eng")).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract is not installed. Install it on this computer, or type the meme text yourself.") from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"OCR could not read the selected language. Check that its Tesseract language data is installed, or type the meme text yourself. ({exc})") from exc


def extract_image_text(image_bytes: bytes, language: str) -> dict[str, Any]:
    """Extract words on-device, falling back to Tesseract if Vision is unavailable."""
    vision_note = ""
    try:
        vision_result = _vision_ocr(image_bytes, language)
        if vision_result is not None:
            text, note = vision_result
            if text:
                return {"text": text, "engine": "macOS Vision", "message": note}
            vision_note = note or "macOS Vision did not find readable words."
    except Exception as exc:
        vision_note = "macOS Vision could not process this image."

    try:
        text = _tesseract_ocr(image_bytes, language)
        return {
            "text": text,
            "engine": "Tesseract, running locally",
            "message": "" if text else "No readable words were detected; you can type them in the box below.",
        }
    except Exception as exc:
        if vision_note:
            return {
                "text": "",
                "engine": "",
                "message": f"{vision_note} Local Tesseract is not ready on this computer. Type the meme words below, or install Tesseract for local OCR. No API is used.",
            }
        return {"text": "", "engine": "", "message": str(exc)}
