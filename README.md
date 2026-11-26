# ClearSignal

**Explainable text screening with local meme text extraction**

ClearSignal is a classroom research prototype. It uses HateXplain for an English three-class text baseline and HASOC for a separate multilingual harmful-content task. For memes, it can extract visible words locally from an uploaded image and pass the reviewed text to the selected text model. It does not classify image visuals. Bengali meme analysis is not part of the current workflow.

> **Research use only.** Model scores are estimates. Check the original post or meme and use human judgment; do not use this app to punish people or automatically remove content. The app does not understand conversation history or the full visual meaning of a meme.

## Features

- Streamlit dashboard with text review, benchmark reporting, feedback notes and dataset/method details.
- HateXplain English baseline: normal/offensive/hate, word and character TF-IDF, logistic regression, token evidence and official train/validation/test divisions.
- HASOC English, Hindi, Marathi and English-Hindi code-mixed baseline: separate broad harmful/not-harmful labels and local validation. HASOC archives require the password from dataset registration.
- Local meme OCR: macOS Vision on Macs, or Tesseract on Windows/Linux. Extracted text can be corrected before analysis. OCR and text classification run on this computer without an API key.
- Gujarati and Bodo can be selected for OCR, but this project does not yet have a trained classifier for those languages. It will show the extracted text without guessing a result.
- Slang expansions are shown for review. Human corrections for the HateXplain model are saved locally and can be included in supervised retraining; this is not RLHF or LLM reward training.

## Run in VS Code

1. Open this `ClearSignal` folder in VS Code.
2. Open **Terminal → New Terminal** and create the environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`.

3. Install the project libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

The repository does not include downloaded benchmark posts or trained model files. To create the English HateXplain model locally, run `python scripts/train_model.py`; the script downloads the dataset and saves the model under the ignored `artifacts/` folder. If no model is available, the app marks its sample output as illustrative.

### Train the Hindi and other HASOC text model

HASOC's official archives are password-protected. Register on the [official HASOC dataset page](https://hasocfire.github.io/hasoc/2021/dataset.html) to receive the dataset password, then run:

```bash
python scripts/train_multilingual.py
```

The script asks for the dataset password in the VS Code terminal and does not print it. It is a dataset access password, separate from an API key; no API is used. After it saves the model, return to the app. The model loader notices the new file automatically. The report uses a local validation split, not the official test set.

### OCR setup

- **macOS:** the project uses built-in Vision text recognition locally. Reinstalling `requirements.txt` adds its Python connector. If Vision cannot process images on your setup, install the local Tesseract program with `brew install tesseract` for the fallback.
- **Windows/Linux:** install the Tesseract program and the language data you need, then install the project requirements. Type or correct meme text in the app if OCR cannot read it.

## Dataset choices and task labels

### HateXplain

HateXplain is the English explainability benchmark with `normal`, `offensive` and `hate` labels, target annotations and human rationale spans. The project preserves its official splits. See [DATASET_CARD.md](DATASET_CARD.md) and the [official repository](https://github.com/hate-alert/HateXplain).

### HASOC text

The [official HASOC 2021 page](https://hasocfire.github.io/hasoc/2021/dataset.html) lists English, Hindi, Marathi and English-Hindi code-mixed tasks. HASOC labels harmful/offensive content more broadly than HateXplain's hate class. The project maps `HOF` to `harmful` and `NOT` to `not_harmful`; those labels remain separate from HateXplain. Archives are password-protected, and the project does not include the password.

### Meme image text

The meme page extracts text with local OCR and then uses the selected text model. English text can use the English baseline; Hindi, Marathi and English-Hindi require the trained HASOC model. The [HASOC 2025 meme page](https://hasocfire.github.io/hasoc/2025/dataset.html) lists Hindi, Gujarati and Bodo, among other tasks, but those meme training archives are not included or trained here. Bengali memes are excluded from the current interface. No result is shown for a language without a matching trained model.

The current OCR path reads visible words only. It is not a visual-language model and cannot infer meaning from a person, symbol, scene, sarcasm or image context. A model trained on social media text is also not a validated meme classifier.

## Slang normalization

`data/slang_lexicon.json` contains conservative whole-token expansions for common chat shorthand, including a few romanized Hindi forms such as `nhi` and `kyu`. It is a normalization hint, not translation. The review page shows the suggested version so the original and expanded text can be compared.

## Project map

```text
app.py                         Streamlit dashboard
src/data.py                    HateXplain download and parsing
src/model.py                   Text classifier, evaluation and word evidence
src/multilingual.py             HASOC download, password-aware parser and labels
src/ocr.py                     Local macOS Vision / Tesseract meme OCR
src/text_utils.py              Slang normalization and language hints
src/feedback.py                Local reviewer corrections
scripts/train_model.py         HateXplain benchmark training
scripts/train_multilingual.py  HASOC multilingual training
artifacts/                     Local models and evaluation reports
deliverables/                  Paper draft and presentation materials
```

## Limits and responsible use

- Language selection is supplied by the reviewer. It is not verified automatically.
- OCR can misread stylized, small, curved or low-quality text; check the extraction against the image.
- Gujarati and Bodo have OCR options but no trained classification model in this project yet.
- A text classifier cannot understand the full image, sarcasm, local context or conversation history.
- Dataset labels can contain disagreement or bias. Metrics do not establish fairness or safety.
- Downloaded dataset files and trained artifacts stay local and must be used under their source terms. Do not publish individual dataset examples.

## Deliverables and citations

The paper and presentation files are drafts and must match the experiments actually run. Insert only verified values from the saved evaluation reports. Cite HateXplain and HASOC as listed in the paper references.
