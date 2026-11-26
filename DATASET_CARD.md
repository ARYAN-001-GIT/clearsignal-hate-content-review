# Dataset cards: ClearSignal experiments

HateXplain is the primary explainability benchmark. HASOC is a separate multilingual text extension. The meme page uses local OCR and then a matching text model; it is not a visual meme classifier. Keep the dataset labels and evaluation results separate.

## Primary dataset: HateXplain

### Source

- Official repository: <https://github.com/hate-alert/HateXplain>
- Data guide: <https://github.com/hate-alert/HateXplain/blob/master/Data/README.md>
- Main file: `Data/dataset.json`
- Split file: `Data/post_id_divisions.json`
- Paper: Mathew et al. (2021), *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*, AAAI 2021.

The source repository warns that examples may contain offensive or hateful language. The app does not show training posts.

### Task, intended use and limits

The model predicts `normal`, `offensive` or `hate`. For each post, the loader derives a consensus label from annotator labels. Rationale spans and target communities are retained. Official train, validation and test IDs are respected.

This is an English classroom baseline, not an automatic moderator or a decision-maker about a person. It does not validate Hindi, Hinglish or other languages. Token evidence is a learned association, not proof of intent. An English text benchmark does not establish meme or image performance.

### Reproduction

Run `python scripts/train_model.py`. The program saves a local dataset snapshot under `data/raw/` and an evaluation report under `artifacts/`.

---

## Extension: HASOC multilingual text

### Source and languages

- Official dataset page: <https://hasocfire.github.io/hasoc/2021/dataset.html>
- Tasks used: English, Hindi, Marathi and English-Hindi code-mixed.
- HASOC archives are password-protected. Register with the organizers to obtain the dataset password; the script prompts for it and does not display it. It is not an API key.

### Task and processing

The adapter maps the broad `HOF` class to `harmful` and `NOT` to `not_harmful`. These labels are not the same as HateXplain's `hate` class. Rows with unsupported labels are skipped. The model uses word/character TF-IDF and class-balanced logistic regression.

### Evaluation and limits

The trainer makes a fixed-seed stratified 80/20 validation split from downloaded training archives. It reports overall and source-language scores. This is local validation, not official test performance. Language groups can be small; the model is not a multilingual transformer and only covers the dataset languages.

---

## Meme image text: OCR-assisted review

### Source and languages

The app uses the trained English baseline for English text. Hindi, Marathi and English-Hindi code-mixed text require a trained HASOC model. The [HASOC 2025 meme page](https://hasocfire.github.io/hasoc/2025/dataset.html) lists Hindi, Gujarati and Bodo meme tasks, but this project does not yet contain or train on those meme archives. Bengali meme analysis is excluded from the current app.

### Processing and limits

On macOS, the app uses Apple Vision locally when its OCR framework is available. Windows/Linux can use Tesseract with matching language data. The reviewer can edit OCR output before classification. No API is called.

The model analyzes the extracted words only. It does not classify visual scenes, symbols, people, sarcasm or image-text interactions, and text models trained on social posts are not validated meme models. OCR output may be wrong; inspect it against the image. The app does not show a prediction for languages without a matching trained model.

---

## Text normalization shared by the baselines

`data/slang_lexicon.json` contains conservative whole-token expansions for common chat shorthand, including a few romanized Hindi forms. This is not a translation or validated slang lexicon. The review page shows the proposed expansion so a person can compare it to the original.
