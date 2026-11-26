# ClearSignal: Explainable Text Review and OCR-Assisted Meme Analysis

**Research paper draft · multilingual results require the registered HASOC data password**

> This draft reports only results already present in the local evaluation files. The HASOC model has not been trained because its archives require a separate organizer-provided dataset password. Do not present the OCR pathway as a visual meme classifier.

## Abstract

We present ClearSignal, a classroom research prototype for reviewing short text and meme text with visible model evidence. Its English HateXplain baseline predicts normal, offensive or hate with word and character TF-IDF and logistic regression. It displays learned word evidence, conservative slang expansions and a local reviewer-correction workflow. For an uploaded meme, local OCR extracts visible words and lets the reviewer correct them before text classification. The app can use a separate HASOC harmful/not-harmful model for English, Hindi, Marathi and English-Hindi code-mixed text once the password-protected training archives are supplied. The English fallback uses the HateXplain label set and is not validated on memes. HASOC 2025 lists additional Hindi, Gujarati and Bodo meme tasks, but these data are not included or trained in this prototype. The app does not classify image visuals or include Bengali meme analysis. These are bounded classroom experiments, not an automated moderation system.

## 1. Introduction

Short posts can depend on irony, quotation, reclaimed language, local slang and conversation history. Memes add visual context that can change the meaning of their words. A classifier that returns only a label can be hard to review, and dataset associations can reflect shortcuts rather than intent.

ClearSignal keeps text tasks and labels separate. Its interface supports a trained English baseline, a multilingual HASOC training workflow, local meme text extraction and human review. It does not claim full multilingual coverage or image understanding. Reviewer corrections are stored locally for later supervised retraining of the English experiment; no LLM reward model is trained.

## 2. Data and tasks

### HateXplain

HateXplain provides English posts with normal, offensive or hate labels, target-community annotations and human rationale spans. The loader derives consensus labels and retains rationale and target annotations. The official train, validation and test partitions are preserved. The saved local report records macro F1 0.6573 and accuracy 0.6694 on 1,924 held-out test examples. These are results for the English text benchmark only.

### HASOC

The HASOC 2021 tasks cover English, Hindi, Marathi and English-Hindi code-mixed text. The local adapter maps `HOF` to `harmful` and `NOT` to `not_harmful`; these broad labels are not equivalent to HateXplain's `hate` class. The training script creates an 80/20 stratified validation split from official training archives. Those archives are password-protected. The password is not included in this project, and no HASOC model or metric is reported until training succeeds.

### Meme images

The app can upload common image formats, extract visible text on-device with macOS Vision or Tesseract, and let a reviewer edit the extraction. It then applies the selected text model. English uses HateXplain if the HASOC model is unavailable. Hindi, Marathi and English-Hindi require a trained HASOC model. Gujarati and Bodo OCR options do not have a matching classifier in this project. The official [HASOC 2025 meme task](https://hasocfire.github.io/hasoc/2025/dataset.html) lists Hindi, Gujarati, Bodo and Bangla data; only the first three are in the interface, and their archives are not included here. The interface excludes Bengali meme analysis.

OCR plus text classification does not analyze image scenes, symbols, sarcasm or image-text interaction. Results from text-trained models are not validated meme results.

## 3. Method

### Text classification and evidence

The English baseline combines word unigrams/bigrams and character n-grams with TF-IDF and class-balanced logistic regression. A local evidence view ranks present word features by their contribution to the selected linear-model class. This is a feature association, not a causal explanation. Human rationale spans are retained in the data pipeline but are not used to train the initial model.

The separate HASOC workflow uses the same model family and its own binary labels. Reports preserve the validation split and language sample counts. Its model can only be trained after the official data password is entered locally.

### Slang and corrections

A small JSON lexicon expands exact whole-token chat abbreviations, including a few romanized Hindi forms. The interface displays original and normalized text for comparison. Expansions can be wrong and do not translate text. Reviewer corrections can be appended to the HateXplain training examples with a 2.0 sample weight at a later supervised run. Test examples are not used for updates. This is not RLHF or LLM fine-tuning.

### Local OCR

For image uploads, macOS Vision is used when available; other systems can use Tesseract and installed language data. OCR output remains editable. The classifier receives text only. OCR can miss or misread decorative, low-resolution or curved text.

## 4. Results and status

| Experiment | Status | Split or evidence |
|---|---|---|
| HateXplain English text | Trained | Official held-out test: accuracy 0.6694, macro F1 0.6573, 1,924 examples |
| HASOC multilingual text | Training workflow available; model not trained | Dataset password required; no result claimed |
| Meme review | OCR-assisted text only | No visual classifier or meme-specific evaluation |
| Gujarati/Bodo classification | Not trained | OCR text can be shown; app does not predict |

Do not compare the three-class HateXplain score with a HASOC binary score as if the tasks were identical. Do not claim meme performance from the English text benchmark.

## 5. Limitations and responsible use

The main classifier is a conventional TF-IDF baseline, not a contextual language model. OCR and slang normalization can fail. Selected languages are supplied by the reviewer and not verified. HASOC results will cover only their source data, once trained. Gujarati and Bodo have no classifier in this project. The app does not analyze images visually, resolve sarcasm or use conversation context. Dataset annotations can be subjective or biased. No score should independently trigger punishment or content removal.

## References

Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., and Mukherjee, A. (2021). HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection. *AAAI*, 35(17), 14867–14875. https://doi.org/10.1609/aaai.v35i17.17745.

HASOC. (2021). Hate Speech and Offensive Content Identification in English and Indo-Aryan Languages: Dataset page. https://hasocfire.github.io/hasoc/2021/dataset.html.

HASOC. (2025). Hate Speech and Offensive Content Identification in Memes in Bangla, Hindi, Gujarati and Bodo: Dataset page. https://hasocfire.github.io/hasoc/2025/dataset.html.
