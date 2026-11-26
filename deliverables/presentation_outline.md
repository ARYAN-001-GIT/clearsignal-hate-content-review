# Presentation outline: ClearSignal

**Suggested length: 8–10 minutes · 10 slides**

## 1. Title

**ClearSignal: Explainable Text Review and Meme Text Extraction**

Subtitle: Multilingual text, slang review and human feedback. Add presenter name, course and instructor.

## 2. Why the task is difficult

- Short posts can depend on context, irony, quotation and local language.
- A model score alone is difficult to review.
- Meme text may be stylized, and the image can change how its words read.

Use classroom-safe examples and avoid showing dataset examples.

## 3. What the project includes

- HateXplain: three-class English text and learned word evidence.
- HASOC: separate English, Hindi, Marathi and English-Hindi code-mixed text workflow.
- Meme images: local OCR, editable extracted text and text-only review.
- Inspectable slang expansion and a local human-correction workflow.

These are separate experiments, not one merged model. Bengali meme analysis is excluded.

## 4. Datasets and label meanings

- HateXplain: normal, offensive, hate; official train/validation/test split.
- HASOC: broad harmful/not-harmful mapping and local validation from official training data; password required.
- HASOC 2025 lists Hindi, Gujarati and Bodo meme tasks, but those meme archives are not included or trained here.

Explain that a text benchmark does not establish meme performance. Cite [HateXplain](https://github.com/hate-alert/HateXplain), [HASOC 2021](https://hasocfire.github.io/hasoc/2021/dataset.html) and [HASOC 2025](https://hasocfire.github.io/hasoc/2025/dataset.html).

## 5. Baseline systems

- HateXplain uses word/character TF-IDF and logistic regression for English text.
- HASOC has a separate binary model; it still needs its registered dataset password before training.
- Meme review uses local OCR followed by a selected text model. It does not interpret visual image content.

Use a flow diagram: image → local OCR → reviewer correction → language text model → human review.

## 6. Slang and language review

- A small lexicon expands common whole-word abbreviations.
- The interface displays the suggested version while preserving the original.
- The language choice is supplied by a reviewer, not verified automatically.
- Ask a reviewer to check regional meanings before trusting normalization.

## 7. Explanation and human feedback

- Show model class scores and strongest positive word contributions.
- A person can correct a label and add an optional note.
- Corrections can be weighted in later supervised HateXplain retraining; no LLM reward model is trained.

## 8. Evaluation and current status

- HateXplain: saved official held-out test macro F1 0.6573, accuracy 0.6694, 1,924 examples.
- HASOC: no score yet; the official training archives require organizer registration and a password.
- Meme review: OCR-assisted text only, with no meme-specific evaluation.

Keep different labels and tasks in separate tables.

## 9. Limitations and responsible use

- OCR, language selection and slang expansion can be wrong.
- A text model cannot interpret visual meaning, sarcasm or conversation history.
- HASOC results will be limited to their source languages and labels; Gujarati and Bodo have no trained classifier here.
- Never use a prototype score alone to punish a person or remove content.

## 10. Next steps and references

- Register for HASOC access and train the multilingual text model locally.
- Add an openly accessible, license-reviewed meme dataset before claiming meme classification.
- Compare reviewer feedback before/after on the same held-out split.
- Cite Mathew et al. (2021) and HASOC.

Close with: “Clearer evidence supports better review, while the decision stays with a human.”
