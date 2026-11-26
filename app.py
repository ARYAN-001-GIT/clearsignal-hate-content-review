from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_hatexplain
from src.feedback import FEEDBACK_PATH, read_feedback, save_feedback
from src.model import PIPELINE_VERSION, build_pipeline, top_evidence
from src.ocr import extract_image_text
from src.text_utils import detect_language, normalize_slang

MODEL_PATH = ROOT / "artifacts" / "clearsignal_model.joblib"
REPORT_PATH = ROOT / "artifacts" / "evaluation.json"
HASOC_MODEL_PATH = ROOT / "artifacts" / "hasoc_multilingual_model.joblib"
HASOC_REPORT_PATH = ROOT / "artifacts" / "hasoc_multilingual_evaluation.json"
LABEL_META = {
    "normal": ("Normal", "#35b88a"),
    "offensive": ("Offensive", "#f2a65a"),
    "hate": ("Hate speech", "#f06b76"),
}

st.set_page_config(page_title="ClearSignal", page_icon="◌", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#eaf1f5; --muted:#97aab5; --line:rgba(180,205,218,.13); --panel:#101c24; --mint:#55d6aa; --blue:#73a8ff; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background: radial-gradient(ellipse at 10% -8%, #19333c 0, #0b141b 37%, #081118 85%); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#0c171e; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b9c9d1; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
    h1 { font-weight:800; }
    .brand { display:flex; align-items:center; gap:12px; padding:8px 0 22px; }
    .brand-mark { width:38px; height:38px; border-radius:12px; display:grid; place-items:center; background:linear-gradient(135deg,#72ecc1,#7bafff); color:#082019; font-weight:900; font-size:21px; }
    .brand-title { font-family:'Manrope',sans-serif; color:#f1f7f9; font-size:19px; font-weight:800; letter-spacing:-.04em; }
    .brand-sub { color:#859ba7; font:10px 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
    .eyebrow { font:11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.16em; color:var(--mint); margin-bottom:8px; }
    .hero { border:1px solid var(--line); border-radius:22px; padding:28px 32px; background:linear-gradient(120deg,rgba(40,84,80,.42),rgba(23,48,67,.48)); margin:4px 0 20px; }
    .hero h1 { margin:0; font-size:40px; line-height:1.12; }
    .hero p { color:#b0c1c9; max-width:760px; margin:12px 0 0; font-size:15px; }
    .panel { border:1px solid var(--line); background:rgba(16,28,36,.83); border-radius:16px; padding:20px 22px; min-height:120px; }
    .panel-label { color:#99adb6; font:10px 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; }
    .panel-value { color:#f3f7f8; font:700 25px 'Manrope',sans-serif; margin-top:9px; }
    .panel-detail { color:#9eb0b9; font-size:12px; margin-top:5px; }
    .status-pill { display:inline-block; border:1px solid rgba(85,214,170,.32); background:rgba(85,214,170,.08); color:#8ee6c5; border-radius:999px; padding:5px 10px; font:10px 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
    .note { color:#9eafb8; font-size:12px; line-height:1.6; }
    .stButton > button { border-radius:10px; border:1px solid rgba(85,214,170,.35); background:#153c36; color:#dffaf0; font-weight:600; }
    .stButton > button:hover { border-color:#55d6aa; background:#1b5147; color:white; }
    .stTextArea textarea { background:#101c24; border:1px solid var(--line); border-radius:12px; color:#eef5f7; }
    [data-testid="stMetric"] { background:#101c24; border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
    [data-testid="stMetricLabel"] { color:#9eb0b9; }
    [data-testid="stMetricValue"] { color:#f2f7f8; }
    hr { border-color:var(--line); }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def demo_model():
    """A tiny synthetic bootstrap for UI exploration; never represented as benchmark evidence."""
    examples = [
        ("People deserve safety dignity and respect in every community", "normal"),
        ("I disagree with this policy and want to discuss it peacefully", "normal"),
        ("Everyone should be welcome at the local library", "normal"),
        ("We can criticize an idea without attacking a person", "normal"),
        ("That comment was rude cruel and completely unacceptable", "offensive"),
        ("Stop making insulting personal attacks during the meeting", "offensive"),
        ("Your behavior is obnoxious and disrespectful", "offensive"),
        ("That was a mean and nasty thing to say", "offensive"),
        ("People from that community should be driven out", "hate"),
        ("That group does not deserve equal rights", "hate"),
        ("Calls to threaten an entire community are unacceptable", "hate"),
        ("No one should be targeted because of their identity", "normal"),
        ("Harassment and threats against a group are harmful", "hate"),
        ("The speaker made an insulting remark about a neighbor", "offensive"),
        ("A fair discussion protects the rights of every group", "normal"),
        ("They should be banned from public spaces because of who they are", "hate"),
        ("This is a thoughtless and offensive joke", "offensive"),
        ("Equal treatment matters for all people", "normal"),
    ]
    model = build_pipeline()
    model.fit([x for x, _ in examples], [y for _, y in examples])
    return model


@st.cache_resource(show_spinner=False)
def get_model():
    if MODEL_PATH.exists():
        try:
            saved_model = joblib.load(MODEL_PATH)
            if "normalize" in getattr(saved_model, "named_steps", {}):
                return saved_model, True
        except Exception:
            pass
    return demo_model(), False


def load_metrics():
    if not REPORT_PATH.exists():
        return None
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        return report if report.get("pipeline_version") == PIPELINE_VERSION else None
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner=False)
def _load_optional_model(path: str, modified_ns: int, size: int):
    return joblib.load(path)


def load_optional_model(path: Path):
    """Reload a model when a training run creates or replaces its artifact."""
    try:
        details = path.stat()
    except OSError:
        return None
    try:
        return _load_optional_model(str(path), details.st_mtime_ns, details.st_size)
    except Exception:
        return None


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def metric_card(label: str, value: str, detail: str):
    st.markdown(
        f'<div class="panel"><div class="panel-label">{label}</div><div class="panel-value">{value}</div><div class="panel-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


model, trained = get_model()
metrics = load_metrics() if trained else None
feedback = read_feedback()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">◌</div><div><div class="brand-title">ClearSignal</div><div class="brand-sub">research prototype</div></div></div>', unsafe_allow_html=True)
    status = "benchmark model" if trained else "illustrative demo"
    st.markdown(f'<span class="status-pill">{status}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "Post review", "Languages & memes", "Feedback lab", "Benchmark", "Dataset & methods"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="note">Separate text labels · local OCR · human review required</div>', unsafe_allow_html=True)


if page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">Explainable content research</div><h1>Make each prediction<br>easier to question.</h1><p>Explore separate text models, inspect slang normalization and review meme text extracted locally from images.</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="panel"><div class="panel-label">HateXplain labels</div><div class="panel-value">03</div><div class="panel-detail">Normal · offensive · hate</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="panel"><div class="panel-label">Evidence view</div><div class="panel-value">Token</div><div class="panel-detail">Learned word contributions</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="panel"><div class="panel-label">Reviewer corrections</div><div class="panel-value">{len(feedback):,}</div><div class="panel-detail">Saved on this computer</div></div>', unsafe_allow_html=True)
    if metrics:
        c4.markdown(f'<div class="panel"><div class="panel-label">Held-out macro F1</div><div class="panel-value">{metrics["macro_f1"]:.3f}</div><div class="panel-detail">Official test split · local run</div></div>', unsafe_allow_html=True)
    else:
        c4.markdown('<div class="panel"><div class="panel-label">Benchmark status</div><div class="panel-value">Not run</div><div class="panel-detail">Train with the official data</div></div>', unsafe_allow_html=True)
    st.markdown("### A transparent first baseline")
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("The text classifier combines word and character patterns. Review screens show a slang-normalized preview and the strongest learned word signals so a person can inspect what may have influenced the output.")
        st.markdown("Use **Post review** to analyze a short text, then save a human correction. The next training run adds reviewer-labeled examples to the training split with a modest sample weight.")
    with right:
        st.markdown('<div class="panel"><div class="panel-label">Scope check</div><div class="panel-detail" style="font-size:14px;line-height:1.8;margin-top:12px">✓ HateXplain English text baseline<br>✓ HASOC multilingual text workflow<br>✓ Local meme text extraction (OCR)<br>✓ Visible slang normalization<br>✓ Local reviewer corrections<br>— Reward-optimized LLM</div></div>', unsafe_allow_html=True)
    if not trained:
        st.info("HateXplain is using the illustrative demo until you run `python scripts/train_model.py`. That creates the current slang-aware benchmark model.")
    st.info("HASOC uses its own multilingual labels. English can use the trained HateXplain fallback while the registered HASOC files are prepared.")
    st.caption("Scores are model estimates, not truth. This prototype should not be used to remove content or make decisions about people.")


elif page == "Post review":
    st.markdown('<div class="eyebrow">Human-in-the-loop analysis</div>', unsafe_allow_html=True)
    st.title("Review a post")
    st.markdown("Enter a short text to inspect the baseline prediction, language hint and suggested slang expansions.")
    text = st.text_area("Text to review", height=150, placeholder="Paste a short public post here…", max_chars=2500, label_visibility="collapsed")
    analyze = st.button("Analyze text", type="primary", disabled=len(text.strip()) < 3)
    if analyze:
        label, evidence, scores = top_evidence(model, text.strip())
        normalized_text = normalize_slang(text.strip())
        st.caption(f"Language hint: **{detect_language(text.strip())}** · heuristic only, not a verified language detector.")
        if normalized_text != text.strip():
            with st.expander("Review suggested slang normalization"):
                st.write(normalized_text)
                st.caption("The model uses this expanded version. Check the original too: shorthand can have local or multiple meanings.")
        pretty, color = LABEL_META[label]
        st.markdown("---")
        a, b = st.columns([.9, 1.1], gap="large")
        with a:
            st.markdown(f'<div class="panel"><div class="panel-label">Model estimate</div><div class="panel-value" style="color:{color}">{pretty}</div><div class="panel-detail">The model is uncertain about context, sarcasm and intent.</div></div>', unsafe_allow_html=True)
            st.markdown("#### Class scores")
            st.dataframe(pd.DataFrame([{"Class": LABEL_META[key][0], "Score": value} for key, value in scores.items()]).sort_values("Score", ascending=False), hide_index=True, use_container_width=True)
        with b:
            st.markdown("#### Words that influenced this estimate")
            if evidence:
                evidence_frame = pd.DataFrame(evidence).rename(columns={"token": "Word", "score": "Contribution"})
                chart = px.bar(evidence_frame.sort_values("Contribution"), x="Contribution", y="Word", orientation="h", color="Contribution", color_continuous_scale=["#4d7890", "#6ce0b5"])
                chart.update_layout(height=290, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cedbe0", coloraxis_showscale=False)
                st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No positive single-word signal stood out. Character and phrase features can still affect the prediction.")
            st.markdown('<div class="note">These are learned associations, not causal reasons. The classifier may miss sarcasm, reclaimed language, quotation, conversation history and cultural context.</div>', unsafe_allow_html=True)
        st.markdown("#### Add a reviewer correction")
        with st.form("review-correction", clear_on_submit=True):
            correction = st.selectbox("Human-reviewed label", ["normal", "offensive", "hate"], format_func=lambda value: LABEL_META[value][0])
            note = st.text_input("Short reason (optional)", max_chars=240, placeholder="For your own review record")
            submitted = st.form_submit_button("Save correction")
            if submitted:
                try:
                    save_feedback(text, correction, note)
                    st.success("Saved locally. This correction will be included in training the next time you run the training script.")
                except ValueError as exc:
                    st.error(str(exc))
    st.caption("Please avoid entering names, personal details or private messages. Corrections are saved only in the local project folder.")


elif page == "Languages & memes":
    st.markdown('<div class="eyebrow">Local OCR and multilingual text review</div>', unsafe_allow_html=True)
    st.title("Languages, slang & memes")
    st.markdown("Read text from an uploaded meme on this computer, then review the extracted words with a language-specific text model. Bengali meme analysis is excluded from this workflow.")
    st.warning("Research estimates only. OCR can misread stylized text, and text-only models cannot understand the full image, sarcasm or visual context. Review the original meme yourself.")

    hasoc_model = load_optional_model(HASOC_MODEL_PATH)
    hasoc_report = read_json(HASOC_REPORT_PATH)
    status_left, status_right = st.columns(2)
    with status_left:
        st.markdown("#### Multilingual text model")
        if hasoc_model is None:
            st.success("English review is available with the English text model.")
            st.info("Hindi, Marathi and English–Hindi review use a separate HASOC model. To enable those languages, run the local training step with the dataset password from HASOC registration.")
            st.caption("In the VS Code terminal, run `python scripts/train_multilingual.py` and enter the dataset password when asked. No API key is used.")
            st.link_button("Get HASOC dataset access", "https://hasocfire.github.io/hasoc/2021/dataset.html")
        elif hasoc_report:
            st.caption(f"Model ready · local validation macro F1: {hasoc_report.get('macro_f1', 0):.3f} · {hasoc_report.get('validation_examples', 0):,} examples")
            language_rows = [{"Source language": language, **values} for language, values in hasoc_report.get("validation_language_metrics", {}).items()]
            if language_rows:
                st.dataframe(pd.DataFrame(language_rows), hide_index=True, use_container_width=True)
            with st.expander("View HASOC confusion matrix"):
                matrix = hasoc_report.get("confusion_matrix")
                if matrix:
                    classes = [name.replace("_", " ").title() for name in hasoc_report.get("classes", [])]
                    st.dataframe(pd.DataFrame(matrix, index=[f"Actual {name}" for name in classes], columns=[f"Predicted {name}" for name in classes]), use_container_width=True)
    with status_right:
        st.markdown("#### Meme image text")
        st.caption("OCR runs locally with macOS Vision, falling back to Tesseract when installed. The extracted text stays editable before review.")

    language_choice = st.selectbox(
        "Language in the post or meme",
        ["English", "Hindi", "Marathi", "English-Hindi code-mixed", "Gujarati", "Bodo", "Other / unsure"],
    )
    meme_image = st.file_uploader("Meme image (optional)", type=["png", "jpg", "jpeg", "webp", "bmp"])
    image_bytes = meme_image.getvalue() if meme_image is not None else None
    if image_bytes is not None:
        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Please choose an image smaller than 10 MB.")
            image_bytes = None
        else:
            st.image(image_bytes, caption="Uploaded meme", use_container_width=True)

    image_key = hashlib.sha256(image_bytes + language_choice.encode("utf-8")).hexdigest()[:16] if image_bytes else "no-image"
    ocr_key = f"meme_ocr_{image_key}"
    if image_bytes is not None and ocr_key not in st.session_state:
        with st.spinner("Reading meme text locally…"):
            st.session_state[ocr_key] = extract_image_text(image_bytes, language_choice)
    ocr_result = st.session_state.get(ocr_key, {"text": "", "engine": "", "message": ""})
    if image_bytes is not None and ocr_result.get("engine"):
        st.caption(f"Text extraction: {ocr_result['engine']} · local only")
    if image_bytes is not None and ocr_result.get("message"):
        st.info(ocr_result["message"])
    default_text = ocr_result.get("text", "") if image_bytes is not None else ""
    if image_bytes is not None and st.session_state.get("meme_review_image_key") != image_key:
        previous_image_key = st.session_state.get("meme_review_image_key")
        previous_text = st.session_state.get("meme_review_text", "").strip()
        if previous_image_key:
            st.session_state["meme_review_text"] = default_text
        elif default_text and previous_text:
            st.session_state["meme_review_text"] = f"{previous_text}\n{default_text}"
        else:
            st.session_state["meme_review_text"] = default_text or previous_text
        st.session_state["meme_review_image_key"] = image_key
    user_text = st.text_area(
        "Post text / extracted meme text (edit if OCR made a mistake)",
        height=130,
        max_chars=2500,
        placeholder="Type a short public post, or upload a meme and check its extracted words here.",
        key="meme_review_text",
    )
    analyze_multimodal = st.button("Review input", type="primary", disabled=not user_text.strip() and image_bytes is None)

    if analyze_multimodal:
        clean_text = user_text.strip()
        if not clean_text:
            st.warning("No readable text was found. Type the meme words in the box above and review again.")
        else:
            normalized_text = normalize_slang(clean_text)
            effective_language = language_choice
            if language_choice == "Other / unsure":
                language_hint = detect_language(clean_text)
                st.caption(f"Automatic language hint: **{language_hint}** · heuristic only, please check it.")
                detected_languages = {
                    "English / Latin-script": "English",
                    "Hindi / Devanagari": "Hindi",
                    "Hindi-English code-mixed": "English-Hindi code-mixed",
                    "Marathi / Devanagari": "Marathi",
                }
                effective_language = detected_languages.get(language_hint, "Other / unsure")
            else:
                st.caption(f"Selected task: **{language_choice}** · language matching is your choice, not an automatic verification.")
            if normalized_text != clean_text:
                with st.expander("Check the suggested slang expansion"):
                    st.write(normalized_text)
                    st.caption("The English baseline uses these suggested expansions. Check the original because slang can have several meanings.")

            hasoc_languages = set((hasoc_report or {}).get("source_counts", {}).keys())
            task_language = {
                "English": "English",
                "Hindi": "Hindi",
                "Marathi": "Marathi",
                "English-Hindi code-mixed": "English-Hindi code-mixed",
            }.get(effective_language)
            if hasoc_model is not None and task_language and (not hasoc_languages or task_language in hasoc_languages):
                probabilities = hasoc_model.predict_proba([clean_text])[0]
                class_names = list(hasoc_model.named_steps["classifier"].classes_)
                score_rows = sorted(zip(class_names, probabilities), key=lambda item: item[1], reverse=True)
                st.markdown("#### HASOC text estimate")
                st.dataframe(pd.DataFrame([{"Class": name.replace("_", " ").title(), "Model score": float(score)} for name, score in score_rows]), hide_index=True, use_container_width=True)
                st.caption("HASOC labels cover a broad harmful/offensive-content task; they are not the same as HateXplain's hate class.")
            elif effective_language == "English":
                label, evidence, scores = top_evidence(model, clean_text)
                pretty, color = LABEL_META[label]
                english_detail = "The trained English text model analyzed the typed or OCR-extracted words." if trained else "The illustrative English demo analyzed the typed or OCR-extracted words."
                st.markdown(f'<div class="panel"><div class="panel-label">HateXplain English text estimate</div><div class="panel-value" style="color:{color}">{pretty}</div><div class="panel-detail">{english_detail}</div></div>', unsafe_allow_html=True)
                score_rows = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                st.dataframe(pd.DataFrame([{"Class": LABEL_META[name][0], "Model score": float(score)} for name, score in score_rows]), hide_index=True, use_container_width=True)
                if evidence:
                    st.caption("Learned English word evidence: " + ", ".join(item["token"] for item in evidence[:6]))
                if not trained:
                    st.info("The main English model is an illustrative demo until the HateXplain training script has been run.")
            elif effective_language in {"Hindi", "Marathi", "English-Hindi code-mixed"}:
                st.info("OCR text is ready, but this language model is not trained in this project yet. HASOC files require a registration password; run the training command above and enter that dataset password when prompted.")
            elif effective_language in {"Gujarati", "Bodo"}:
                st.info("OCR text is ready. Gujarati and Bodo meme datasets are listed in HASOC 2025, but this project does not have those training files yet, so it will not guess a label.")
            elif language_choice == "Other / unsure" and detect_language(clean_text).startswith("Bengali"):
                st.info("Bengali meme analysis is excluded. Choose English, Hindi or another supported task when that matches the image text.")
            else:
                st.info("OCR text is ready. Choose English, Hindi, Marathi or English-Hindi code-mixed only when it matches the content and that model is available.")

elif page == "Feedback lab":
    st.markdown('<div class="eyebrow">Reward-guided prototype</div>', unsafe_allow_html=True)
    st.title("Feedback lab")
    st.markdown("A reviewer correction acts as a learning signal. This page shows the local queue used to reweight labeled examples during retraining.")
    f1, f2, f3 = st.columns(3)
    f1.metric("Corrections queued", len(feedback))
    f2.metric("Training weight", "2×", "compared with a benchmark training example")
    f3.metric("Update type", "Supervised", "not LLM reward training")
    if feedback.empty:
        st.info("No reviewer corrections yet. Use Post review to add one.")
    else:
        st.dataframe(feedback[["label", "reviewer_note", "created_at"]].rename(columns={"label": "Reviewer label", "reviewer_note": "Note", "created_at": "Saved (UTC)"}), hide_index=True, use_container_width=True)
        st.markdown('<div class="note">Text is not shown in this queue view. Local corrections are used as additional training examples; they are not used for final test evaluation.</div>', unsafe_allow_html=True)
    st.markdown("#### What a true reward-optimized LLM study would add")
    st.markdown("Preference pairs, a chosen multilingual language model, a documented reward or preference-learning method, and a separate evaluation before and after the update. This prototype records supervised corrections; it does not claim to train or align an LLM.")


elif page == "Benchmark":
    st.markdown('<div class="eyebrow">Evaluation</div>', unsafe_allow_html=True)
    st.title("HateXplain benchmark report")
    if not metrics:
        st.info("No benchmark report yet. Run `python scripts/train_model.py` to download the official files, train the baseline and create the report.")
        st.stop()
    st.caption(metrics.get("note", ""))
    a, b, c = st.columns(3)
    a.metric("Macro F1", f'{metrics["macro_f1"]:.3f}')
    b.metric("Accuracy", f'{metrics.get("accuracy", 0):.3f}')
    c.metric("Held-out examples", f'{metrics["test_examples"]:,}')
    st.markdown("#### Confusion matrix")
    matrix = metrics["confusion_matrix"]
    labels = [LABEL_META[name][0] for name in metrics["classes"]]
    figure = go.Figure(data=go.Heatmap(z=matrix, x=labels, y=labels, colorscale=[[0, "#152833"], [.5, "#287a70"], [1, "#66dfb4"]], text=matrix, texttemplate="%{text}", showscale=False))
    figure.update_layout(xaxis_title="Predicted", yaxis_title="Human label", height=360, margin=dict(l=0, r=0, t=12, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cedbe0")
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.markdown("#### Per-class scores")
    class_rows = []
    for name in metrics["classes"]:
        result = metrics["classification_report"].get(name, {})
        class_rows.append({"Class": LABEL_META[name][0], "Precision": result.get("precision", 0), "Recall": result.get("recall", 0), "F1": result.get("f1-score", 0), "Support": result.get("support", 0)})
    st.dataframe(pd.DataFrame(class_rows).style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}), hide_index=True, use_container_width=True)
    target_slices = metrics.get("target_slices", {})
    if target_slices:
        with st.expander("Descriptive target-community slices"):
            st.dataframe(pd.DataFrame([{"Annotated target": key, **value} for key, value in target_slices.items()]).sort_values("examples", ascending=False), hide_index=True, use_container_width=True)
            st.caption("Slice sizes vary and annotations may be incomplete. These descriptive scores do not establish fairness.")


else:
    st.markdown('<div class="eyebrow">Data and study design</div>', unsafe_allow_html=True)
    st.title("Dataset & methods")
    st.markdown("#### Primary dataset: HateXplain")
    st.markdown("The project uses the official three-class benchmark (normal, offensive, hate), crowd target annotations and token-level rationale spans. The official training, validation and test divisions are retained.")
    st.link_button("Open the official dataset repository", "https://github.com/hate-alert/HateXplain")
    st.warning("The source may contain hateful or offensive text. The app does not display individual dataset posts. Training reads those examples locally.")
    st.markdown("#### Model")
    st.markdown("Word and character TF-IDF features feed a class-balanced logistic regression classifier. Common chat shorthand receives a conservative whole-word expansion before training and prediction; the review screen shows the expanded text. The evidence chart shows word features with positive learned contribution to the selected class. This is a readable baseline, not a contextual language model.")
    st.markdown("#### Separate extensions")
    st.markdown("HASOC English, Hindi, Marathi and English-Hindi code-mixed examples train a separate broad harmful-content text classifier. The meme screen extracts text locally with OCR and passes the reviewed text to the selected language model; it does not classify image visuals. Gujarati and Bodo appear in the HASOC 2025 meme task, but their training files have not been added to this project yet. Results stay separate because the datasets use different labels and tasks.")
    st.link_button("Open the official HASOC dataset page", "https://hasocfire.github.io/hasoc/2021/dataset.html")
    st.link_button("Open the HASOC 2025 meme dataset page", "https://hasocfire.github.io/hasoc/2025/dataset.html")
    st.caption("The HASOC 2021 archives require a password obtained through dataset registration. It is separate from an API key; the training script asks for it locally.")
    st.markdown("#### Brief coverage")
    st.dataframe(pd.DataFrame([
        {"Brief requirement": "Explainable predictions", "Prototype status": "Implemented", "Evidence": "Token contributions; human rationale annotations retained in data pipeline"},
        {"Brief requirement": "Feedback / reward", "Prototype status": "Partially implemented", "Evidence": "Reviewer corrections reweighted at next supervised retrain; no LLM reward optimization"},
        {"Brief requirement": "Multilingual and code-mixed", "Prototype status": "Workflow ready; dataset access required", "Evidence": "HASOC English, Hindi, Marathi and English-Hindi task; password-protected data; no result until trained"},
        {"Brief requirement": "Meme image and text", "Prototype status": "OCR-assisted review", "Evidence": "Local OCR extracts meme text; the selected language text model reviews those words. No visual image classifier"},
        {"Brief requirement": "Slang and chat shorthand", "Prototype status": "Implemented", "Evidence": "Visible whole-token normalization preview; expansions can be reviewed against original"},
        {"Brief requirement": "Conversation context", "Prototype status": "Not implemented", "Evidence": "Both model pathways analyze one post or meme at a time"},
        {"Brief requirement": "Responsible use", "Prototype status": "Implemented", "Evidence": "Research-only notice, held-out report and subgroup caveats"},
    ]), hide_index=True, use_container_width=True)
    st.markdown("#### Dataset profile")
    profile_path = ROOT / "data" / "raw" / "dataset.json"
    if profile_path.exists():
        try:
            with st.spinner("Summarizing dataset labels…"):
                frame = load_hatexplain()
            counts = frame.groupby(["split", "label"]).size().reset_index(name="posts")
            profile = px.bar(counts, x="split", y="posts", color="label", barmode="stack", color_discrete_map={"normal": "#55d6aa", "offensive": "#f2a65a", "hate": "#f06b76"})
            profile.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cedbe0", height=330, margin=dict(l=0, r=0, t=12, b=0), legend_title_text="Class")
            st.plotly_chart(profile, use_container_width=True, config={"displayModeBar": False})
        except Exception as exc:
            st.warning(f"Dataset files are present but could not be read: {exc}")
    else:
        st.info("Dataset files are not downloaded. Run the training command in the README to fetch and profile them.")
    st.markdown("#### Reproduce the benchmark")
    st.code("python -m pip install -r requirements.txt\npython scripts/train_model.py\nstreamlit run app.py", language="bash")
    st.markdown("#### Train the extensions separately")
    st.code("python scripts/train_multilingual.py", language="bash")
    st.caption("HASOC downloads require the organizer-provided dataset password. Data and trained artifacts stay on this computer and are not included in the project files.")
