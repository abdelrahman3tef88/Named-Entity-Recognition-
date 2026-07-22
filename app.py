from __future__ import annotations

import html
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


APP_TITLE = "Named Entity Recognition using DistilBERT"
MODEL_DIR = Path(__file__).resolve().parent / "Best_DistilBERT_Model"
REQUIRED_FILES = {
    "Model": "model.safetensors",
    "Config": "config.json",
    "Tokenizer": "tokenizer.json",
    "Tokenizer Config": "tokenizer_config.json",
    "idx2tag": "idx2tag.pkl",
}
EXAMPLE_TEXT = "John works at Google in London. Ahmed joined Microsoft in Cairo for the FIFA technology summit."

ENTITY_COLORS = {
    "PER": "#2563eb",
    "ORG": "#16a34a",
    "LOC": "#f97316",
    "MISC": "#9333ea",
    "O": "#111827",
}
ENTITY_NAMES = {
    "PER": "Persons",
    "ORG": "Organizations",
    "LOC": "Locations",
    "MISC": "Miscellaneous",
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-soft: #f8fafc;
                --ink: #111827;
                --muted: #64748b;
                --line: #e2e8f0;
                --brand: #2563eb;
            }
            .main .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            h1, h2, h3 {
                letter-spacing: 0;
            }
            .hero {
                padding: 1.5rem 1.6rem;
                border: 1px solid var(--line);
                border-radius: 18px;
                background:
                    radial-gradient(circle at 12% 10%, rgba(37,99,235,.10), transparent 30%),
                    linear-gradient(135deg, #ffffff 0%, #f8fafc 55%, #eef2ff 100%);
                box-shadow: 0 18px 45px rgba(15, 23, 42, .06);
                margin-bottom: 1.1rem;
                animation: fadeIn .45s ease-out;
            }
            .hero h1 {
                font-size: clamp(2rem, 4vw, 3.1rem);
                line-height: 1.08;
                margin: 0 0 .65rem 0;
                color: var(--ink);
            }
            .hero p {
                max-width: 850px;
                color: var(--muted);
                font-size: 1.05rem;
                margin: 0;
            }
            .metric-card {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem;
                background: #ffffff;
                box-shadow: 0 10px 28px rgba(15, 23, 42, .055);
                min-height: 104px;
            }
            .metric-card .label {
                color: var(--muted);
                font-size: .86rem;
                margin-bottom: .35rem;
            }
            .metric-card .value {
                color: var(--ink);
                font-size: 2rem;
                line-height: 1;
                font-weight: 750;
            }
            .entity-box {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                background: #ffffff;
                min-height: 150px;
            }
            .entity-title {
                font-weight: 750;
                color: var(--ink);
                margin-bottom: .65rem;
            }
            .entity-chip {
                display: inline-flex;
                align-items: center;
                gap: .35rem;
                margin: .2rem .25rem .2rem 0;
                padding: .38rem .65rem;
                border-radius: 999px;
                color: #ffffff;
                font-size: .9rem;
                font-weight: 650;
            }
            .highlight-panel {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem;
                background: #ffffff;
                line-height: 2.15;
                font-size: 1.02rem;
            }
            .entity-token {
                color: #ffffff;
                border-radius: 8px;
                padding: .18rem .38rem;
                margin: 0 .04rem;
                font-weight: 700;
                white-space: nowrap;
            }
            .entity-label {
                opacity: .9;
                font-size: .7rem;
                margin-left: .28rem;
            }
            .status-ok {
                color: #15803d;
                font-weight: 700;
            }
            .status-bad {
                color: #b91c1c;
                font-weight: 700;
            }
            div[data-testid="stSidebar"] {
                background: #f8fafc;
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 999px;
                font-weight: 700;
                border: 1px solid #dbe4ef;
            }
            .stButton > button[kind="primary"] {
                background: var(--brand);
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def file_statuses() -> dict[str, bool]:
    return {label: (MODEL_DIR / filename).exists() for label, filename in REQUIRED_FILES.items()}


def first_existing_notebook() -> Path | None:
    candidates = [
        MODEL_DIR / "ner-transformer-distilbert.ipynb",
        MODEL_DIR / "ner_transformer_distilbert.ipynb",
    ]
    return next((path for path in candidates if path.exists()), None)


def extract_notebook_setting(name: str, default: int) -> int:
    notebook_path = first_existing_notebook()
    if notebook_path is None:
        return default
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return default
    pattern = re.compile(rf"\b{name}\s*=\s*(\d+)")
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        match = pattern.search(source)
        if match:
            return int(match.group(1))
    return default


def load_idx2tag() -> dict[int, str]:
    path = MODEL_DIR / "idx2tag.pkl"
    with path.open("rb") as handle:
        raw_mapping = pickle.load(handle)
    return {int(key): str(value) for key, value in raw_mapping.items()}


@st.cache_resource(show_spinner=False)
def load_resources() -> tuple[Any, Any, dict[int, str], int]:
    missing = [name for name, ok in file_statuses().items() if not ok]
    if missing:
        raise FileNotFoundError(f"Missing required resource(s): {', '.join(missing)}")

    idx2tag = load_idx2tag()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()

    model_max = getattr(tokenizer, "model_max_length", 512) or 512
    if model_max > 100_000:
        model_max = 512
    max_length = extract_notebook_setting("MAX_LEN", min(int(model_max), 512))
    return model, tokenizer, idx2tag, max_length


def normalize_label(label: str) -> str:
    if label == "<pad>":
        return "O"
    if label.startswith(("B-", "I-")):
        return label
    return "O" if label.upper() == "O" else label


def entity_type(label: str) -> str:
    label = normalize_label(label)
    return label.split("-", 1)[1] if "-" in label else "O"


def run_prediction(text: str) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    model, tokenizer, idx2tag, max_length = load_resources()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.inference_mode():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    rows: list[dict[str, Any]] = []
    token_spans: list[dict[str, Any]] = []
    for token, offset, token_probs in zip(tokens, offsets, probabilities):
        start, end = int(offset[0]), int(offset[1])
        if start == end:
            continue
        label_idx = int(np.argmax(token_probs))
        label = normalize_label(idx2tag.get(label_idx, f"LABEL_{label_idx}"))
        confidence = float(token_probs[label_idx])
        clean_token = text[start:end]
        rows.append(
            {
                "Token": clean_token,
                "Predicted Label": label,
                "Confidence Score": confidence,
            }
        )
        token_spans.append(
            {
                "token": clean_token,
                "label": label,
                "type": entity_type(label),
                "start": start,
                "end": end,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(rows), token_spans, text


def merge_entities(token_spans: list[dict[str, Any]], text: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {key: [] for key in ENTITY_NAMES}
    active_type: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def flush() -> None:
        nonlocal active_type, active_start, active_end
        if active_type and active_start is not None and active_end is not None:
            value = text[active_start:active_end].strip()
            if value and value not in grouped[active_type]:
                grouped[active_type].append(value)
        active_type = None
        active_start = None
        active_end = None

    for span in token_spans:
        label = span["label"]
        typ = span["type"]
        if typ == "O" or typ not in grouped:
            flush()
            continue
        is_begin = label.startswith("B-")
        if active_type == typ and not is_begin and active_end is not None and span["start"] <= active_end + 2:
            active_end = span["end"]
        else:
            flush()
            active_type = typ
            active_start = span["start"]
            active_end = span["end"]
    flush()
    return grouped


def highlighted_text(text: str, token_spans: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    cursor = 0
    entities: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    for span in token_spans:
        typ = span["type"]
        if typ == "O":
            if active:
                entities.append(active)
                active = None
            continue
        starts_entity = span["label"].startswith("B-")
        if active and active["type"] == typ and not starts_entity and span["start"] <= active["end"] + 2:
            active["end"] = span["end"]
        else:
            if active:
                entities.append(active)
            active = {"type": typ, "label": typ, "start": span["start"], "end": span["end"]}
    if active:
        entities.append(active)

    for entity in entities:
        start, end = entity["start"], entity["end"]
        if start < cursor:
            continue
        parts.append(html.escape(text[cursor:start]))
        color = ENTITY_COLORS.get(entity["type"], ENTITY_COLORS["MISC"])
        label = html.escape(entity["label"])
        token = html.escape(text[start:end])
        parts.append(
            f"<span class='entity-token' style='background:{color}'>{token}"
            f"<span class='entity-label'>{label}</span></span>"
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "<div class='highlight-panel'>" + "".join(parts).replace("\n", "<br>") + "</div>"


def stats_from_predictions(rows: pd.DataFrame, entities: dict[str, list[str]]) -> dict[str, int]:
    return {
        "Total Tokens": int(len(rows)),
        "Total Entities": int(sum(len(values) for values in entities.values())),
        "Persons": int(len(entities["PER"])),
        "Organizations": int(len(entities["ORG"])),
        "Locations": int(len(entities["LOC"])),
        "Miscellaneous": int(len(entities["MISC"])),
    }


def render_metric_card(label: str, value: int) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    statuses = file_statuses()
    with st.sidebar:
        st.subheader("Project Information")
        st.caption("Inference-only DistilBERT deployment")
        st.divider()
        st.write("**Model Type**")
        st.write("DistilBERT")
        st.write("**Task**")
        st.write("Named Entity Recognition")

        label_count = "Unavailable"
        try:
            label_count = str(len([v for v in load_idx2tag().values() if v != "<pad>"]))
        except Exception:
            pass
        st.write("**Number of Labels**")
        st.write(label_count)

        st.divider()
        loaded_names = {
            "Loaded Model Status": statuses["Model"] and statuses["Config"],
            "Loaded Tokenizer Status": statuses["Tokenizer"] and statuses["Tokenizer Config"],
            "Loaded idx2tag Status": statuses["idx2tag"],
        }
        for label, ok in loaded_names.items():
            klass = "status-ok" if ok else "status-bad"
            icon = "✓" if ok else "✕"
            text = "Loaded" if ok else "Missing"
            st.markdown(f"<span class='{klass}'>{icon} {label}: {text}</span>", unsafe_allow_html=True)


def initialize_state() -> None:
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""


def clear_text() -> None:
    st.session_state.input_text = ""


def set_example() -> None:
    st.session_state.input_text = EXAMPLE_TEXT


def render_downloads(rows: pd.DataFrame, entities: dict[str, list[str]]) -> None:
    export = rows.copy()
    if not export.empty:
        export["Confidence Score"] = export["Confidence Score"].map(lambda value: f"{value:.4f}")
    payload = {
        "predictions": rows.to_dict(orient="records"),
        "entities": entities,
    }
    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button(
            "⬇️ Download CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="ner_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_json:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name="ner_predictions.json",
            mime="application/json",
            use_container_width=True,
        )


def render_results(rows: pd.DataFrame, token_spans: list[dict[str, Any]], text: str) -> None:
    if rows.empty:
        st.warning("No valid tokens were produced for this input. Try a sentence with words or names.")
        return

    entities = merge_entities(token_spans, text)
    stats = stats_from_predictions(rows, entities)

    st.subheader("Statistics Dashboard")
    metric_cols = st.columns(6)
    for col, (label, value) in zip(metric_cols, stats.items()):
        with col:
            render_metric_card(label, value)

    st.subheader("Prediction Table")
    display_rows = rows.copy()
    display_rows["Confidence Score"] = display_rows["Confidence Score"].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    st.subheader("Colored Text Visualization")
    st.markdown(highlighted_text(text, token_spans), unsafe_allow_html=True)

    st.subheader("Extracted Entities")
    entity_cols = st.columns(4)
    for col, (typ, title) in zip(entity_cols, ENTITY_NAMES.items()):
        with col:
            chips = "".join(
                f"<span class='entity-chip' style='background:{ENTITY_COLORS[typ]}'>{html.escape(entity)}</span>"
                for entity in entities[typ]
            )
            if not chips:
                chips = "<span style='color:#64748b'>No entities found</span>"
            st.markdown(
                f"<div class='entity-box'><div class='entity-title'>{title}</div>{chips}</div>",
                unsafe_allow_html=True,
            )

    chart_col, dist_col = st.columns(2)
    with chart_col:
        st.subheader("Confidence Visualization")
        confidence_chart = rows.copy()
        confidence_chart["Confidence"] = confidence_chart["Confidence Score"] * 100
        fig = px.bar(
            confidence_chart,
            x="Token",
            y="Confidence",
            color="Predicted Label",
            text=confidence_chart["Confidence"].map(lambda value: f"{value:.1f}%"),
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(yaxis_range=[0, 100], margin=dict(l=10, r=10, t=20, b=10), height=410)
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with dist_col:
        st.subheader("Entity Distribution")
        distribution = Counter(span["type"] for span in token_spans if span["type"] in ENTITY_NAMES)
        if distribution:
            dist_df = pd.DataFrame(
                [{"Entity Type": ENTITY_NAMES[key], "Count": value} for key, value in distribution.items()]
            )
            fig = px.pie(
                dist_df,
                names="Entity Type",
                values="Count",
                hole=0.42,
                template="plotly_white",
                color="Entity Type",
                color_discrete_map={
                    "Persons": ENTITY_COLORS["PER"],
                    "Organizations": ENTITY_COLORS["ORG"],
                    "Locations": ENTITY_COLORS["LOC"],
                    "Miscellaneous": ENTITY_COLORS["MISC"],
                },
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=410)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No named entities were detected, so there is no distribution to plot yet.")

    st.subheader("Download Results")
    render_downloads(rows, entities)


def main() -> None:
    inject_styles()
    initialize_state()
    render_sidebar()

    st.markdown(
        f"""
        <section class="hero">
            <h1>{APP_TITLE}</h1>
            <p>
                A polished inference-only application that uses the saved DistilBERT token classification model
                to identify people, organizations, locations, and miscellaneous named entities in natural language text.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.text_area(
            "Input Text",
            key="input_text",
            height=190,
            placeholder="Type or paste your text here...",
            label_visibility="collapsed",
        )
        col_predict, col_clear, col_example = st.columns([1, 1, 1])
        with col_predict:
            predict = st.button("🔎 Predict", type="primary", use_container_width=True)
        with col_clear:
            st.button("🧹 Clear", on_click=clear_text, use_container_width=True)
        with col_example:
            st.button("✨ Example Sentence", on_click=set_example, use_container_width=True)

    if predict:
        text = st.session_state.input_text.strip()
        if not text:
            st.warning("Please enter text before running prediction.")
            return
        try:
            with st.spinner("Running DistilBERT inference..."):
                rows, token_spans, original_text = run_prediction(text)
            render_results(rows, token_spans, original_text)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
