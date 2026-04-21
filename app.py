import html
import json
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

from services.nutrition_math import (
    generate_glucose_curve,
    compute_metrics,
    blend_curves,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is not set. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=api_key)

st.set_page_config(
    page_title="FutureFork",
    page_icon="🍽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Clinical styling — off-white paper, serif hero, thin hairlines
# ---------------------------------------------------------------------------
CLINICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #f5f2ec;
    --paper: #fbf9f4;
    --ink: #1a1f24;
    --ink-2: #5a6168;
    --ink-3: #8a9099;
    --line: #e4ddd0;
    --line-2: #efe9dc;
    --accent: #0b6b6b;
    --accent-soft: #d8e8e7;
}

.stApp, .stApp > header { background: var(--bg) !important; }
.stApp { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
.main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

/* Hide the default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Typography */
.ff-hero {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 3.6rem;
    line-height: 1;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin: 0;
}
.ff-hero em { color: var(--accent); font-style: italic; }

.ff-kicker {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin: 0 0 14px 0;
}

.ff-lede {
    font-size: 15px;
    color: var(--ink-2);
    line-height: 1.6;
    max-width: 520px;
    margin-top: 1rem;
}

.ff-sidenote { font-size: 12px; color: var(--ink-3); line-height: 1.8; text-align: right; }

/* Cards */
.ff-card { background: var(--paper); border: 1px solid var(--line); padding: 22px 26px; margin-bottom: 20px; }
.ff-card-image { background: var(--paper); border: 1px solid var(--line); padding: 14px; margin-bottom: 20px; }

/* Identification card bits */
.ff-food-name { font-family: 'Instrument Serif', Georgia, serif; font-size: 28px; letter-spacing: -0.01em; }
.ff-weight { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
.ff-ingredient-label { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 4px; }
.ff-ingredient-value { font-size: 13px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.ff-hairline { height: 1px; background: var(--line); margin: 18px 0 14px; }
.ff-confidence-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--ink-2); }
.ff-mono-num { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 14px; }
.ff-confidence-bar { height: 2px; background: var(--line); margin-top: 8px; position: relative; }
.ff-confidence-fill { position: absolute; left: 0; top: 0; height: 100%; background: var(--accent); }
.ff-note { font-size: 11px; color: var(--ink-3); margin-top: 10px; line-height: 1.5; }

/* Metrics strip */
.ff-metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; }
.ff-metric-cell { padding: 0 20px; border-right: 1px solid var(--line); }
.ff-metric-cell:first-child { padding-left: 0; }
.ff-metric-cell:last-child { border-right: none; }
.ff-metric-label { font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 10px; }
.ff-metric-value { font-family: 'Instrument Serif', Georgia, serif; font-size: 38px; line-height: 1; font-variant-numeric: tabular-nums; }
.ff-metric-sub { font-size: 11px; color: var(--ink-3); margin-top: 4px; }

/* What if */
.ff-swap-headline { font-family: 'Instrument Serif', Georgia, serif; font-size: 22px; letter-spacing: -0.01em; }
.ff-swap-headline em { color: var(--accent); font-style: italic; }
.ff-swap-delta { font-family: 'JetBrains Mono', monospace; font-size: 22px; color: var(--accent); font-variant-numeric: tabular-nums; }

/* Legend line under chart */
.ff-legend { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 11px; color: var(--ink-3); }
.ff-legend-item { display: inline-flex; align-items: center; gap: 6px; margin-right: 18px; }
.ff-legend-solid { width: 14px; height: 2px; background: var(--accent); display: inline-block; }
.ff-legend-dashed { width: 14px; border-top: 1px dashed var(--ink); display: inline-block; }

/* Streamlit widget overrides */
[data-testid="stFileUploader"] section { background: var(--paper); border: 1px solid var(--line); border-radius: 0; }

/* Camera "Take photo" button */
[data-testid="stCameraInput"] button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 2px !important;
    font-size: 13px !important;
    letter-spacing: 0.02em;
    padding: 8px 16px;
}
[data-testid="stCameraInput"] button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: var(--paper) !important;
}

/* File uploader "Browse files" button */
[data-testid="stFileUploader"] button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 2px !important;
    font-size: 13px !important;
    letter-spacing: 0.02em;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: var(--paper) !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 0;
    padding: 8px 16px;
    font-size: 13px;
    color: var(--ink-2);
}
.stTabs [aria-selected="true"] { background: var(--ink) !important; color: var(--paper) !important; border-color: var(--ink) !important; }

.stButton > button {
    background: var(--paper);
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 2px;
    font-size: 13px;
    letter-spacing: 0.02em;
    padding: 8px 16px;
    font-weight: 400;
}
.stButton > button:hover { border-color: var(--ink); background: var(--paper); color: var(--ink); }

.stSlider [data-baseweb="slider"] > div > div > div { background: var(--accent) !important; }
.stSlider [role="slider"] { background: var(--accent) !important; border-color: var(--accent) !important; }

/* Swap section container */
[data-testid="stVerticalBlock"]:has(> [data-testid="stMarkdownContainer"] .ff-swap-section) {
    background: var(--paper);
    border: 1px solid var(--line);
    padding: 22px 26px;
}

/* Footer */
.ff-footer {
    display: flex; justify-content: space-between;
    padding: 20px 0;
    font-size: 11px; color: var(--ink-3);
    border-top: 1px solid var(--line);
    margin-top: 40px;
}
.ff-footer .ff-mono { font-family: 'JetBrains Mono', monospace; }
</style>
"""
st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


@st.cache_resource
def get_gemini_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        safety_settings=SAFETY_SETTINGS,
    )


ANALYSIS_PROMPT = """You are an expert Indian nutritionist. Analyze this food image and respond with STRICTLY a single JSON object (no markdown fences, no prose, no explanations outside the JSON).

Schema:
{
  "food_name": "string — the dish name",
  "total_weight_g": integer — estimated total plate weight in grams,
  "ingredients": [
    {"name": "string — short component name, e.g. 'Rice · basmati' or 'Mixed vegetables'", "weight_g": integer}
  ],
  "glycemic_index": integer 0-100,
  "confidence": integer 0-100 — how sure you are of the identification,
  "healthy_swap": {
    "description": "string — e.g. 'Swap basmati → foxtail millet biryani'",
    "swap_name": "string — name of the alternative dish",
    "glycemic_index": integer 0-100 — GI of the swapped version,
    "explanation": "string — 1-2 sentences, evidence-based, why the swap lowers glucose response"
  }
}

Rules:
- If no food is visible in the image (e.g. a face, empty room, object, scenery), return exactly this JSON: {"food_name": "No food detected", "glycemic_index": 0, "confidence": 0, "total_weight_g": 0, "ingredients": [], "healthy_swap": "None"}. Do not hallucinate ingredients.
- Provide 3 to 5 main ingredients, ordered by contribution.
- The swap must be culturally relevant (Indian cuisine preferred).
- Return ONLY the JSON object. No markdown. No commentary.
"""


def analyze_food(image_bytes: bytes) -> dict:
    """Send a food image to Gemini and return the structured analysis dict."""
    model = get_gemini_model()
    response = model.generate_content(
        [ANALYSIS_PROMPT, {"mime_type": "image/jpeg", "data": image_bytes}]
    )
    # Strip any accidental code fences just in case
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_glucose_chart(
    original: np.ndarray,
    blended: np.ndarray,
    show_original_dash: bool,
) -> go.Figure:
    """Plotly chart matching the clinical design: zone bands, teal curve, peak marker."""
    minutes = np.arange(len(blended))
    fig = go.Figure()

    # Background zone bands
    zones = [
        (70, 100, "#c7d9c0", "IN RANGE"),
        (100, 140, "#eccf9a", "ELEVATED"),
        (140, 165, "#e0a090", "HIGH"),
    ]
    for y0, y1, color, label in zones:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1, y0=y0, y1=y1,
            fillcolor=color, opacity=0.20,
            line=dict(width=0),
            layer="below",
        )
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.995, y=(y0 + y1) / 2,
            text=label, showarrow=False,
            font=dict(size=9, color="#5a6168", family="Inter"),
            xanchor="right",
        )

    # Original (dashed) — only drawn once the user actually starts swapping
    if show_original_dash:
        fig.add_trace(go.Scatter(
            x=minutes, y=original,
            mode="lines",
            line=dict(color="#1a1f24", width=1, dash="dash"),
            opacity=0.40,
            name="Original",
            hoverinfo="skip",
        ))

    # Blended (solid accent curve with faint fill)
    fig.add_trace(go.Scatter(
        x=minutes, y=blended,
        mode="lines",
        line=dict(color="#0b6b6b", width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(11, 107, 107, 0.07)",
        name="Predicted",
        hovertemplate="t=+%{x}min<br>%{y:.0f} mg/dL<extra></extra>",
    ))

    # Peak marker + label
    peak_idx = int(np.argmax(blended))
    peak_val = float(blended[peak_idx])
    fig.add_trace(go.Scatter(
        x=[peak_idx], y=[peak_val],
        mode="markers",
        marker=dict(size=9, color="#fbf9f4", line=dict(color="#0b6b6b", width=2)),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_annotation(
        x=peak_idx, y=peak_val + 7,
        text=f"peak {round(peak_val)} mg/dL",
        showarrow=False,
        font=dict(size=10, color="#0b6b6b", family="JetBrains Mono"),
    )

    fig.update_layout(
        height=340,
        margin=dict(l=50, r=30, t=30, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(
            range=[70, 165],
            tickvals=[70, 100, 130, 160],
            gridcolor="#e4ddd0",
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#8a9099"),
            ticksuffix="  ",
        ),
        xaxis=dict(
            range=[0, 120],
            tickvals=[0, 30, 60, 90, 120],
            ticktext=["+0'", "+30'", "+60'", "+90'", "+120'"],
            gridcolor="#e4ddd0",
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#8a9099"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
hero_left, hero_right = st.columns([3, 1])
with hero_left:
    st.markdown('<div class="ff-kicker">Metabolic Preview · Meal analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="ff-hero">See your glucose future<br/><em>before you eat.</em></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="ff-lede">A photograph, analyzed by our metabolic model. We estimate the '
        "two-hour glucose response and suggest evidence-based alternatives.</p>",
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        '<div class="ff-sidenote" style="margin-top: 56px;">'
        "Model v4.2 · Calibrated<br/>"
        "Personal baseline: 90 mg/dL<br/>"
        "Time in range (7d): 84%"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="ff-hairline" style="margin: 32px 0;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main grid
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1.45], gap="large")

# ----- LEFT: Specimen + Identification -----
with left_col:
    st.markdown('<div class="ff-kicker">01 · Specimen</div>', unsafe_allow_html=True)

    # Only show capture UI if we don't yet have an analyzed image
    if st.session_state.image_bytes is None:
        tab_cam, tab_upload = st.tabs(["📷  Camera", "📤  Upload"])
        new_img = None
        with tab_cam:
            cam = st.camera_input("Capture food picture", label_visibility="collapsed")
            if cam:
                new_img = cam.getvalue()
        with tab_upload:
            up = st.file_uploader(
                "Upload food photo",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if up:
                new_img = up.getvalue()

        if new_img:
            if len(new_img) > MAX_IMAGE_BYTES:
                st.error(
                    f"Image is too large ({len(new_img) / (1024 * 1024):.1f} MB). "
                    "Please use an image under 10 MB."
                )
                st.stop()
            st.session_state.image_bytes = new_img
            st.session_state.last_image_hash = hash(new_img)
            with st.spinner("Analyzing metabolic impact..."):
                try:
                    st.session_state.analysis = analyze_food(new_img)
                except google_exceptions.InvalidArgument:
                    st.error("The image could not be processed. Please try a clearer photo.")
                    st.session_state.image_bytes = None
                    st.stop()
                except google_exceptions.PermissionDenied:
                    st.error("API key is invalid or lacks permissions.")
                    st.session_state.image_bytes = None
                    st.stop()
                except google_exceptions.ResourceExhausted:
                    st.error("API rate limit reached. Please wait and try again.")
                    st.session_state.image_bytes = None
                    st.stop()
                except google_exceptions.GoogleAPICallError:
                    st.error("Could not reach the analysis service.")
                    st.session_state.image_bytes = None
                    st.stop()
                except (json.JSONDecodeError, ValueError):
                    st.error("Failed to parse the AI response. Please retake the photo.")
                    st.session_state.image_bytes = None
                    st.stop()
            st.rerun()
    else:
        # Show the captured image in a paper-card
        food_label = "Captured food image"
        if st.session_state.analysis:
            food_label = st.session_state.analysis.get("food_name", food_label)
        st.markdown('<div class="ff-card-image">', unsafe_allow_html=True)
        st.image(st.session_state.image_bytes, caption=food_label, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_retake, col_reup = st.columns(2)
        with col_retake:
            if st.button("Retake", use_container_width=True, key="retake"):
                st.session_state.image_bytes = None
                st.session_state.analysis = None
                st.session_state.last_image_hash = None
                if "swap_factor" in st.session_state:
                    del st.session_state["swap_factor"]
                st.rerun()
        with col_reup:
            if st.button("Clear session", use_container_width=True, key="clear"):
                for k in ["image_bytes", "analysis", "last_image_hash", "swap_factor"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # Identification card
    if st.session_state.analysis:
        data = st.session_state.analysis
        food_name = data.get("food_name", "Unknown dish")
        no_food = food_name == "No food detected"

        st.markdown(
            '<div class="ff-kicker" style="margin-top: 28px;">02 · Identification</div>',
            unsafe_allow_html=True,
        )

        if no_food:
            st.markdown(
                '<div class="ff-card">'
                '<div style="padding: 20px; color: var(--ink-3); font-size: 13px;">'
                'No food was detected in this image. Please point the camera at a meal to see metabolic insights.'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # Build ingredients grid (up to 4 cells for neat 2x2)
            ingredients = data.get("ingredients", [])[:4]
            cells = ""
            for ing in ingredients:
                name = html.escape(str(ing.get("name", "—")))
                weight = int(ing.get("weight_g", 0))
                cells += (
                    f'<div>'
                    f'  <div class="ff-ingredient-label">{name}</div>'
                    f'  <div class="ff-ingredient-value">{weight} g</div>'
                    f'</div>'
                )

            confidence = min(max(int(data.get("confidence", 85)), 0), 100)
            total_weight = int(data.get("total_weight_g", sum(i.get("weight_g", 0) for i in ingredients)))
            safe_food_name = html.escape(food_name)

            identification_html = f"""
            <div class="ff-card" role="region" aria-label="Food identification">
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px;">
                    <div class="ff-food-name">{safe_food_name}</div>
                    <div class="ff-weight">~ {total_weight} g</div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px 24px;">
                    {cells}
                </div>
                <div class="ff-hairline" role="separator"></div>
                <div class="ff-confidence-row">
                    <span>Identification confidence</span>
                    <span class="ff-mono-num">{confidence}%</span>
                </div>
                <div class="ff-confidence-bar" role="progressbar" aria-valuenow="{confidence}" aria-valuemin="0" aria-valuemax="100" aria-label="Identification confidence {confidence}%">
                    <div class="ff-confidence-fill" style="width:{confidence}%;"></div>
                </div>
                <div class="ff-note">
                    Edit any component if this looks wrong — we recalibrate the response curve in real time.
                </div>
            </div>
            """
            st.markdown(identification_html, unsafe_allow_html=True)

# ----- RIGHT: Predicted Response + What If -----
with right_col:
    no_food_right = (
        not st.session_state.analysis
        or st.session_state.analysis.get("food_name") == "No food detected"
    )
    if no_food_right:
        st.markdown(
            '<div class="ff-card" style="text-align:center;padding:80px 40px;">'
            '<div class="ff-kicker">Awaiting specimen</div>'
            '<div class="ff-food-name" style="color:var(--ink-3);">Capture a meal to begin analysis</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        data = st.session_state.analysis
        original_gi = int(data.get("glycemic_index", 60))
        swap_gi = int(data.get("healthy_swap", {}).get("glycemic_index", max(0, original_gi - 20)))

        original_curve = generate_glucose_curve(original_gi)["Glucose (mg/dL)"].values
        swap_curve = generate_glucose_curve(swap_gi)["Glucose (mg/dL)"].values

        # swap_factor lives in session_state (written by the slider below)
        swap_factor = float(st.session_state.get("swap_factor", 0.0))

        blended = blend_curves(original_curve, swap_curve, swap_factor)
        metrics = compute_metrics(blended)
        baseline_metrics = compute_metrics(original_curve)

        blended_gi = round(original_gi * (1 - swap_factor) + swap_gi * swap_factor)
        if swap_factor > 0.02:
            gi_label = "swapped"
        elif blended_gi <= 55:
            gi_label = "low"
        elif blended_gi <= 69:
            gi_label = "medium"
        else:
            gi_label = "high"

        # ----- Metrics strip -----
        st.markdown('<div class="ff-kicker">03 · Predicted Response</div>', unsafe_allow_html=True)

        metrics_html = f"""
        <div class="ff-card" style="padding: 24px 28px 10px;" role="region" aria-label="Predicted glucose response metrics">
            <div class="ff-metrics-row">
                <div class="ff-metric-cell">
                    <div class="ff-metric-label">Glycemic Index</div>
                    <div class="ff-metric-value" aria-label="Glycemic index {blended_gi}">{blended_gi}</div>
                    <div class="ff-metric-sub">{gi_label}</div>
                </div>
                <div class="ff-metric-cell">
                    <div class="ff-metric-label">Est. Peak</div>
                    <div class="ff-metric-value" aria-label="Estimated peak {metrics["peak"]} milligrams per deciliter">{metrics["peak"]}</div>
                    <div class="ff-metric-sub">mg/dL</div>
                </div>
                <div class="ff-metric-cell">
                    <div class="ff-metric-label">2-hr AUC</div>
                    <div class="ff-metric-value" aria-label="Two hour area under curve {metrics["auc"]}">{metrics["auc"]}</div>
                    <div class="ff-metric-sub">mg&middot;min/dL</div>
                </div>
                <div class="ff-metric-cell">
                    <div class="ff-metric-label">Time to Peak</div>
                    <div class="ff-metric-value" aria-label="Time to peak {metrics["time_to_peak"]} minutes">{metrics["time_to_peak"]}</div>
                    <div class="ff-metric-sub">minutes</div>
                </div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

        # ----- Chart -----
        with st.container():
            fig = make_glucose_chart(
                original_curve, blended, show_original_dash=swap_factor > 0.02
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        legend_html = """
        <div class="ff-legend">
            <div>
                <span class="ff-legend-item"><span class="ff-legend-solid"></span> Predicted response</span>
                <span class="ff-legend-item"><span class="ff-legend-dashed"></span> Original meal</span>
            </div>
            <span class="ff-mono-num" style="font-size: 11px;">±11 mg/dL · 95% CI</span>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        # ----- What If -----
        swap_info = data.get("healthy_swap", {})
        description = html.escape(
            swap_info.get("description", "Swap to a lower-GI alternative")
        )
        explanation = html.escape(
            swap_info.get(
                "explanation",
                "A lower-GI alternative will produce a gentler glucose response.",
            )
        )
        peak_reduction = max(0, baseline_metrics["peak"] - metrics["peak"])

        with st.container():
            # Marker div for CSS :has() selector to apply --paper background
            st.markdown('<div class="ff-swap-section"></div>', unsafe_allow_html=True)

            # Title row: swap headline left, peak reduction right
            title_left, title_right = st.columns([2, 1])
            with title_left:
                st.markdown(
                    '<div class="ff-kicker" style="margin-bottom: 4px;">What if</div>'
                    f'<div class="ff-swap-headline">{description}</div>',
                    unsafe_allow_html=True,
                )
            with title_right:
                st.markdown(
                    f'<div style="text-align:right;padding-top:14px;">'
                    f'<div class="ff-swap-delta" aria-label="Estimated peak reduction {peak_reduction} milligrams per deciliter">\u2212{peak_reduction} mg/dL</div>'
                    f'<div style="font-size:12px;color:var(--ink-2);">estimated peak reduction</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Slider — native Streamlit, no HTML wrapper
            st.slider(
                "Swap blend — slide to preview the alternative meal",
                min_value=0.0,
                max_value=1.0,
                value=swap_factor,
                step=0.01,
                key="swap_factor",
                label_visibility="collapsed",
            )

            # Explanation
            st.markdown(
                f'<div class="ff-hairline" role="separator"></div>'
                f'<div style="font-size:16px;color:var(--ink-2);line-height:1.6;">{explanation}</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="ff-footer">'
    '<span>Not a medical device. Estimates are educational, derived from population-level models.</span>'
    '<span class="ff-mono">FUTUREFORK · v0.9.4</span>'
    "</div>",
    unsafe_allow_html=True,
)
