import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import base64
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
from torchvision import transforms

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Authentic",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analyzed_image" not in st.session_state:
    st.session_state.analyzed_image = None
if "result" not in st.session_state:
    st.session_state.result = None
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "history" not in st.session_state:
    st.session_state.history = []
if "model_type" not in st.session_state:
    st.session_state.model_type = "binary"  # "binary" or "multiclass"
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "single"  # "single", "compare", "batch"
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "compare_images" not in st.session_state:
    st.session_state.compare_images = [None, None]
if "compare_results" not in st.session_state:
    st.session_state.compare_results = [None, None]
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS - Theme aware
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
is_dark = st.session_state.theme == "dark"

# Theme colors
bg_primary = "#0b1121" if is_dark else "#f0f4fa"
bg_card = "#121d33" if is_dark else "#ffffff"
bg_hover = "#1a2744" if is_dark else "#e8eef6"
border_color = "#1e3055" if is_dark else "#c7d4e4"
text_primary = "#eef2f9" if is_dark else "#0b1a3e"
text_secondary = "#8a9bb8" if is_dark else "#4a5e80"
text_muted = "#5c7099" if is_dark else "#8a9bb8"
accent_color = "#3b82f6"  # bright blue accent for both themes
accent_hover = "#2563eb"  # darker blue for hover states

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Hide Streamlit chrome */
#MainMenu, footer, header, .stDeployButton,
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"], .stApp > header,
[data-testid="collapsedControl"], section[data-testid="stSidebar"] {{
    display: none !important;
}}

.stApp {{
    background: {bg_primary} !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.main .block-container {{
    padding: 0 2rem 3rem !important;
    max-width: 900px !important;
    margin-top: 0 !important;
}}

/* Force remove Streamlit's built-in top padding */
.stApp [data-testid="stAppViewContainer"] {{
    padding-top: 0 !important;
}}

.stApp [data-testid="stAppViewBlockContainer"] {{
    padding-top: 1rem !important;
}}

.stApp .main {{
    padding-top: 0 !important;
}}

.stApp > div:first-child {{
    padding-top: 0 !important;
}}

[data-testid="stAppViewContainer"] {{
    padding-top: 0 !important;
}}

[data-testid="stVerticalBlock"] {{
    gap: 0 !important;
}}

[data-testid="stAppViewContainer"] > section > div {{
    padding-top: 0 !important;
}}

/* Remove top spacing */
.main > div:first-child {{
    padding-top: 0 !important;
    margin-top: 0 !important;
}}

section.main > div {{
    padding-top: 0 !important;
}}

/* ═══════════════════════════════════════════
   NAVBAR
═══════════════════════════════════════════ */
.navbar-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid {border_color};
    margin-bottom: 8px;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 18px;
    color: {text_primary};
}}

.brand-logo {{
    width: 28px;
    height: 28px;
    background: {accent_color};
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 14px;
    font-weight: 600;
}}

/* Button styling */
.stButton > button {{
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    transition: all 0.15s !important;
    height: auto !important;
    min-height: 36px !important;
}}

.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {{
    background: transparent !important;
    color: {text_secondary} !important;
    border: none !important;
}}

.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {{
    background: {bg_hover} !important;
    color: {text_primary} !important;
}}

.stButton > button[kind="primary"] {{
    background: {accent_color} !important;
    color: #ffffff !important;
    border: none !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: {accent_hover} !important;
}}

/* ═══════════════════════════════════════════
   HERO SECTION
═══════════════════════════════════════════ */
.hero {{
    text-align: center;
    padding: 32px 0 24px;
}}

.hero-label {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: {'#0d2457' if is_dark else '#dbeafe'};
    border: 1px solid {'#1e3a6e' if is_dark else '#93c5fd'};
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    color: {accent_color};
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.hero-title {{
    font-size: 40px;
    font-weight: 700;
    color: {text_primary} !important;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 10px;
}}

.hero h1, .hero-title, h1 {{
    color: {text_primary} !important;
    -webkit-text-fill-color: {text_primary} !important;
}}

.hero-subtitle {{
    font-size: 16px !important;
    color: {text_secondary} !important;
    max-width: 520px !important;
    margin: 0 auto !important;
    line-height: 1.6 !important;
    text-align: center !important;
    display: block !important;
}}

.hero p.hero-subtitle {{
    text-align: center !important;
}}

/* How it works section */
.how-it-works {{
    display: flex;
    justify-content: center;
    gap: 32px;
    margin: 24px 0 8px;
    padding: 20px 0;
}}

.step {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    max-width: 140px;
}}

.step-number {{
    width: 32px;
    height: 32px;
    background: {accent_color};
    color: #ffffff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
}}

.step-text {{
    font-size: 13px;
    color: {text_secondary};
    text-align: center;
    line-height: 1.4;
}}

.step-arrow {{
    color: {text_muted};
    font-size: 20px;
    margin-top: 4px;
}}

/* ═══════════════════════════════════════════
   UPLOAD AREA
═══════════════════════════════════════════ */
[data-testid="stFileUploader"] > section {{
    background: {bg_card} !important;
    border: 2px dashed {border_color} !important;
    border-radius: 16px !important;
    padding: 28px !important;
}}

[data-testid="stFileUploader"] > section:hover {{
    border-color: {accent_color} !important;
    background: {bg_hover} !important;
}}

[data-testid="stFileUploader"] button {{
    background: {accent_color} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
}}

[data-testid="stFileUploader"] small {{
    color: {text_muted} !important;
}}

/* Fix file uploader text contrast */
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div {{
    color: {text_secondary} !important;
}}

[data-testid="stFileUploader"] section > div:first-child {{
    color: {text_primary} !important;
}}

/* Fix all text visibility */
p, span, div, label {{
    color: inherit;
}}

.stMarkdown p {{
    color: {text_secondary} !important;
}}

.divider {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 18px 0;
}}

.divider-line {{
    flex: 1;
    height: 1px;
    background: {border_color};
}}

.divider-text {{
    font-size: 12px;
    font-weight: 600;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Sample buttons */
.sample-btn > button {{
    width: 100% !important;
    background: {bg_card} !important;
    border: 1px solid {border_color} !important;
    color: {text_secondary} !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    transition: all 0.2s ease !important;
}}

.sample-btn > button:hover {{
    background: {bg_hover} !important;
    border-color: {accent_color} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.15) !important;
}}

/* ═══════════════════════════════════════════
   RESULT CARD
═══════════════════════════════════════════ */
.result-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 10px 40px -10px rgba(0,0,0,0.1);
    margin-top: 16px;
    animation: fadeInUp 0.4s ease-out;
}}

.result-image-wrap {{
    position: relative;
    background: {bg_hover};
    display: flex;
    justify-content: center;
    padding: 20px;
}}

.result-img {{
    max-width: 100%;
    height: auto;
    max-height: 380px;
    object-fit: contain;
    border-radius: 12px;
}}

.result-badge {{
    position: absolute;
    top: 28px;
    left: 28px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border-radius: 100px;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}

.result-badge.ai {{
    background: #ef4444;
    color: white;
}}

.result-badge.real {{
    background: #22c55e;
    color: white;
}}

.result-body {{
    padding: 24px;
}}

.result-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 20px;
}}

.result-title {{
    font-size: 24px;
    font-weight: 700;
    color: {text_primary};
    margin-bottom: 4px;
}}

.result-subtitle {{
    font-size: 14px;
    color: {text_muted};
}}

.confidence-display {{
    text-align: right;
}}

.confidence-number {{
    font-size: 30px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {text_primary};
    line-height: 1;
}}

.confidence-label {{
    font-size: 11px;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}}

.progress-bar {{
    height: 8px;
    background: {bg_hover};
    border-radius: 100px;
    overflow: hidden;
    margin-bottom: 20px;
}}

.progress-fill {{
    height: 100%;
    border-radius: 100px;
}}

.progress-fill.ai {{
    background: linear-gradient(90deg, #ef4444, #f97316);
}}

.progress-fill.real {{
    background: linear-gradient(90deg, #22c55e, #10b981);
}}

.stats-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}}

.stat-box {{
    background: {bg_hover};
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}}

.stat-value {{
    font-size: 16px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {text_primary};
}}

.stat-label {{
    font-size: 10px;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}}

/* ═══════════════════════════════════════════
   ABOUT PAGE
═══════════════════════════════════════════ */
.about-section {{
    padding: 24px 0;
}}

.about-header {{
    text-align: center;
    margin-bottom: 32px;
}}

.about-title {{
    font-size: 30px;
    font-weight: 700;
    color: {text_primary};
    margin-bottom: 8px;
}}

.about-desc {{
    font-size: 15px;
    color: {text_secondary};
    line-height: 1.7;
}}

.feature-card {{
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 20px;
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 14px;
    margin-bottom: 10px;
}}

.feature-icon {{
    width: 40px;
    height: 40px;
    min-width: 40px;
    background: {bg_hover};
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}}

.feature-content h3 {{
    font-size: 15px;
    font-weight: 600;
    color: {text_primary};
    margin: 0 0 4px 0;
}}

.feature-content p {{
    font-size: 13px;
    color: {text_secondary};
    margin: 0;
    line-height: 1.5;
}}

.tech-section {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 14px;
    padding: 24px;
    margin-top: 20px;
}}

.tech-title {{
    font-size: 15px;
    font-weight: 600;
    color: {text_primary};
    margin-bottom: 14px;
}}

.tech-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}

.tech-tag {{
    padding: 7px 12px;
    background: {bg_hover};
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: {text_secondary};
}}

/* History Section */
.history-section {{
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid {border_color};
}}

.history-title {{
    font-size: 14px;
    font-weight: 600;
    color: {text_secondary};
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.history-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}}

.history-item {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.history-thumb {{
    width: 48px;
    height: 48px;
    border-radius: 8px;
    object-fit: cover;
    flex-shrink: 0;
}}

.history-info {{
    flex: 1;
    min-width: 0;
}}

.history-label {{
    font-size: 12px;
    font-weight: 600;
    color: {text_primary};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.history-label.ai {{ color: #ef4444; }}
.history-label.real {{ color: #22c55e; }}

.history-conf {{
    font-size: 11px;
    color: {text_muted};
    font-family: 'JetBrains Mono', monospace;
}}

/* Loading Animation */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

@keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}

.loading-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 20px;
    overflow: hidden;
    margin-top: 16px;
}}

.loading-image {{
    height: 300px;
    background: linear-gradient(90deg, {bg_hover} 25%, {border_color} 50%, {bg_hover} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}}

.loading-body {{
    padding: 24px;
}}

.loading-title {{
    height: 28px;
    width: 60%;
    background: linear-gradient(90deg, {bg_hover} 25%, {border_color} 50%, {bg_hover} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 6px;
    margin-bottom: 12px;
}}

.loading-subtitle {{
    height: 16px;
    width: 40%;
    background: linear-gradient(90deg, {bg_hover} 25%, {border_color} 50%, {bg_hover} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
}}

.loading-bar {{
    height: 8px;
    background: linear-gradient(90deg, {bg_hover} 25%, {border_color} 50%, {bg_hover} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 100px;
    margin: 20px 0;
}}

/* Confidence interpretation */
.confidence-interpret {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: {bg_hover};
    border-radius: 10px;
    margin-top: 16px;
    animation: fadeInUp 0.4s ease-out 0.1s both;
}}

.interpret-icon {{
    font-size: 20px;
}}

.interpret-text {{
    font-size: 13px;
    color: {text_secondary};
    line-height: 1.5;
}}

.interpret-text strong {{
    color: {text_primary};
}}

/* Image preview with loading overlay */
.preview-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 20px;
    overflow: hidden;
    margin-top: 16px;
}}

.preview-image-wrap {{
    position: relative;
    background: {bg_hover};
    display: flex;
    justify-content: center;
    padding: 20px;
}}

.preview-img {{
    max-width: 100%;
    height: auto;
    max-height: 380px;
    object-fit: contain;
    border-radius: 12px;
    filter: brightness(0.7);
}}

.preview-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
}}

.preview-spinner {{
    width: 48px;
    height: 48px;
    border: 4px solid {border_color};
    border-top-color: {text_primary};
    border-radius: 50%;
    animation: spin 1s linear infinite;
}}

@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}

.preview-text {{
    font-size: 14px;
    font-weight: 600;
    color: white;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}}

/* Fade-in animation */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(20px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

/* Mode tabs */
.mode-tabs {{
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    padding: 4px;
    background: {bg_hover};
    border-radius: 12px;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
}}

.mode-tab {{
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: {text_secondary};
    cursor: pointer;
    transition: all 0.15s;
    border: none;
    background: transparent;
}}

.mode-tab:hover {{
    color: {text_primary};
}}

.mode-tab.active {{
    background: {bg_card};
    color: {text_primary};
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

/* URL input */
.url-input-wrap {{
    margin-top: 16px;
}}

/* Feedback button */
.feedback-btn {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: transparent;
    border: 1px solid {border_color};
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: {text_muted};
    cursor: pointer;
    transition: all 0.15s;
    margin-top: 12px;
}}

.feedback-btn:hover {{
    border-color: {text_secondary};
    color: {text_secondary};
}}

.feedback-success {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: #22c55e20;
    border: 1px solid #22c55e;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: #22c55e;
    margin-top: 12px;
}}

/* Comparison view */
.compare-container {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 16px;
}}

.compare-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 16px;
    overflow: hidden;
}}

.compare-image-wrap {{
    position: relative;
    background: {bg_hover};
}}

.compare-img {{
    width: 100%;
    height: 200px;
    object-fit: cover;
}}

.compare-body {{
    padding: 16px;
}}

.compare-badge {{
    position: absolute;
    top: 24px;
    left: 24px;
    padding: 6px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
}}

.compare-badge.ai {{
    background: #ef4444;
    color: white;
}}

.compare-badge.real {{
    background: #22c55e;
    color: white;
}}

.compare-title {{
    font-size: 16px;
    font-weight: 600;
    color: {text_primary};
    margin-bottom: 4px;
}}

.compare-conf {{
    font-size: 24px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {text_primary};
}}

/* Batch results */
.batch-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 16px;
}}

.batch-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 12px;
    overflow: hidden;
    animation: fadeInUp 0.3s ease-out;
}}

.batch-image {{
    width: 100%;
    height: 180px;
    object-fit: cover;
    background: {bg_hover};
    display: block;
}}

.batch-body {{
    padding: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.batch-label {{
    font-size: 13px;
    font-weight: 600;
}}

.batch-label.ai {{ color: #ef4444; }}
.batch-label.real {{ color: #22c55e; }}

.batch-conf {{
    font-size: 14px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: {text_primary};
}}

.batch-summary {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
    display: flex;
    justify-content: space-around;
    text-align: center;
}}

.summary-stat {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

.summary-number {{
    font-size: 28px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}

.summary-number.ai {{ color: #ef4444; }}
.summary-number.real {{ color: #22c55e; }}

.summary-label {{
    font-size: 12px;
    color: {text_muted};
    text-transform: uppercase;
}}

/* Responsive */
@media (max-width: 768px) {{
    .main .block-container {{
        padding: 1rem !important;
    }}
    .hero-title {{
        font-size: 28px;
    }}
    .stats-row {{
        grid-template-columns: 1fr;
    }}
    .how-it-works {{
        gap: 16px;
    }}
    .step-arrow {{
        font-size: 16px;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMG_SIZE = (128, 128)
MODEL_PATH = "models/basic_cnn.keras"
VIT_IMG_SIZE = 224

# ViT Model configurations
BINARY_MODEL_NAME = "gechen98/AI_image_classification"
MULTICLASS_MODEL_NAME = "gechen98/AI_image_generator_classification"
VIT_BASE_MODEL = "google/vit-base-patch16-224"

# Multiclass labels
MULTICLASS_LABELS = ['glide', 'midjourney', 'wukong', 'adm', 'sdv5', 'vqdm', 'biggan']

SAMPLES = [
    {"name": "Real Photo", "icon": "📷", "url": "https://images.unsplash.com/photo-1772307956262-42d4f7696876?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", "type": "real"},
    {"name": "AI Portrait", "icon": "🤖", "url": "https://raw.githubusercontent.com/Gechen989898/AI_Art_vs_Human_Art/master/ai-image.jpeg", "type": "ai"},
    {"name": "Real Dog", "icon": "🐕", "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400", "type": "real"},
]

# Device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_vit_binary_model():
    """Load the binary classification ViT model from Hugging Face"""
    model = ViTForImageClassification.from_pretrained(BINARY_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_vit_multiclass_model():
    """Load the multiclass classification ViT model from Hugging Face"""
    model = ViTForImageClassification.from_pretrained(MULTICLASS_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_vit_processor():
    """Load the ViT image processor"""
    return AutoImageProcessor.from_pretrained(VIT_BASE_MODEL)

def get_vit_transforms(processor):
    """Get transforms for ViT models"""
    mean = processor.image_mean
    std = processor.image_std
    return transforms.Compose([
        transforms.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def preprocess(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    return np.array(image).reshape((1, 128, 128, 3)) / 255.0

def preprocess_vit(image, transform):
    """Preprocess image for ViT model"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor

def predict(model, img_array):
    pred = model.predict(img_array, verbose=0)
    score = float(pred[0][0])
    if score < 0.5:
        return {"label": "AI Generated", "is_ai": True, "confidence": (1 - score) * 100, "raw": score}
    return {"label": "Real Image", "is_ai": False, "confidence": score * 100, "raw": score}

def predict_vit_binary(model, image_tensor):
    """Predict using binary ViT model"""
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = int(probs.argmax().item())
        confidence = float(probs[pred_id].item()) * 100
        
        # id2label: {0: 'ai', 1: 'nature'}
        is_ai = pred_id == 0
        label = "AI Generated" if is_ai else "Real Image"
        
        return {
            "label": label,
            "is_ai": is_ai,
            "confidence": confidence,
            "raw": float(probs[0].item()),  # AI probability
            "probs": probs.cpu().numpy()
        }

def predict_vit_multiclass(model, image_tensor):
    """Predict using multiclass ViT model to identify AI generator type"""
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = int(probs.argmax().item())
        confidence = float(probs[pred_id].item()) * 100
        
        # Get label from model config or use our list
        if hasattr(model.config, 'id2label') and model.config.id2label:
            pred_label = model.config.id2label[pred_id]
        else:
            pred_label = MULTICLASS_LABELS[pred_id]
        
        # Build all class probabilities
        all_probs = {}
        for i, prob in enumerate(probs.cpu().numpy()):
            if hasattr(model.config, 'id2label') and model.config.id2label:
                label = model.config.id2label[i]
            else:
                label = MULTICLASS_LABELS[i]
            all_probs[label] = float(prob) * 100
        
        return {
            "label": pred_label,
            "is_ai": True,  # All multiclass predictions are AI generators
            "confidence": confidence,
            "raw": float(probs[pred_id].item()),
            "all_probs": all_probs,
            "pred_id": pred_id
        }

def load_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

def img_to_b64(image, max_size=800):
    img = image.copy().convert("RGB")
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def get_confidence_interpretation(confidence, is_ai):
    """Return interpretation text and icon based on confidence level"""
    if confidence >= 90:
        icon = "✅"
        if is_ai:
            text = "<strong>Very High Confidence:</strong> This image shows strong indicators of AI generation. The model is highly certain about this classification."
        else:
            text = "<strong>Very High Confidence:</strong> This image shows strong characteristics of a real photograph. The model is highly certain about this classification."
    elif confidence >= 75:
        icon = "🟢"
        if is_ai:
            text = "<strong>High Confidence:</strong> This image likely contains AI-generated elements. Most visual patterns align with known AI generators."
        else:
            text = "<strong>High Confidence:</strong> This image appears to be a genuine photograph with natural characteristics."
    elif confidence >= 60:
        icon = "🟡"
        if is_ai:
            text = "<strong>Moderate Confidence:</strong> Some AI-like patterns detected, but the image may contain mixed elements or be ambiguous."
        else:
            text = "<strong>Moderate Confidence:</strong> The image appears real, but some elements are ambiguous. Consider reviewing manually."
    else:
        icon = "🟠"
        text = "<strong>Low Confidence:</strong> The model is uncertain about this image. The result should be interpreted with caution and manual review is recommended."
    
    return icon, text

def analyze_image(img, vit_transform, vit_binary_model, vit_multiclass_model):
    """Analyze an image and return results with generator info if AI"""
    tensor = preprocess_vit(img, vit_transform)
    res = predict_vit_binary(vit_binary_model, tensor)
    
    # If AI-generated, also identify the generator
    if res["is_ai"]:
        multiclass_res = predict_vit_multiclass(vit_multiclass_model, tensor)
        res["generator"] = multiclass_res["label"]
        res["generator_confidence"] = multiclass_res["confidence"]
        res["all_probs"] = multiclass_res["all_probs"]
    
    return res

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NAVBAR - Using st.columns for real buttons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col1, col2, col3, col4, col5 = st.columns([3, 4, 1, 1, 0.5])

with col1:
    st.markdown('<div class="brand"><div class="brand-logo">A</div>Authentic</div>', unsafe_allow_html=True)

with col3:
    if st.button("Detect", key="nav_detect", type="primary" if st.session_state.page == "home" else "secondary"):
        st.session_state.page = "home"
        st.session_state.analyzed_image = None
        st.session_state.result = None
        st.rerun()

with col4:
    if st.button("About", key="nav_about", type="primary" if st.session_state.page == "about" else "secondary"):
        st.session_state.page = "about"
        st.rerun()

with col5:
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    if st.button(theme_icon, key="theme_toggle"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

st.markdown(f'<hr style="margin: 8px 0 16px; border: none; border-top: 1px solid {border_color};">', unsafe_allow_html=True)

# Load models
try:
    model = load_model()  # Legacy CNN model
except:
    model = None

# Load ViT models and processor
try:
    vit_processor = load_vit_processor()
    vit_transform = get_vit_transforms(vit_processor)
    vit_binary_model = load_vit_binary_model()
    vit_multiclass_model = load_vit_multiclass_model()
except Exception as e:
    st.error(f"ViT models failed to load: {e}")
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HOME PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.page == "home":
    
    # Show result if we have one
    if st.session_state.analyzed_image and st.session_state.result:
        img = st.session_state.analyzed_image
        res = st.session_state.result
        b64 = img_to_b64(img)
        badge_class = "ai" if res["is_ai"] else "real"
        
        # Back button
        if st.button("← Analyze another image"):
            st.session_state.analyzed_image = None
            st.session_state.result = None
            st.session_state.feedback_submitted = False
            st.rerun()
        
        # Check if AI-generated with generator info
        has_generator_info = res["is_ai"] and "generator" in res
        
        if has_generator_info:
            # AI-generated image with generator identification
            sorted_probs = sorted(res["all_probs"].items(), key=lambda x: x[1], reverse=True)
            
            # Build the probability bars HTML
            probs_html = ""
            for label, prob in sorted_probs:
                probs_html += f'<div style="margin-bottom: 8px;"><div style="display: flex; justify-content: space-between; margin-bottom: 4px;"><span style="font-size: 13px; color: {text_secondary}; text-transform: capitalize;">{label}</span><span style="font-size: 13px; font-weight: 600; color: {text_primary};">{prob:.1f}%</span></div><div style="height: 8px; background: {bg_hover}; border-radius: 4px; overflow: hidden;"><div style="height: 100%; width: {prob}%; background: linear-gradient(90deg, #f43f5e, #ec4899); border-radius: 4px;"></div></div></div>'
            
            full_html = f"""
            <div class="result-card">
                <div class="result-image-wrap">
                    <img src="data:image/jpeg;base64,{b64}" class="result-img"/>
                    <div class="result-badge ai">
                        🤖 AI GENERATED
                    </div>
                </div>
                <div class="result-body">
                    <div class="result-header">
                        <div>
                            <div class="result-title">AI Generated Image</div>
                            <div class="result-subtitle">Generator: <strong style="text-transform: capitalize;">{res["generator"]}</strong> ({res["generator_confidence"]:.1f}% confidence)</div>
                        </div>
                        <div class="confidence-display">
                            <div class="confidence-number">{res["confidence"]:.1f}%</div>
                            <div class="confidence-label">AI Confidence</div>
                        </div>
                    </div>
                    <div style="margin-top: 16px;">
                        <div style="font-size: 12px; font-weight: 600; color: {text_muted}; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Generator Probability Breakdown</div>
                        """ + probs_html + f"""
                    </div>
                    <div class="stats-row">
                        <div class="stat-box">
                            <div class="stat-value">{res["raw"]:.4f}</div>
                            <div class="stat-label">AI Score</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{img.size[0]}×{img.size[1]}</div>
                            <div class="stat-label">Dimensions</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{"High" if res["confidence"] > 80 else "Medium" if res["confidence"] > 60 else "Low"}</div>
                            <div class="stat-label">Certainty</div>
                        </div>
                    </div>
                </div>
            </div>
            """
            st.markdown(full_html, unsafe_allow_html=True)
            
            # Add confidence interpretation
            interp_icon, interp_text = get_confidence_interpretation(res["confidence"], res["is_ai"])
            st.markdown(f'''
            <div class="confidence-interpret">
                <span class="interpret-icon">{interp_icon}</span>
                <span class="interpret-text">{interp_text}</span>
            </div>
            ''', unsafe_allow_html=True)
            
            # Feedback button
            if not st.session_state.feedback_submitted:
                feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 2, 1])
                with feedback_col2:
                    if st.button("🚩 Report Incorrect Result", use_container_width=True, key="feedback_ai"):
                        st.session_state.feedback_submitted = True
                        st.rerun()
            else:
                st.markdown(f'''
                <div style="text-align: center; padding: 12px; background: {bg_hover}; border-radius: 8px; margin-top: 16px;">
                    <span style="color: #22c55e;">✓</span>
                    <span style="color: {text_secondary}; font-size: 14px;"> Thanks for your feedback! We'll use it to improve our model.</span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            # Binary result display
            st.markdown(f"""
            <div class="result-card">
                <div class="result-image-wrap">
                    <img src="data:image/jpeg;base64,{b64}" class="result-img"/>
                    <div class="result-badge {badge_class}">
                        {"🤖" if res["is_ai"] else "📷"} {res["label"]}
                    </div>
                </div>
                <div class="result-body">
                    <div class="result-header">
                        <div>
                            <div class="result-title">{"AI Generated" if res["is_ai"] else "Real Photograph"}</div>
                            <div class="result-subtitle">Analysis completed</div>
                        </div>
                        <div class="confidence-display">
                            <div class="confidence-number">{res["confidence"]:.1f}%</div>
                            <div class="confidence-label">Confidence</div>
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill {badge_class}" style="width: {res['confidence']}%;"></div>
                    </div>
                    <div class="stats-row">
                        <div class="stat-box">
                            <div class="stat-value">{res["raw"]:.4f}</div>
                            <div class="stat-label">Raw Score</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{img.size[0]}×{img.size[1]}</div>
                            <div class="stat-label">Dimensions</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{"High" if res["confidence"] > 80 else "Medium" if res["confidence"] > 60 else "Low"}</div>
                            <div class="stat-label">Certainty</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add confidence interpretation
            interp_icon, interp_text = get_confidence_interpretation(res["confidence"], res["is_ai"])
            st.markdown(f'''
            <div class="confidence-interpret">
                <span class="interpret-icon">{interp_icon}</span>
                <span class="interpret-text">{interp_text}</span>
            </div>
            ''', unsafe_allow_html=True)
            
            # Feedback button
            if not st.session_state.feedback_submitted:
                feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 2, 1])
                with feedback_col2:
                    if st.button("🚩 Report Incorrect Result", use_container_width=True, key="feedback_binary"):
                        st.session_state.feedback_submitted = True
                        st.rerun()
            else:
                st.markdown(f'''
                <div style="text-align: center; padding: 12px; background: {bg_hover}; border-radius: 8px; margin-top: 16px;">
                    <span style="color: #22c55e;">✓</span>
                    <span style="color: {text_secondary}; font-size: 14px;"> Thanks for your feedback! We'll use it to improve our model.</span>
                </div>
                ''', unsafe_allow_html=True)
    
    # Show upload form
    else:
        st.markdown("""
        <div class="hero">
            <div class="hero-label">◉ AI Detection</div>
            <h1 class="hero-title">Real or AI?</h1>
            <p class="hero-subtitle">Upload an image and know instantly if it was created by AI or captured with a camera. If AI-generated, we'll identify the generator.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mode tabs
        mode_cols = st.columns([1, 1, 1, 1, 1])
        with mode_cols[1]:
            if st.button("📷 Single", key="mode_single", type="primary" if st.session_state.analysis_mode == "single" else "secondary", use_container_width=True):
                st.session_state.analysis_mode = "single"
                st.rerun()
        with mode_cols[2]:
            if st.button("⚖️ Compare", key="mode_compare", type="primary" if st.session_state.analysis_mode == "compare" else "secondary", use_container_width=True):
                st.session_state.analysis_mode = "compare"
                st.rerun()
        with mode_cols[3]:
            if st.button("📚 Batch", key="mode_batch", type="primary" if st.session_state.analysis_mode == "batch" else "secondary", use_container_width=True):
                st.session_state.analysis_mode = "batch"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════
        # SINGLE MODE
        # ═══════════════════════════════════════════
        if st.session_state.analysis_mode == "single":
            # File upload
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            
            # URL input
            st.markdown(f'<p style="text-align: center; font-size: 12px; color: {text_muted}; margin: 12px 0;">Or paste an image URL</p>', unsafe_allow_html=True)
            url_input = st.text_input("Image URL", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
            
            img_to_analyze = None
            
            if uploaded:
                img_to_analyze = Image.open(uploaded)
            elif url_input:
                try:
                    img_to_analyze = load_url(url_input)
                except Exception as e:
                    st.error(f"Failed to load image from URL: {str(e)}")
            
            if img_to_analyze:
                # Show image preview with loading overlay
                preview_b64 = img_to_b64(img_to_analyze)
                preview_placeholder = st.empty()
                preview_placeholder.markdown(f'''
                <div class="preview-card">
                    <div class="preview-image-wrap">
                        <img src="data:image/jpeg;base64,{preview_b64}" class="preview-img"/>
                        <div class="preview-overlay">
                            <div class="preview-spinner"></div>
                            <div class="preview-text">Analyzing image...</div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Run analysis
                res = analyze_image(img_to_analyze, vit_transform, vit_binary_model, vit_multiclass_model)
                
                # Clear preview
                preview_placeholder.empty()
                
                st.session_state.analyzed_image = img_to_analyze
                st.session_state.result = res
                # Add to history
                thumb = img_to_analyze.copy()
                thumb.thumbnail((80, 80), Image.Resampling.LANCZOS)
                st.session_state.history.insert(0, {
                    "thumb": img_to_b64(thumb, 80),
                    "label": res["label"],
                    "is_ai": res["is_ai"],
                    "confidence": res["confidence"]
                })
                st.session_state.history = st.session_state.history[:8]
                st.rerun()
            
            # Sample images
            st.markdown("""
            <div class="divider">
                <div class="divider-line"></div>
                <span class="divider-text">or try a sample</span>
                <div class="divider-line"></div>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, sample in enumerate(SAMPLES):
                with cols[i]:
                    st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
                    if st.button(f"{sample['icon']} {sample['name']}", key=f"sample_{i}", use_container_width=True):
                        try:
                            img = load_url(sample['url'])
                            res = analyze_image(img, vit_transform, vit_binary_model, vit_multiclass_model)
                            st.session_state.analyzed_image = img
                            st.session_state.result = res
                            thumb = img.copy()
                            thumb.thumbnail((80, 80), Image.Resampling.LANCZOS)
                            st.session_state.history.insert(0, {
                                "thumb": img_to_b64(thumb, 80),
                                "label": res["label"],
                                "is_ai": res["is_ai"],
                                "confidence": res["confidence"]
                            })
                            st.session_state.history = st.session_state.history[:8]
                            st.rerun()
                        except Exception as e:
                            st.error("Failed to load sample")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════
        # COMPARE MODE
        # ═══════════════════════════════════════════
        elif st.session_state.analysis_mode == "compare":
            st.markdown(f'<p style="text-align: center; font-size: 14px; color: {text_secondary}; margin-bottom: 16px;">Upload two images to compare their AI detection results side by side</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<p style="font-size: 13px; font-weight: 600; color: {text_primary}; margin-bottom: 8px;">Image 1</p>', unsafe_allow_html=True)
                uploaded1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="compare_1")
            
            with col2:
                st.markdown(f'<p style="font-size: 13px; font-weight: 600; color: {text_primary}; margin-bottom: 8px;">Image 2</p>', unsafe_allow_html=True)
                uploaded2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="compare_2")
            
            if uploaded1 and uploaded2:
                img1 = Image.open(uploaded1)
                img2 = Image.open(uploaded2)
                
                with st.spinner("Analyzing both images..."):
                    res1 = analyze_image(img1, vit_transform, vit_binary_model, vit_multiclass_model)
                    res2 = analyze_image(img2, vit_transform, vit_binary_model, vit_multiclass_model)
                
                # Display comparison results using Streamlit columns
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    b64_1 = img_to_b64(img1)
                    badge1 = "ai" if res1["is_ai"] else "real"
                    icon1 = "🤖" if res1["is_ai"] else "📷"
                    
                    card_html1 = f'''<div class="compare-card">
                        <div class="compare-image-wrap">
                            <img src="data:image/jpeg;base64,{b64_1}" class="compare-img"/>
                            <div class="compare-badge {badge1}">{icon1} {res1["label"]}</div>
                        </div>
                        <div class="compare-body">
                            <div class="compare-title">{res1["label"]}</div>
                            <div class="compare-conf">{res1["confidence"]:.1f}%</div>'''
                    
                    if res1.get("generator"):
                        card_html1 += f'<div style="font-size: 12px; color: {text_muted}; margin-top: 4px;">Generator: {res1["generator"]}</div>'
                    
                    card_html1 += '</div></div>'
                    
                    st.markdown(card_html1, unsafe_allow_html=True)
                
                with comp_col2:
                    b64_2 = img_to_b64(img2)
                    badge2 = "ai" if res2["is_ai"] else "real"
                    icon2 = "🤖" if res2["is_ai"] else "📷"
                    
                    card_html2 = f'''<div class="compare-card">
                        <div class="compare-image-wrap">
                            <img src="data:image/jpeg;base64,{b64_2}" class="compare-img"/>
                            <div class="compare-badge {badge2}">{icon2} {res2["label"]}</div>
                        </div>
                        <div class="compare-body">
                            <div class="compare-title">{res2["label"]}</div>
                            <div class="compare-conf">{res2["confidence"]:.1f}%</div>'''
                    
                    if res2.get("generator"):
                        card_html2 += f'<div style="font-size: 12px; color: {text_muted}; margin-top: 4px;">Generator: {res2["generator"]}</div>'
                    
                    card_html2 += '</div></div>'
                    
                    st.markdown(card_html2, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════
        # BATCH MODE
        # ═══════════════════════════════════════════
        elif st.session_state.analysis_mode == "batch":
            st.markdown(f'<p style="text-align: center; font-size: 14px; color: {text_secondary}; margin-bottom: 16px;">Upload multiple images at once for batch analysis</p>', unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Upload images", 
                type=["jpg", "jpeg", "png", "webp"], 
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                if st.button("🔍 Analyze All", type="primary", use_container_width=True):
                    batch_results = []
                    progress_bar = st.progress(0)
                    
                    for idx, uploaded in enumerate(uploaded_files):
                        try:
                            img = Image.open(uploaded)
                            res = analyze_image(img, vit_transform, vit_binary_model, vit_multiclass_model)
                            batch_results.append({
                                "image": img,
                                "b64": img_to_b64(img, 300),
                                "result": res,
                                "filename": uploaded.name
                            })
                        except Exception as e:
                            batch_results.append({
                                "image": None,
                                "b64": None,
                                "result": {"error": str(e)},
                                "filename": uploaded.name
                            })
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    st.session_state.batch_results = batch_results
                    progress_bar.empty()
                    st.rerun()
            
            # Show batch results
            if st.session_state.batch_results:
                results = st.session_state.batch_results
                ai_count = sum(1 for r in results if r["result"].get("is_ai", False))
                real_count = len(results) - ai_count
                
                # Summary
                st.markdown(f'''
                <div class="batch-summary">
                    <div class="summary-stat">
                        <div class="summary-number">{len(results)}</div>
                        <div class="summary-label">Total Images</div>
                    </div>
                    <div class="summary-stat">
                        <div class="summary-number ai">{ai_count}</div>
                        <div class="summary-label">AI Generated</div>
                    </div>
                    <div class="summary-stat">
                        <div class="summary-number real">{real_count}</div>
                        <div class="summary-label">Real Photos</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Grid of results
                cols = st.columns(2)
                for idx, item in enumerate(results):
                    if item["b64"] and "error" not in item["result"]:
                        res = item["result"]
                        label_class = "ai" if res["is_ai"] else "real"
                        icon = "🤖" if res["is_ai"] else "📷"
                        with cols[idx % 2]:
                            st.markdown(f'''
                            <div class="batch-card">
                                <img src="data:image/jpeg;base64,{item["b64"]}" class="batch-image"/>
                                <div class="batch-body">
                                    <div class="batch-label {label_class}">{icon} {res["label"]}</div>
                                    <div class="batch-conf">{res["confidence"]:.1f}%</div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                
                if st.button("🗑️ Clear Results", use_container_width=True):
                    st.session_state.batch_results = []
                    st.rerun()
        
        # History section
        if st.session_state.history and st.session_state.analysis_mode == "single":
            items_html = ""
            for item in st.session_state.history:
                label_class = "ai" if item["is_ai"] else "real"
                icon = "🤖" if item["is_ai"] else "📷"
                items_html += f'<div class="history-item"><img src="data:image/jpeg;base64,{item["thumb"]}" class="history-thumb"/><div class="history-info"><div class="history-label {label_class}">{icon} {item["label"]}</div><div class="history-conf">{item["confidence"]:.1f}%</div></div></div>'
            
            st.markdown(f'<div class="history-section"><div class="history-title">Recent Analyses</div><div class="history-grid">{items_html}</div></div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABOUT PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "about":
    st.markdown("""
    <div class="about-section">
        <div class="about-header">
            <h1 class="about-title">About Authentic</h1>
            <p class="about-desc">Advanced AI detection powered by Vision Transformer (ViT) models.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature 1
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-content">
            <h3>Vision Transformer (ViT)</h3>
            <p>State-of-the-art transformer architecture fine-tuned for AI image detection. First detects if an image is AI-generated, then automatically identifies which AI generator created it.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature 2
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎨</div>
        <div class="feature-content">
            <h3>Smart Generator Detection</h3>
            <p>When an AI image is detected, automatically identifies the generator: Midjourney, Stable Diffusion, GLIDE, BigGAN, ADM, VQDM, or Wukong. Real photos won't show misleading generator results.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature 3
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-content">
            <h3>Instant Detection</h3>
            <p>Get results in milliseconds. Simply upload an image and receive a confidence score.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature 4
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔒</div>
        <div class="feature-content">
            <h3>Privacy First</h3>
            <p>Images are processed locally. Nothing is stored or sent to external servers.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tech stack
    st.markdown("""
    <div class="tech-section">
        <div class="tech-title">Built with</div>
        <div class="tech-tags">
            <span class="tech-tag">Python</span>
            <span class="tech-tag">PyTorch</span>
            <span class="tech-tag">Transformers</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">Vision Transformer (ViT)</span>
            <span class="tech-tag">Hugging Face</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
