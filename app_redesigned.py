import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from io import BytesIO
import time
import base64
import json
import hashlib
from streamlit_javascript import st_javascript
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Page configuration
st.set_page_config(
    page_title="ArtLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False
if "load_attempt" not in st.session_state:
    st.session_state.load_attempt = 0
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "current_prediction_id" not in st.session_state:
    st.session_state.current_prediction_id = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "single"
if "batch_files_uploaded" not in st.session_state:
    st.session_state.batch_files_uploaded = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# Load history from localStorage
if not st.session_state.history_loaded:
    stored_history = st_javascript("localStorage.getItem('ai_classifier_history')")
    if stored_history == 0:
        st.session_state.load_attempt += 1
        if st.session_state.load_attempt < 3:
            time.sleep(0.1)
            st.rerun()
    else:
        if stored_history and stored_history != "null" and isinstance(stored_history, str):
            try:
                loaded_history = json.loads(stored_history)
                if isinstance(loaded_history, list) and len(loaded_history) > 0:
                    st.session_state.history = loaded_history
            except (json.JSONDecodeError, TypeError):
                pass
        st.session_state.history_loaded = True

# ============== CUSTOM CSS - COMPLETE REDESIGN ==============
st.markdown("""
<style>
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== ROOT VARIABLES ===== */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(255, 255, 255, 0.03);
        --bg-card-hover: rgba(255, 255, 255, 0.06);
        --border-color: rgba(255, 255, 255, 0.08);
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --text-muted: #6b6b6b;
        --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --ai-gradient: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
        --real-gradient: linear-gradient(135deg, #00d9a6 0%, #00b4d8 100%);
        --glass-bg: rgba(18, 18, 26, 0.8);
        --glass-border: rgba(255, 255, 255, 0.1);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 20px;
        --radius-xl: 28px;
    }
    
    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stStatusWidget"] {display: none;}
    .stApp > header {display: none;}
    
    /* Hide default sidebar toggle */
    button[kind="header"] {display: none;}
    
    /* ===== BASE STYLES ===== */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.8);
    }
    
    /* ===== HERO SECTION ===== */
    .hero-section {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        padding: 3rem 2rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(102, 126, 234, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(118, 75, 162, 0.1) 0%, transparent 50%);
        animation: aurora 15s ease-in-out infinite;
        pointer-events: none;
    }
    
    @keyframes aurora {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(-5%, 5%) rotate(5deg); }
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .logo-container {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin-bottom: 0.5rem;
    }
    
    .logo-icon {
        width: 48px;
        height: 48px;
        background: var(--accent-gradient);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .brand-name {
        font-size: 2rem;
        font-weight: 800;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .tagline {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    /* ===== MODE SELECTOR ===== */
    .mode-selector {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 1.5rem auto;
        padding: 6px;
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        max-width: 400px;
        backdrop-filter: blur(20px);
    }
    
    .mode-btn {
        flex: 1;
        padding: 12px 20px;
        border: none;
        background: transparent;
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        border-radius: var(--radius-md);
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .mode-btn:hover {
        color: var(--text-primary);
        background: var(--bg-card-hover);
    }
    
    .mode-btn.active {
        background: var(--accent-gradient);
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* ===== MAIN CONTAINER ===== */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1.5rem 3rem;
    }
    
    /* ===== GLASS CARD ===== */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-xl);
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-lg);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.6);
    }
    
    /* ===== UPLOAD ZONE ===== */
    .upload-zone {
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: var(--radius-xl);
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        transform: scale(1.01);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    
    .upload-title {
        color: var(--text-primary);
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        color: var(--text-muted);
        font-size: 0.875rem;
    }
    
    /* Override Streamlit file uploader */
    [data-testid="stFileUploader"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: 2px dashed rgba(102, 126, 234, 0.4) !important;
        border-radius: var(--radius-xl) !important;
        padding: 2.5rem !important;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(102, 126, 234, 0.8) !important;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        color: var(--text-secondary) !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: var(--accent-gradient) !important;
        border: none !important;
        color: white !important;
        border-radius: var(--radius-md) !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    /* ===== RESULT DISPLAY ===== */
    .result-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .result-image-card {
        background: var(--bg-card);
        border-radius: var(--radius-xl);
        padding: 1.5rem;
        border: 1px solid var(--border-color);
    }
    
    .result-prediction-card {
        background: var(--bg-card);
        border-radius: var(--radius-xl);
        padding: 2rem;
        border: 1px solid var(--border-color);
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: var(--radius-lg);
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .badge-ai {
        background: var(--ai-gradient);
        color: white;
    }
    
    .badge-real {
        background: var(--real-gradient);
        color: white;
    }
    
    .confidence-ring {
        width: 160px;
        height: 160px;
        margin: 0 auto 1rem;
        position: relative;
    }
    
    .confidence-value {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .confidence-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* ===== SAMPLE IMAGES ===== */
    .samples-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .sample-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-4px);
    }
    
    .sample-emoji {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .sample-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* ===== BATCH RESULTS ===== */
    .batch-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .batch-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .batch-item:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .batch-item-image {
        width: 100%;
        height: 140px;
        object-fit: cover;
    }
    
    .batch-item-info {
        padding: 1rem;
    }
    
    .batch-item-name {
        color: var(--text-primary);
        font-size: 0.875rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.5rem;
    }
    
    .batch-item-result {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .batch-tag {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .batch-tag-ai {
        background: rgba(255, 107, 107, 0.2);
        color: #ff6b6b;
    }
    
    .batch-tag-real {
        background: rgba(0, 217, 166, 0.2);
        color: #00d9a6;
    }
    
    /* ===== STATS BAR ===== */
    .stats-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .stat-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ===== BUTTONS ===== */
    .btn-primary {
        background: var(--accent-gradient);
        color: white;
        border: none;
        padding: 0.875rem 2rem;
        border-radius: var(--radius-md);
        font-weight: 600;
        font-size: 0.9375rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }
    
    .btn-secondary {
        background: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        padding: 0.875rem 2rem;
        border-radius: var(--radius-md);
        font-weight: 500;
        font-size: 0.9375rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .btn-secondary:hover {
        background: var(--bg-card-hover);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Override Streamlit buttons */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-card-hover) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
    }
    
    /* ===== TEXT INPUT ===== */
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.9375rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: 4px !important;
    }
    
    .stProgress > div > div {
        background: var(--bg-card) !important;
        border-radius: 4px !important;
    }
    
    /* ===== FEEDBACK BUTTONS ===== */
    .feedback-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .feedback-btn {
        padding: 0.75rem 2rem;
        border-radius: var(--radius-md);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .feedback-correct {
        background: rgba(0, 217, 166, 0.2);
        border: 1px solid rgba(0, 217, 166, 0.5);
        color: #00d9a6;
    }
    
    .feedback-correct:hover {
        background: rgba(0, 217, 166, 0.3);
        transform: scale(1.05);
    }
    
    .feedback-wrong {
        background: rgba(255, 107, 107, 0.2);
        border: 1px solid rgba(255, 107, 107, 0.5);
        color: #ff6b6b;
    }
    
    .feedback-wrong:hover {
        background: rgba(255, 107, 107, 0.3);
        transform: scale(1.05);
    }
    
    /* ===== HISTORY PANEL ===== */
    .history-panel {
        position: fixed;
        right: -320px;
        top: 0;
        width: 320px;
        height: 100vh;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border-left: 1px solid var(--glass-border);
        z-index: 1000;
        transition: right 0.3s ease;
        overflow-y: auto;
        padding: 1.5rem;
    }
    
    .history-panel.open {
        right: 0;
    }
    
    .history-toggle {
        position: fixed;
        right: 1rem;
        top: 1rem;
        width: 40px;
        height: 40px;
        background: var(--accent-gradient);
        border: none;
        border-radius: 50%;
        color: white;
        font-size: 1.25rem;
        cursor: pointer;
        z-index: 1001;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        transition: transform 0.3s ease;
    }
    
    .history-toggle:hover {
        transform: scale(1.1);
    }
    
    .history-title {
        color: var(--text-primary);
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .history-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        display: flex;
        gap: 0.75rem;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: var(--bg-card-hover);
        transform: translateX(-4px);
    }
    
    .history-thumb {
        width: 48px;
        height: 48px;
        border-radius: var(--radius-sm);
        object-fit: cover;
    }
    
    .history-info {
        flex: 1;
        min-width: 0;
    }
    
    .history-label {
        color: var(--text-primary);
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .history-meta {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-top: 2px;
    }
    
    /* ===== MOBILE RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero-section {
            padding: 2rem 1rem 1.5rem;
        }
        
        .brand-name {
            font-size: 1.5rem;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            font-size: 1.25rem;
        }
        
        .tagline {
            font-size: 0.875rem;
        }
        
        .mode-selector {
            flex-direction: column;
            max-width: 280px;
        }
        
        .mode-btn {
            padding: 10px 16px;
            font-size: 0.8125rem;
        }
        
        .main-container {
            padding: 0 1rem 2rem;
        }
        
        .glass-card {
            padding: 1.25rem;
            border-radius: var(--radius-lg);
        }
        
        .result-container {
            grid-template-columns: 1fr;
        }
        
        .samples-grid {
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
        }
        
        .sample-card {
            padding: 0.75rem 0.5rem;
        }
        
        .sample-emoji {
            font-size: 1.5rem;
        }
        
        .sample-label {
            font-size: 0.75rem;
        }
        
        .stats-bar {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
        
        .stat-item {
            padding: 1rem;
        }
        
        .stat-value {
            font-size: 1.5rem;
        }
        
        .batch-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
        
        .batch-item-image {
            height: 100px;
        }
        
        .batch-item-info {
            padding: 0.75rem;
        }
        
        .batch-item-name {
            font-size: 0.75rem;
        }
        
        .history-panel {
            width: 100%;
            right: -100%;
        }
        
        .history-toggle {
            width: 36px;
            height: 36px;
            font-size: 1rem;
        }
        
        .confidence-ring {
            width: 120px;
            height: 120px;
        }
        
        .confidence-value {
            font-size: 1.5rem;
        }
        
        .prediction-badge {
            font-size: 1rem;
            padding: 0.75rem 1.5rem;
        }
        
        [data-testid="stFileUploader"] section {
            padding: 1.5rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .samples-grid {
            grid-template-columns: repeat(3, 1fr);
        }
        
        .batch-grid {
            grid-template-columns: 1fr 1fr;
        }
        
        .stats-bar {
            grid-template-columns: 1fr 1fr;
        }
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .animate-fadeIn {
        animation: fadeIn 0.5s ease forwards;
    }
    
    .animate-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .animate-slideIn {
        animation: slideIn 0.3s ease forwards;
    }
    
    /* ===== GRAD-CAM SECTION ===== */
    .gradcam-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .gradcam-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1rem;
        text-align: center;
    }
    
    .gradcam-title {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    
    @media (max-width: 768px) {
        .gradcam-container {
            grid-template-columns: 1fr;
        }
    }
    
    /* ===== SECTION TITLES ===== */
    .section-title {
        color: var(--text-primary);
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-subtitle {
        color: var(--text-muted);
        font-size: 0.875rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    
    /* ===== INFO CARDS ===== */
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .info-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
    }
    
    .info-card-title {
        color: var(--text-primary);
        font-size: 0.9375rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .info-card-content {
        color: var(--text-secondary);
        font-size: 0.8125rem;
        line-height: 1.6;
    }
    
    @media (max-width: 768px) {
        .info-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* ===== URL INPUT GROUP ===== */
    .url-input-group {
        display: flex;
        gap: 0.75rem;
    }
    
    .url-input-group .stTextInput {
        flex: 1;
    }
    
    @media (max-width: 768px) {
        .url-input-group {
            flex-direction: column;
        }
    }
    
    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        box-shadow: none !important;
    }
    
    .stDownloadButton > button:hover {
        background: var(--bg-card-hover) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* ===== SUCCESS/ERROR/WARNING/INFO MESSAGES ===== */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stAlert > div {
        color: var(--text-primary) !important;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.2) !important;
        border-top-color: #667eea !important;
    }
    
    /* ===== TABS OVERRIDE (hidden, using custom nav) ===== */
    .stTabs [data-baseweb="tab-list"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0 !important;
    }
    
    /* ===== CAPTION ===== */
    .stCaption {
        color: var(--text-muted) !important;
    }
    
    /* ===== IMAGE DISPLAY ===== */
    [data-testid="stImage"] {
        border-radius: var(--radius-lg);
        overflow: hidden;
    }
    
    [data-testid="stImage"] img {
        border-radius: var(--radius-lg);
    }
    
</style>
""", unsafe_allow_html=True)

# ============== CONSTANTS ==============
IMG_SIZE = (128, 128)
MODEL_PATH = "models/basic_cnn.keras"

SAMPLE_IMAGES = {
    "Landscape": {
        "url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400",
        "emoji": "🏔️"
    },
    "Portrait": {
        "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400", 
        "emoji": "🐕"
    },
    "Nature": {
        "url": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400",
        "emoji": "🌸"
    },
}

# ============== HELPER FUNCTIONS ==============
@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)"""
    model = keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = img_array.reshape((1, 128, 128, 3))
    img_array = img_array / 255.0
    return img_array


def predict(model, img_array: np.ndarray) -> tuple[str, str, float, float]:
    """Run prediction and return result with confidence"""
    prediction = model.predict(img_array, verbose=0)
    score = float(prediction[0][0])
    
    if score < 0.5:
        label = "AI Generated"
        emoji = "🤖"
        confidence = (1 - score) * 100
    else:
        label = "Real Photo"
        emoji = "📷"
        confidence = score * 100
    
    return label, emoji, confidence, score


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


def create_gauge_svg(confidence: float, is_ai: bool) -> str:
    """Create an animated SVG gauge for confidence display"""
    radius = 45
    circumference = 2 * 3.14159 * radius
    offset = circumference - (confidence / 100 * circumference)
    
    color = "#ff6b6b" if is_ai else "#00d9a6"
    
    svg = f"""
    <svg width="160" height="160" viewBox="0 0 100 100" style="transform: rotate(-90deg);">
        <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:{'#ff6b6b' if is_ai else '#00d9a6'}" />
                <stop offset="100%" style="stop-color:{'#ff8e53' if is_ai else '#00b4d8'}" />
            </linearGradient>
        </defs>
        <circle cx="50" cy="50" r="{radius}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8"/>
        <circle cx="50" cy="50" r="{radius}" fill="none" stroke="url(#gaugeGrad)" 
                stroke-width="8" stroke-linecap="round"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                style="transition: stroke-dashoffset 1s ease;"/>
    </svg>
    """
    return svg


def get_last_conv_layer_name(model):
    """Find the name of the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam(model, img_array: np.ndarray, pred_class: int = None) -> np.ndarray:
    """Generate Grad-CAM heatmap for the given image."""
    try:
        last_conv_layer_name = get_last_conv_layer_name(model)
        if last_conv_layer_name is None:
            return None
        
        if not model.built:
            model.build(input_shape=(None, 128, 128, 3))
        
        inputs = keras.Input(shape=(128, 128, 3))
        x = inputs
        
        last_conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                last_conv_output = x
        
        if last_conv_output is None:
            return None
        
        grad_model = keras.Model(inputs=inputs, outputs=[last_conv_output, x])
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_class is None:
                pred_class = 1 if predictions[0][0] >= 0.5 else 0
            
            if pred_class == 1:
                class_output = predictions[0][0]
            else:
                class_output = 1 - predictions[0][0]
        
        grads = tape.gradient(class_output, conv_outputs)
        
        if grads is None:
            return None
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + keras.backend.epsilon())
        
        return heatmap.numpy()
    
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def create_gradcam_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay Grad-CAM heatmap on the original image."""
    img = image.copy().convert("RGB")
    original_size = img.size
    
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        original_size, Image.Resampling.BILINEAR
    )) / 255.0
    
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored)
    
    overlaid = Image.blend(img, heatmap_pil, alpha)
    return overlaid


def create_result_image(image: Image.Image, label: str, confidence: float) -> bytes:
    """Create a downloadable result image with prediction overlay"""
    img = image.copy()
    img = img.convert("RGB")
    
    max_width = 800
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    draw = ImageDraw.Draw(img)
    banner_height = 60
    banner_y = img.height - banner_height
    
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    banner_color = (255, 107, 107, 220) if "AI" in label else (0, 217, 166, 220)
    overlay_draw.rectangle([0, banner_y, img.width, img.height], fill=banner_color)
    
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    
    draw = ImageDraw.Draw(img)
    text = f"{label} • {confidence:.1f}% confidence"
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (img.width - text_width) // 2
    text_y = banner_y + (banner_height - 20) // 2
    draw.text((text_x, text_y), text, fill="white")
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_base64_thumbnail(image: Image.Image, size: tuple = (60, 60)) -> str:
    """Convert image to base64 thumbnail for history storage"""
    img = image.copy()
    img = img.convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode()


def add_to_history(label: str, confidence: float, is_ai: bool, image: Image.Image = None):
    """Add prediction to session history with thumbnail"""
    entry = {
        "label": label,
        "confidence": confidence,
        "is_ai": is_ai,
        "time": time.strftime("%H:%M:%S"),
        "thumbnail": image_to_base64_thumbnail(image) if image else None
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]
    save_history_to_local_storage()


def save_feedback(prediction_id: str, prediction: str, confidence: float, 
                  is_correct: bool, thumbnail: str = None):
    """Save user feedback to session state and file"""
    feedback_entry = {
        "id": prediction_id,
        "prediction": prediction,
        "confidence": confidence,
        "is_correct": is_correct,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thumbnail": thumbnail
    }
    st.session_state.feedback_log.append(feedback_entry)
    
    feedback_file = "feedback_log.json"
    try:
        import os
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                existing = json.load(f)
        else:
            existing = []
        
        file_entry = {k: v for k, v in feedback_entry.items() if k != "thumbnail"}
        existing.append(file_entry)
        
        with open(feedback_file, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"Error saving feedback: {e}")


def save_history_to_local_storage():
    """Save history to localStorage using streamlit-javascript"""
    history_for_storage = []
    for item in st.session_state.history:
        history_for_storage.append({
            "label": item["label"],
            "confidence": item["confidence"],
            "is_ai": item["is_ai"],
            "time": item["time"],
            "thumbnail": item.get("thumbnail", "")
        })
    
    history_json = json.dumps(history_for_storage)
    history_json_escaped = history_json.replace("'", "\\'")
    st_javascript(f"localStorage.setItem('ai_classifier_history', '{history_json_escaped}')")


# ============== MAIN APP ==============

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="logo-container">
            <div class="logo-icon">🔍</div>
            <span class="brand-name">ArtLens AI</span>
        </div>
        <p class="tagline">Detect AI-generated images with deep learning</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

if model_loaded:
    # Mode Selector (Custom navigation instead of tabs)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Custom mode selector using columns and buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col2:
        if st.button("📷 Single Image", use_container_width=True, 
                     type="primary" if st.session_state.active_mode == "single" else "secondary",
                     key="mode_single"):
            st.session_state.active_mode = "single"
            st.rerun()
    
    with col3:
        if st.button("🔗 From URL", use_container_width=True,
                     type="primary" if st.session_state.active_mode == "url" else "secondary",
                     key="mode_url"):
            st.session_state.active_mode = "url"
            st.rerun()
    
    with col4:
        if st.button("📦 Batch", use_container_width=True,
                     type="primary" if st.session_state.active_mode == "batch" else "secondary",
                     key="mode_batch"):
            st.session_state.active_mode = "batch"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    image = None
    
    # ============== SINGLE IMAGE MODE ==============
    if st.session_state.active_mode == "single":
        st.markdown("""
        <div class="glass-card animate-fadeIn">
            <div class="section-title">📤 Upload an Image</div>
            <p class="section-subtitle">Drop your image here to analyze whether it's AI-generated or real</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP",
            label_visibility="collapsed",
            key="single_upload"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        
        # Sample images section
        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem;">
            <div class="section-title">🎯 Quick Test</div>
            <p class="section-subtitle">Try with sample images</p>
        </div>
        """, unsafe_allow_html=True)
        
        sample_cols = st.columns(3)
        for idx, (name, data) in enumerate(SAMPLE_IMAGES.items()):
            with sample_cols[idx]:
                if st.button(f"{data['emoji']} {name}", use_container_width=True, key=f"sample_{idx}"):
                    try:
                        image = load_image_from_url(data['url'])
                    except Exception as e:
                        st.error(f"Failed to load: {e}")
    
    # ============== URL MODE ==============
    elif st.session_state.active_mode == "url":
        st.markdown("""
        <div class="glass-card animate-fadeIn">
            <div class="section-title">🔗 Analyze from URL</div>
            <p class="section-subtitle">Enter an image URL to analyze</p>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.jpg",
            label_visibility="collapsed",
            key="url_input"
        )
        
        col_load, col_space = st.columns([1, 2])
        with col_load:
            load_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")
        
        if url and load_btn:
            try:
                with st.spinner("Loading image..."):
                    image = load_image_from_url(url)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
    
    # ============== BATCH MODE ==============
    elif st.session_state.active_mode == "batch":
        st.markdown("""
        <div class="glass-card animate-fadeIn">
            <div class="section-title">📦 Batch Processing</div>
            <p class="section-subtitle">Upload multiple images to analyze them all at once</p>
        </div>
        """, unsafe_allow_html=True)
        
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Select multiple images to process in batch",
            key="batch_uploader",
            label_visibility="collapsed"
        )
        
        if batch_files:
            st.info(f"📁 {len(batch_files)} image(s) selected")
            
            col_process, col_clear = st.columns(2)
            with col_process:
                process_batch = st.button("🚀 Analyze All", use_container_width=True, type="primary")
            with col_clear:
                clear_results = st.button("🗑️ Clear Results", use_container_width=True)
            
            if clear_results:
                st.session_state.batch_results = []
                st.rerun()
            
            if process_batch:
                st.session_state.batch_results = []
                progress_bar = st.progress(0, text="Processing images...")
                
                for idx, file in enumerate(batch_files):
                    try:
                        img = Image.open(file)
                        
                        img_bytes = img.tobytes()
                        img_hash = hashlib.md5(img_bytes).hexdigest()
                        is_duplicate = img_hash in st.session_state.processed_images
                        
                        img_array = preprocess_image(img)
                        label, emoji, confidence, raw_score = predict(model, img_array)
                        is_ai = "AI" in label
                        
                        st.session_state.processed_images[img_hash] = {
                            "label": label,
                            "confidence": confidence,
                            "time": time.strftime("%H:%M:%S")
                        }
                        
                        st.session_state.batch_results.append({
                            "filename": file.name,
                            "prediction": label,
                            "confidence": round(confidence, 1),
                            "raw_score": round(raw_score, 4),
                            "is_ai": is_ai,
                            "emoji": emoji,
                            "thumbnail": image_to_base64_thumbnail(img),
                            "is_duplicate": is_duplicate
                        })
                    except Exception as e:
                        st.session_state.batch_results.append({
                            "filename": file.name,
                            "prediction": "Error",
                            "confidence": 0,
                            "raw_score": 0,
                            "is_ai": False,
                            "emoji": "❌",
                            "thumbnail": "",
                            "is_duplicate": False
                        })
                    
                    progress_bar.progress((idx + 1) / len(batch_files), 
                                          text=f"Processing {idx + 1}/{len(batch_files)}...")
                
                progress_bar.empty()
                st.success(f"✅ Processed {len(batch_files)} images!")
                st.rerun()
        
        # Display batch results
        if st.session_state.batch_results:
            st.markdown("---")
            
            # Stats
            total = len(st.session_state.batch_results)
            ai_count = sum(1 for r in st.session_state.batch_results if r["is_ai"])
            real_count = total - ai_count
            dup_count = sum(1 for r in st.session_state.batch_results if r.get("is_duplicate", False))
            
            st.markdown(f"""
            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #ff6b6b;">{ai_count}</div>
                    <div class="stat-label">AI Generated</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #00d9a6;">{real_count}</div>
                    <div class="stat-label">Real Photos</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #ffd93d;">{dup_count}</div>
                    <div class="stat-label">Duplicates</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Results grid
            st.markdown('<div class="batch-grid">', unsafe_allow_html=True)
            cols = st.columns(4)
            for idx, result in enumerate(st.session_state.batch_results):
                with cols[idx % 4]:
                    tag_class = "batch-tag-ai" if result["is_ai"] else "batch-tag-real"
                    st.markdown(f"""
                    <div class="batch-item animate-slideIn" style="animation-delay: {idx * 0.05}s;">
                        <img class="batch-item-image" src="data:image/jpeg;base64,{result['thumbnail']}" alt="{result['filename']}"/>
                        <div class="batch-item-info">
                            <div class="batch-item-name">{result['filename']}</div>
                            <div class="batch-item-result">
                                <span class="batch-tag {tag_class}">{result['emoji']} {result['prediction'][:10]}</span>
                                <span style="color: var(--text-muted); font-size: 0.75rem;">{result['confidence']}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Export
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = "Filename,Prediction,Confidence (%),Raw Score,Duplicate\n"
            for r in st.session_state.batch_results:
                is_dup = "Yes" if r.get("is_duplicate", False) else "No"
                csv_data += f"{r['filename']},{r['prediction']},{r['confidence']},{r['raw_score']},{is_dup}\n"
            
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # ============== RESULTS SECTION ==============
    if image is not None and st.session_state.active_mode != "batch":
        st.markdown("---")
        
        # Compute image hash for duplicate detection
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        is_duplicate = img_hash in st.session_state.processed_images
        
        if is_duplicate:
            prev = st.session_state.processed_images[img_hash]
            st.warning(f"⚠️ This image was already analyzed at {prev['time']} — {prev['label']} ({prev['confidence']:.1f}%)")
            reanalyze = st.checkbox("🔄 Re-analyze anyway", key=f"reanalyze_{img_hash}")
            if not reanalyze:
                st.stop()
        
        # Analyze
        with st.spinner("🔍 Analyzing..."):
            img_array = preprocess_image(image)
            time.sleep(0.3)
            label, emoji, confidence, raw_score = predict(model, img_array)
            is_ai = "AI" in label
            
            gradcam_heatmap = generate_gradcam(model, img_array)
            gradcam_overlay = None
            if gradcam_heatmap is not None:
                gradcam_overlay = create_gradcam_overlay(image, gradcam_heatmap)
            
            st.session_state.processed_images[img_hash] = {
                "label": label,
                "confidence": confidence,
                "time": time.strftime("%H:%M:%S")
            }
            
            if not is_duplicate:
                add_to_history(label, confidence, is_ai, image)
        
        # Results display
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">📷 Input Image</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
            w, h = image.size
            st.caption(f"Size: {w} × {h} px")
            
            result_img = create_result_image(image, label, confidence)
            st.download_button(
                label="💾 Download Result",
                data=result_img,
                file_name="ai_detection_result.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_result:
            badge_class = "badge-ai" if is_ai else "badge-real"
            
            # Prediction badge
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="section-title" style="justify-content: center;">🎯 Prediction</div>
                <div style="margin: 1.5rem 0;">
                    <span class="prediction-badge {badge_class}">{emoji} {label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge - separate markdown call
            gauge_svg = create_gauge_svg(confidence, is_ai)
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0;">
                <div style="display: inline-block;">{gauge_svg}</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-top: 0.5rem;">{confidence:.0f}%</div>
                <div style="color: #6b6b6b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Certainty indicator
            certainty = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
            cert_color = "#00d9a6" if confidence > 80 else "#ffd93d" if confidence > 60 else "#ff6b6b"
            st.markdown(f"""
            <p style="text-align: center; margin-top: 0.5rem;">
                <span style="background: {cert_color}20; border: 1px solid {cert_color}; color: {cert_color}; 
                             padding: 0.35rem 1rem; border-radius: 20px; font-size: 0.8125rem; font-weight: 500;">
                    {certainty} certainty
                </span>
            </p>
            """, unsafe_allow_html=True)
        
        # Feedback section
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center;">
            <div class="section-title" style="justify-content: center;">💬 Was this prediction correct?</div>
            <p style="color: var(--text-muted); font-size: 0.875rem; margin-top: -0.5rem;">Your feedback helps improve the model</p>
        </div>
        """, unsafe_allow_html=True)
        
        prediction_id = f"{hash(image.tobytes()) % 10000}_{int(time.time())}"
        current_thumbnail = image_to_base64_thumbnail(image)
        
        if st.session_state.current_prediction_id != prediction_id:
            st.session_state.current_prediction_id = prediction_id
            st.session_state.feedback_given = False
        
        if not st.session_state.feedback_given:
            col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 1])
            with col_fb1:
                if st.button("👍 Correct", use_container_width=True, type="primary"):
                    save_feedback(prediction_id, label, confidence, True, current_thumbnail)
                    st.session_state.feedback_given = True
                    st.rerun()
            with col_fb3:
                if st.button("👎 Wrong", use_container_width=True):
                    save_feedback(prediction_id, label, confidence, False, current_thumbnail)
                    st.session_state.feedback_given = True
                    st.rerun()
        else:
            st.success("✅ Thanks for your feedback!")
            if st.session_state.feedback_log:
                total_fb = len(st.session_state.feedback_log)
                correct_fb = sum(1 for f in st.session_state.feedback_log if f["is_correct"])
                accuracy = (correct_fb / total_fb * 100) if total_fb > 0 else 0
                st.caption(f"📊 Session accuracy: {accuracy:.0f}% ({correct_fb}/{total_fb} correct)")
        
        # Grad-CAM section
        if gradcam_overlay is not None:
            st.markdown("---")
            st.markdown("""
            <div class="section-title">🔥 Model Explainability</div>
            <p class="section-subtitle">Grad-CAM visualization shows which regions influenced the prediction</p>
            """, unsafe_allow_html=True)
            
            gcam_col1, gcam_col2 = st.columns(2)
            
            with gcam_col1:
                st.markdown("**Original**")
                st.image(image, use_container_width=True)
            
            with gcam_col2:
                st.markdown("**Attention Heatmap**")
                st.image(gradcam_overlay, use_container_width=True)
            
            with st.expander("💡 How to interpret"):
                st.markdown("""
                **Red/Yellow regions** = High importance • **Blue/Green** = Lower importance
                
                AI images often show uniform attention or focus on texture artifacts.
                Real photos typically show attention on semantic objects (faces, animals, etc.)
                """)
    
    # About section
    st.markdown("---")
    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-card-title">ℹ️ About</div>
            <div class="info-card-content">
                CNN trained on <b>Tiny GenImage</b> dataset with 7 AI generators: 
                BigGAN, VQDM, Stable Diffusion v5, Wukong, ADM, Glide, Midjourney
            </div>
        </div>
        <div class="info-card">
            <div class="info-card-title">⚠️ Limitations</div>
            <div class="info-card-content">
                Trained on nature/outdoor images. May misclassify portraits, 
                indoor photos, or digital artwork.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: var(--text-muted); font-size: 0.8125rem;">
        Built with ❤️ by the AI Art Detection Team • 
        <a href="https://github.com/Gechen989898/AI_Art_vs_Human_Art" style="color: #667eea;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# ============== HISTORY SIDEBAR ==============
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div class="history-title">📜 History</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        for item in st.session_state.history:
            is_ai = item.get("is_ai", False)
            confidence = float(item.get("confidence", 0))
            label = item.get("label", "Unknown")
            item_time = item.get("time", "")
            thumbnail = item.get("thumbnail", "")
            
            border_color = "#ff6b6b" if is_ai else "#00d9a6"
            emoji = "🤖" if is_ai else "📷"
            
            thumbnail_html = ""
            if thumbnail:
                thumbnail_html = f'<img class="history-thumb" src="data:image/jpeg;base64,{thumbnail}" alt="thumbnail"/>'
            
            st.markdown(f"""
            <div class="history-item" style="border-left: 3px solid {border_color};">
                {thumbnail_html}
                <div class="history-info">
                    <div class="history-label">{emoji} {label}</div>
                    <div class="history-meta">{confidence:.0f}% • {item_time}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.history_loaded = True
            st_javascript("localStorage.removeItem('ai_classifier_history')")
            st.rerun()
    else:
        st.caption("No predictions yet")
    
    st.markdown("---")
    if st.session_state.history:
        ai_count = sum(1 for h in st.session_state.history if h["is_ai"])
        real_count = len(st.session_state.history) - ai_count
        col1, col2 = st.columns(2)
        col1.metric("🤖 AI", ai_count)
        col2.metric("📷 Real", real_count)
