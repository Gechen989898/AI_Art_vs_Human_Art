import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw
import numpy as np
import requests
from io import BytesIO
import time
import base64
import json
import hashlib
from streamlit_javascript import st_javascript
import matplotlib.cm as cm

# Page configuration - centered layout
st.set_page_config(
    page_title="ArtLens AI",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "analyzed_image" not in st.session_state:
    st.session_state.analyzed_image = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "show_explainer" not in st.session_state:
    st.session_state.show_explainer = False
if "input_method" not in st.session_state:
    st.session_state.input_method = "upload"

# Theme CSS
def get_theme_css(theme):
    if theme == "dark":
        return """
        :root {
            --bg-primary: #09090b;
            --bg-secondary: #18181b;
            --bg-card: #1f1f23;
            --bg-card-hover: #27272a;
            --bg-input: #27272a;
            --border-color: #3f3f46;
            --border-hover: #52525b;
            --text-primary: #fafafa;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --accent-primary: #8b5cf6;
            --accent-secondary: #a78bfa;
            --accent-glow: rgba(139, 92, 246, 0.4);
            --success: #22c55e;
            --success-bg: rgba(34, 197, 94, 0.1);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.1);
            --warning: #f59e0b;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.4);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.5);
            --shadow-lg: 0 12px 40px rgba(0,0,0,0.6);
            --shadow-glow: 0 0 30px var(--accent-glow);
        }
        """
    else:
        return """
        :root {
            --bg-primary: #fafafa;
            --bg-secondary: #f4f4f5;
            --bg-card: #ffffff;
            --bg-card-hover: #f8f8f8;
            --bg-input: #ffffff;
            --border-color: #e4e4e7;
            --border-hover: #d4d4d8;
            --text-primary: #18181b;
            --text-secondary: #52525b;
            --text-muted: #a1a1aa;
            --accent-primary: #7c3aed;
            --accent-secondary: #8b5cf6;
            --accent-glow: rgba(124, 58, 237, 0.2);
            --success: #16a34a;
            --success-bg: rgba(22, 163, 74, 0.1);
            --danger: #dc2626;
            --danger-bg: rgba(220, 38, 38, 0.1);
            --warning: #d97706;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
            --shadow-lg: 0 12px 40px rgba(0,0,0,0.12);
            --shadow-glow: 0 0 30px var(--accent-glow);
        }
        """

# Inject CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    {get_theme_css(st.session_state.theme)}
    
    /* ===== HIDE STREAMLIT ===== */
    #MainMenu, footer, header, .stDeployButton,
    div[data-testid="stToolbar"], div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"], .stApp > header,
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    
    /* ===== BASE ===== */
    .stApp {{
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    .main .block-container {{
        padding: 1rem 1rem 4rem !important;
        max-width: 720px !important;
    }}
    
    /* ===== TOP BAR ===== */
    .top-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }}
    
    .logo {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .logo-icon {{
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--accent-primary), #ec4899);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }}
    
    .logo-text {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }}
    
    .theme-toggle {{
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 1.25rem;
    }}
    
    .theme-toggle:hover {{
        background: var(--bg-card-hover);
        border-color: var(--border-hover);
        transform: scale(1.05);
    }}
    
    /* ===== CARD ===== */
    .card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }}
    
    .card:hover {{
        border-color: var(--border-hover);
        box-shadow: var(--shadow-lg);
    }}
    
    .card-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }}
    
    .card-icon {{
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: var(--bg-secondary);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }}
    
    .card-title {{
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
    }}
    
    .card-subtitle {{
        font-size: 0.875rem;
        color: var(--text-muted);
    }}
    
    /* ===== INPUT TABS ===== */
    .input-tabs {{
        display: flex;
        gap: 8px;
        padding: 6px;
        background: var(--bg-secondary);
        border-radius: 14px;
        margin-bottom: 1.25rem;
    }}
    
    .input-tab {{
        flex: 1;
        padding: 10px 16px;
        border: none;
        background: transparent;
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }}
    
    .input-tab:hover {{
        color: var(--text-primary);
        background: var(--bg-card);
    }}
    
    .input-tab.active {{
        background: var(--accent-primary);
        color: white;
        box-shadow: var(--shadow-glow);
    }}
    
    /* ===== UPLOAD AREA ===== */
    .upload-area {{
        border: 2px dashed var(--border-color);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        background: var(--bg-secondary);
    }}
    
    .upload-area:hover {{
        border-color: var(--accent-primary);
        background: var(--accent-glow);
    }}
    
    .upload-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }}
    
    .upload-text {{
        color: var(--text-secondary);
        font-size: 0.9375rem;
        margin-bottom: 0.5rem;
    }}
    
    .upload-hint {{
        color: var(--text-muted);
        font-size: 0.8125rem;
    }}
    
    /* ===== RESULT CARD ===== */
    .result-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-lg);
    }}
    
    .result-image {{
        width: 100%;
        aspect-ratio: 4/3;
        object-fit: cover;
        display: block;
    }}
    
    .result-content {{
        padding: 1.5rem;
    }}
    
    .result-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 24px;
        border-radius: 100px;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1.25rem;
    }}
    
    .badge-ai {{
        background: var(--danger-bg);
        color: var(--danger);
        border: 1px solid var(--danger);
    }}
    
    .badge-real {{
        background: var(--success-bg);
        color: var(--success);
        border: 1px solid var(--success);
    }}
    
    /* ===== CONFIDENCE METER ===== */
    .confidence-container {{
        margin: 1.25rem 0;
    }}
    
    .confidence-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }}
    
    .confidence-label {{
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }}
    
    .confidence-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }}
    
    .confidence-bar {{
        height: 10px;
        background: var(--bg-secondary);
        border-radius: 100px;
        overflow: hidden;
    }}
    
    .confidence-fill {{
        height: 100%;
        border-radius: 100px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .confidence-fill.ai {{
        background: linear-gradient(90deg, var(--danger), var(--warning));
    }}
    
    .confidence-fill.real {{
        background: linear-gradient(90deg, var(--success), #10b981);
    }}
    
    /* ===== STATS ROW ===== */
    .stats-row {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 1.25rem;
    }}
    
    .stat-box {{
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }}
    
    .stat-value {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
    }}
    
    .stat-label {{
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }}
    
    /* ===== FEEDBACK ===== */
    .feedback-row {{
        display: flex;
        gap: 12px;
        margin-top: 1.25rem;
    }}
    
    .feedback-btn {{
        flex: 1;
        padding: 14px;
        border-radius: 12px;
        font-size: 0.9375rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    
    .feedback-correct {{
        background: var(--success-bg);
        border: 1px solid var(--success);
        color: var(--success);
    }}
    
    .feedback-correct:hover {{
        background: var(--success);
        color: white;
    }}
    
    .feedback-wrong {{
        background: var(--danger-bg);
        border: 1px solid var(--danger);
        color: var(--danger);
    }}
    
    .feedback-wrong:hover {{
        background: var(--danger);
        color: white;
    }}
    
    /* ===== HISTORY LIST ===== */
    .history-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: var(--bg-secondary);
        border-radius: 12px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }}
    
    .history-item:hover {{
        background: var(--bg-card-hover);
    }}
    
    .history-thumb {{
        width: 48px;
        height: 48px;
        border-radius: 8px;
        object-fit: cover;
    }}
    
    .history-info {{
        flex: 1;
    }}
    
    .history-label {{
        font-size: 0.9375rem;
        font-weight: 500;
        color: var(--text-primary);
    }}
    
    .history-meta {{
        font-size: 0.8125rem;
        color: var(--text-muted);
    }}
    
    .history-badge {{
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    
    .history-badge.ai {{
        background: var(--danger-bg);
        color: var(--danger);
    }}
    
    .history-badge.real {{
        background: var(--success-bg);
        color: var(--success);
    }}
    
    /* ===== STREAMLIT OVERRIDES ===== */
    [data-testid="stFileUploader"] {{
        background: transparent !important;
    }}
    
    [data-testid="stFileUploader"] section {{
        border: 2px dashed var(--border-color) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        background: var(--bg-secondary) !important;
        transition: all 0.3s ease !important;
    }}
    
    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--accent-primary) !important;
    }}
    
    [data-testid="stFileUploader"] button {{
        background: var(--accent-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }}
    
    .stButton > button {{
        background: var(--accent-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-glow) !important;
    }}
    
    .stButton > button[kind="secondary"] {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stTextInput > div > div > input {{
        background: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 14px 16px !important;
        font-size: 1rem !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }}
    
    [data-testid="stImage"] {{
        border-radius: 16px;
        overflow: hidden;
    }}
    
    .stAlert {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }}
    
    .stSpinner > div {{
        border-color: var(--accent-glow) !important;
        border-top-color: var(--accent-primary) !important;
    }}
    
    /* Hide sidebar */
    section[data-testid="stSidebar"] {{
        display: none !important;
    }}
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    .animate-fadeUp {{
        animation: fadeUp 0.5s ease forwards;
    }}
    
    .animate-pulse {{
        animation: pulse 2s ease-in-out infinite;
    }}
</style>
""", unsafe_allow_html=True)

# ============== CONSTANTS ==============
IMG_SIZE = (128, 128)
MODEL_PATH = "models/basic_cnn.keras"

SAMPLE_IMAGES = [
    {"name": "Mountain", "emoji": "🏔️", "url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400"},
    {"name": "Dog", "emoji": "🐕", "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400"},
    {"name": "Flower", "emoji": "🌸", "url": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400"},
]

# ============== HELPER FUNCTIONS ==============
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).reshape((1, 128, 128, 3)) / 255.0
    return img_array

def predict(model, img_array: np.ndarray) -> tuple:
    prediction = model.predict(img_array, verbose=0)
    score = float(prediction[0][0])
    if score < 0.5:
        return "AI Generated", "🤖", (1 - score) * 100, score
    else:
        return "Real Photo", "📷", score * 100, score

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def image_to_base64(image: Image.Image, size: tuple = (400, 300)) -> str:
    img = image.copy().convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

def image_to_thumbnail(image: Image.Image) -> str:
    img = image.copy().convert("RGB")
    img.thumbnail((60, 60), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode()

# ============== TOP BAR ==============
col_logo, col_theme = st.columns([5, 1])

with col_logo:
    st.markdown("""
    <div class="logo">
        <div class="logo-icon">🎨</div>
        <span class="logo-text">ArtLens AI</span>
    </div>
    """, unsafe_allow_html=True)

with col_theme:
    theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
    if st.button(theme_icon, key="theme_toggle", help="Toggle theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

st.markdown("<div style='height: 1px; background: var(--border-color); margin: 1rem 0 1.5rem;'></div>", unsafe_allow_html=True)

# ============== LOAD MODEL ==============
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model failed to load: {e}")
    model_loaded = False

if model_loaded:
    # ============== INPUT SECTION ==============
    st.markdown("""
    <div class="card animate-fadeUp">
        <div class="card-header">
            <div class="card-icon">🔍</div>
            <div>
                <div class="card-title">Analyze Image</div>
                <div class="card-subtitle">Upload or paste a URL to detect AI-generated images</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input method tabs
    tab_col1, tab_col2, tab_col3 = st.columns(3)
    
    with tab_col1:
        if st.button("📤 Upload", use_container_width=True, 
                     type="primary" if st.session_state.input_method == "upload" else "secondary"):
            st.session_state.input_method = "upload"
            st.rerun()
    
    with tab_col2:
        if st.button("🔗 URL", use_container_width=True,
                     type="primary" if st.session_state.input_method == "url" else "secondary"):
            st.session_state.input_method = "url"
            st.rerun()
    
    with tab_col3:
        if st.button("⚡ Samples", use_container_width=True,
                     type="primary" if st.session_state.input_method == "samples" else "secondary"):
            st.session_state.input_method = "samples"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    image = None
    
    # Upload method
    if st.session_state.input_method == "upload":
        uploaded_file = st.file_uploader(
            "Drop image here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
    
    # URL method
    elif st.session_state.input_method == "url":
        url = st.text_input("Image URL", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
        if st.button("🚀 Analyze URL", use_container_width=True):
            if url:
                try:
                    with st.spinner("Loading..."):
                        image = load_image_from_url(url)
                except Exception as e:
                    st.error(f"Failed: {e}")
    
    # Samples method
    elif st.session_state.input_method == "samples":
        cols = st.columns(3)
        for idx, sample in enumerate(SAMPLE_IMAGES):
            with cols[idx]:
                if st.button(f"{sample['emoji']} {sample['name']}", use_container_width=True):
                    try:
                        image = load_image_from_url(sample['url'])
                    except Exception as e:
                        st.error(f"Failed: {e}")
    
    # ============== RESULTS SECTION ==============
    if image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Process image
        with st.spinner("🔍 Analyzing..."):
            img_array = preprocess_image(image)
            label, emoji, confidence, raw_score = predict(model, img_array)
            is_ai = "AI" in label
            
            # Add to history
            st.session_state.history.insert(0, {
                "label": label,
                "confidence": confidence,
                "is_ai": is_ai,
                "time": time.strftime("%H:%M"),
                "thumbnail": image_to_thumbnail(image)
            })
            st.session_state.history = st.session_state.history[:8]
        
        # Display result
        img_b64 = image_to_base64(image)
        badge_class = "ai" if is_ai else "real"
        fill_class = "ai" if is_ai else "real"
        
        st.markdown(f"""
        <div class="result-card animate-fadeUp">
            <img src="data:image/jpeg;base64,{img_b64}" class="result-image" alt="Analyzed image"/>
            <div class="result-content">
                <div class="result-badge badge-{badge_class}">
                    {emoji} {label}
                </div>
                
                <div class="confidence-container">
                    <div class="confidence-header">
                        <span class="confidence-label">Confidence Score</span>
                        <span class="confidence-value">{confidence:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill {fill_class}" style="width: {confidence}%;"></div>
                    </div>
                </div>
                
                <div class="stats-row">
                    <div class="stat-box">
                        <div class="stat-value">{raw_score:.3f}</div>
                        <div class="stat-label">Raw Score</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{image.size[0]}×{image.size[1]}</div>
                        <div class="stat-label">Dimensions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{"High" if confidence > 80 else "Medium" if confidence > 60 else "Low"}</div>
                        <div class="stat-label">Certainty</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback
        st.markdown("<p style='color: var(--text-secondary); text-align: center; font-size: 0.9375rem;'>Was this prediction correct?</p>", unsafe_allow_html=True)
        
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            st.button("👍 Correct", use_container_width=True, type="secondary")
        with fb_col2:
            st.button("👎 Wrong", use_container_width=True, type="secondary")
    
    # ============== HISTORY SECTION ==============
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon">📜</div>
                <div>
                    <div class="card-title">Recent Analyses</div>
                    <div class="card-subtitle">Your last analyzed images</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for item in st.session_state.history[:5]:
            badge_class = "ai" if item.get("is_ai") else "real"
            st.markdown(f"""
            <div class="history-item">
                <img src="data:image/jpeg;base64,{item.get('thumbnail', '')}" class="history-thumb" alt="thumbnail"/>
                <div class="history-info">
                    <div class="history-label">{item.get('label', 'Unknown')}</div>
                    <div class="history-meta">{item.get('confidence', 0):.0f}% • {item.get('time', '')}</div>
                </div>
                <span class="history-badge {badge_class}">{"AI" if item.get("is_ai") else "Real"}</span>
            </div>
            """, unsafe_allow_html=True)

# ============== FOOTER ==============
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: var(--text-muted); font-size: 0.8125rem;">
    Built with neural networks • <a href="https://github.com/Gechen989898/AI_Art_vs_Human_Art" style="color: var(--accent-primary);">GitHub</a>
</div>
""", unsafe_allow_html=True)
