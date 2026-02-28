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

# Page configuration - WIDE layout
st.set_page_config(
    page_title="Authentic • Real vs AI Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "current_page" not in st.session_state:
    st.session_state.current_page = "analyze"
if "input_method" not in st.session_state:
    st.session_state.input_method = "upload"

# Theme CSS
def get_theme_css(theme):
    if theme == "dark":
        return """
        :root {
            --bg-primary: #0c0c0c;
            --bg-secondary: #161616;
            --bg-card: #1c1c1c;
            --bg-card-hover: #242424;
            --bg-elevated: #2a2a2a;
            --border-color: #2e2e2e;
            --border-hover: #404040;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --success: #22c55e;
            --success-bg: rgba(34, 197, 94, 0.15);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.15);
        }
        """
    else:
        return """
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --bg-card: #ffffff;
            --bg-card-hover: #fafafa;
            --bg-elevated: #f0f0f0;
            --border-color: #e5e5e5;
            --border-hover: #d4d4d4;
            --text-primary: #171717;
            --text-secondary: #525252;
            --text-muted: #a3a3a3;
            --accent: #4f46e5;
            --accent-hover: #6366f1;
            --accent-glow: rgba(79, 70, 229, 0.2);
            --success: #16a34a;
            --success-bg: rgba(22, 163, 74, 0.1);
            --danger: #dc2626;
            --danger-bg: rgba(220, 38, 38, 0.1);
        }
        """

# Main CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    {get_theme_css(st.session_state.theme)}
    
    /* Hide Streamlit */
    #MainMenu, footer, header, .stDeployButton,
    div[data-testid="stToolbar"], div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"], .stApp > header,
    [data-testid="collapsedControl"], section[data-testid="stSidebar"] {{
        display: none !important;
    }}
    
    .stApp {{
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}
    
    .main .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}
    
    /* ===== NAVBAR ===== */
    .navbar {{
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--bg-primary);
        border-bottom: 1px solid var(--border-color);
        padding: 0 3rem;
    }}
    
    .navbar-inner {{
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 64px;
    }}
    
    .navbar-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .brand-icon {{
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, var(--accent), #ec4899);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.125rem;
    }}
    
    .brand-text {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }}
    
    .brand-badge {{
        background: var(--accent);
        color: white;
        font-size: 0.625rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .navbar-nav {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .nav-link {{
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-decoration: none;
        transition: all 0.2s;
        cursor: pointer;
        border: none;
        background: transparent;
    }}
    
    .nav-link:hover {{
        color: var(--text-primary);
        background: var(--bg-secondary);
    }}
    
    .nav-link.active {{
        color: var(--text-primary);
        background: var(--bg-card);
    }}
    
    .navbar-actions {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .theme-btn {{
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-size: 1.125rem;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .theme-btn:hover {{
        background: var(--bg-card-hover);
        color: var(--text-primary);
    }}
    
    /* ===== MAIN CONTENT ===== */
    .main-content {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem 3rem 4rem;
    }}
    
    /* ===== HERO ===== */
    .hero {{
        text-align: center;
        padding: 3rem 0 2rem;
    }}
    
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--accent-glow);
        color: var(--accent);
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.8125rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
    }}
    
    .hero-title {{
        font-size: 3rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 1rem;
    }}
    
    .hero-title span {{
        background: linear-gradient(135deg, var(--accent), #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .hero-subtitle {{
        font-size: 1.25rem;
        color: var(--text-secondary);
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }}
    
    /* ===== BENTO GRID ===== */
    .bento-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: 2rem;
    }}
    
    .bento-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }}
    
    .bento-card:hover {{
        border-color: var(--border-hover);
        transform: translateY(-2px);
    }}
    
    .bento-card.main {{
        grid-column: span 1;
    }}
    
    .bento-card.full {{
        grid-column: span 2;
    }}
    
    .bento-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1.5rem;
    }}
    
    .bento-icon {{
        width: 48px;
        height: 48px;
        background: var(--bg-elevated);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }}
    
    .bento-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }}
    
    .bento-subtitle {{
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 2px;
    }}
    
    /* ===== INPUT METHODS ===== */
    .method-tabs {{
        display: flex;
        gap: 8px;
        margin-bottom: 1.5rem;
    }}
    
    .method-tab {{
        flex: 1;
        padding: 12px 16px;
        background: var(--bg-secondary);
        border: 1px solid transparent;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    
    .method-tab:hover {{
        color: var(--text-primary);
        background: var(--bg-elevated);
    }}
    
    .method-tab.active {{
        background: var(--accent);
        color: white;
        border-color: var(--accent);
    }}
    
    /* ===== RESULT CARD ===== */
    .result-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        overflow: hidden;
    }}
    
    .result-image-container {{
        position: relative;
        aspect-ratio: 16/10;
        overflow: hidden;
    }}
    
    .result-image {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
    
    .result-overlay {{
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 2rem;
        background: linear-gradient(transparent, rgba(0,0,0,0.8));
    }}
    
    .result-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        border-radius: 100px;
        font-size: 1rem;
        font-weight: 700;
    }}
    
    .result-badge.ai {{
        background: var(--danger);
        color: white;
    }}
    
    .result-badge.real {{
        background: var(--success);
        color: white;
    }}
    
    .result-body {{
        padding: 1.5rem 2rem 2rem;
    }}
    
    .confidence-section {{
        margin-bottom: 1.5rem;
    }}
    
    .confidence-header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 10px;
    }}
    
    .confidence-label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
    }}
    
    .confidence-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }}
    
    .confidence-bar {{
        height: 8px;
        background: var(--bg-elevated);
        border-radius: 100px;
        overflow: hidden;
    }}
    
    .confidence-fill {{
        height: 100%;
        border-radius: 100px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .confidence-fill.ai {{
        background: linear-gradient(90deg, var(--danger), #f97316);
    }}
    
    .confidence-fill.real {{
        background: linear-gradient(90deg, var(--success), #10b981);
    }}
    
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }}
    
    .stat-item {{
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }}
    
    .stat-value {{
        font-size: 1.125rem;
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
    
    /* ===== HISTORY ===== */
    .history-list {{
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}
    
    .history-item {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 14px;
        background: var(--bg-secondary);
        border-radius: 14px;
        transition: all 0.2s;
    }}
    
    .history-item:hover {{
        background: var(--bg-elevated);
    }}
    
    .history-thumb {{
        width: 56px;
        height: 56px;
        border-radius: 10px;
        object-fit: cover;
    }}
    
    .history-info {{
        flex: 1;
    }}
    
    .history-label {{
        font-size: 0.9375rem;
        font-weight: 600;
        color: var(--text-primary);
    }}
    
    .history-meta {{
        font-size: 0.8125rem;
        color: var(--text-muted);
        margin-top: 2px;
    }}
    
    .history-badge {{
        padding: 6px 12px;
        border-radius: 8px;
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
    
    /* ===== ABOUT PAGE ===== */
    .about-hero {{
        text-align: center;
        padding: 4rem 0;
    }}
    
    .about-title {{
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }}
    
    .about-desc {{
        font-size: 1.125rem;
        color: var(--text-secondary);
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.7;
    }}
    
    .features-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-top: 3rem;
    }}
    
    .feature-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s;
    }}
    
    .feature-card:hover {{
        border-color: var(--accent);
        transform: translateY(-4px);
    }}
    
    .feature-icon {{
        width: 64px;
        height: 64px;
        background: var(--accent-glow);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin: 0 auto 1.25rem;
    }}
    
    .feature-title {{
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }}
    
    .feature-desc {{
        font-size: 0.9375rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }}
    
    .tech-section {{
        margin-top: 4rem;
        padding: 3rem;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 24px;
    }}
    
    .tech-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    
    .tech-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }}
    
    .tech-item {{
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }}
    
    .tech-name {{
        font-size: 0.9375rem;
        font-weight: 600;
        color: var(--text-primary);
    }}
    
    .tech-desc {{
        font-size: 0.8125rem;
        color: var(--text-muted);
        margin-top: 4px;
    }}
    
    /* ===== STREAMLIT OVERRIDES ===== */
    [data-testid="stFileUploader"] section {{
        border: 2px dashed var(--border-color) !important;
        border-radius: 16px !important;
        padding: 2.5rem !important;
        background: var(--bg-secondary) !important;
    }}
    
    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--accent) !important;
    }}
    
    [data-testid="stFileUploader"] button {{
        background: var(--accent) !important;
        color: white !important;
        border-radius: 10px !important;
    }}
    
    .stButton > button {{
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
    }}
    
    .stButton > button:hover {{
        background: var(--accent-hover) !important;
    }}
    
    .stButton > button[kind="secondary"] {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stTextInput > div > div > input {{
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 14px 16px !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--accent) !important;
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
    
    @media (max-width: 1024px) {{
        .bento-grid {{
            grid-template-columns: 1fr;
        }}
        .bento-card.main, .bento-card.full {{
            grid-column: span 1;
        }}
        .features-grid, .tech-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
        .hero-title {{
            font-size: 2.25rem;
        }}
        .navbar, .main-content {{
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }}
    }}
    
    @media (max-width: 640px) {{
        .features-grid, .tech-grid, .stats-grid {{
            grid-template-columns: 1fr;
        }}
        .hero-title {{
            font-size: 1.75rem;
        }}
        .hero-subtitle {{
            font-size: 1rem;
        }}
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

# ============== HELPERS ==============
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    return np.array(image).reshape((1, 128, 128, 3)) / 255.0

def predict(model, img_array):
    prediction = model.predict(img_array, verbose=0)
    score = float(prediction[0][0])
    if score < 0.5:
        return "AI Generated", "🤖", (1 - score) * 100, score
    return "Real Image", "📷", score * 100, score

def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def image_to_base64(image, size=(600, 400)):
    img = image.copy().convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

def image_to_thumbnail(image):
    img = image.copy().convert("RGB")
    img.thumbnail((60, 60), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode()

# ============== NAVBAR ==============
theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"

st.markdown(f"""
<div class="navbar">
    <div class="navbar-inner">
        <div class="navbar-brand">
            <div class="brand-icon">⚡</div>
            <span class="brand-text">Authentic</span>
            <span class="brand-badge">Beta</span>
        </div>
        <div class="navbar-nav">
            <span class="nav-link {'active' if st.session_state.current_page == 'analyze' else ''}" id="nav-analyze">🔍 Analyze</span>
            <span class="nav-link {'active' if st.session_state.current_page == 'about' else ''}" id="nav-about">ℹ️ About</span>
            <span class="nav-link {'active' if st.session_state.current_page == 'history' else ''}" id="nav-history">📜 History</span>
        </div>
        <div class="navbar-actions">
            <span class="theme-btn">{theme_icon}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons (hidden but functional)
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, theme_col = st.columns([1, 1, 1, 1, 1, 1])

with nav_col2:
    if st.button("Analyze", key="nav_analyze_btn", type="secondary" if st.session_state.current_page != "analyze" else "primary"):
        st.session_state.current_page = "analyze"
        st.rerun()

with nav_col3:
    if st.button("About", key="nav_about_btn", type="secondary" if st.session_state.current_page != "about" else "primary"):
        st.session_state.current_page = "about"
        st.rerun()

with nav_col4:
    if st.button("History", key="nav_history_btn", type="secondary" if st.session_state.current_page != "history" else "primary"):
        st.session_state.current_page = "history"
        st.rerun()

with theme_col:
    if st.button(theme_icon, key="theme_btn"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# Load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# ============== ANALYZE PAGE ==============
if st.session_state.current_page == "analyze":
    st.markdown("""
    <div class="main-content">
        <div class="hero">
            <div class="hero-badge">⚡ Powered by Neural Networks</div>
            <h1 class="hero-title">Real <span>vs</span> AI Detection</h1>
            <p class="hero-subtitle">Upload any image and our CNN model will instantly determine if it's a real photograph or AI-generated content.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if model_loaded:
        # Bento grid layout
        col_left, col_right = st.columns([1.2, 0.8])
        
        with col_left:
            st.markdown("""
            <div class="bento-card">
                <div class="bento-header">
                    <div class="bento-icon">🔍</div>
                    <div>
                        <div class="bento-title">Upload Image</div>
                        <div class="bento-subtitle">Drag & drop or choose a file</div>
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
            
            if st.session_state.input_method == "upload":
                uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
                if uploaded_file:
                    image = Image.open(uploaded_file)
            
            elif st.session_state.input_method == "url":
                url = st.text_input("URL", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
                if st.button("🚀 Analyze URL", use_container_width=True):
                    if url:
                        try:
                            with st.spinner("Loading..."):
                                image = load_image_from_url(url)
                        except:
                            st.error("Failed to load image")
            
            elif st.session_state.input_method == "samples":
                sample_cols = st.columns(3)
                for idx, sample in enumerate(SAMPLE_IMAGES):
                    with sample_cols[idx]:
                        if st.button(f"{sample['emoji']} {sample['name']}", use_container_width=True):
                            try:
                                image = load_image_from_url(sample['url'])
                            except:
                                st.error("Failed")
            
            # Show result
            if image is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.spinner("🔍 Analyzing..."):
                    img_array = preprocess_image(image)
                    label, emoji, confidence, raw_score = predict(model, img_array)
                    is_ai = "AI" in label
                    
                    st.session_state.history.insert(0, {
                        "label": label,
                        "confidence": confidence,
                        "is_ai": is_ai,
                        "time": time.strftime("%H:%M"),
                        "thumbnail": image_to_thumbnail(image)
                    })
                    st.session_state.history = st.session_state.history[:10]
                
                img_b64 = image_to_base64(image)
                badge_class = "ai" if is_ai else "real"
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-image-container">
                        <img src="data:image/jpeg;base64,{img_b64}" class="result-image"/>
                        <div class="result-overlay">
                            <div class="result-badge {badge_class}">{emoji} {label}</div>
                        </div>
                    </div>
                    <div class="result-body">
                        <div class="confidence-section">
                            <div class="confidence-header">
                                <span class="confidence-label">Confidence Score</span>
                                <span class="confidence-value">{confidence:.1f}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill {badge_class}" style="width: {confidence}%;"></div>
                            </div>
                        </div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-value">{raw_score:.3f}</div>
                                <div class="stat-label">Raw Score</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{image.size[0]}×{image.size[1]}</div>
                                <div class="stat-label">Resolution</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{"High" if confidence > 80 else "Medium" if confidence > 60 else "Low"}</div>
                                <div class="stat-label">Certainty</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    st.button("👍 Correct", use_container_width=True, type="secondary")
                with fb_col2:
                    st.button("👎 Wrong", use_container_width=True, type="secondary")
        
        with col_right:
            st.markdown("""
            <div class="bento-card">
                <div class="bento-header">
                    <div class="bento-icon">📜</div>
                    <div>
                        <div class="bento-title">Recent</div>
                        <div class="bento-subtitle">Latest analyzed images</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.history:
                for item in st.session_state.history[:5]:
                    badge_class = "ai" if item.get("is_ai") else "real"
                    badge_text = "AI" if item.get("is_ai") else "Real"
                    st.markdown(f"""
                    <div class="history-item">
                        <img src="data:image/jpeg;base64,{item.get('thumbnail', '')}" class="history-thumb"/>
                        <div class="history-info">
                            <div class="history-label">{item.get('label', 'Unknown')}</div>
                            <div class="history-meta">{item.get('confidence', 0):.0f}% • {item.get('time', '')}</div>
                        </div>
                        <span class="history-badge {badge_class}">{badge_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No history yet")
    else:
        st.error("Model failed to load")

# ============== ABOUT PAGE ==============
elif st.session_state.current_page == "about":
    st.markdown("""
    <div class="main-content">
        <div class="about-hero">
            <h1 class="about-title">About Authentic</h1>
            <p class="about-desc">
                Authentic is an AI-powered tool that helps you distinguish between real photographs 
                and AI-generated images. Using a Convolutional Neural Network trained on thousands of images,
                we provide instant, accurate detection to help maintain trust in digital media.
            </p>
        </div>
        
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Deep Learning</div>
                <div class="feature-desc">Powered by a CNN trained on the GenImage dataset with 7 different AI generators.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Instant Results</div>
                <div class="feature-desc">Get detection results in milliseconds with confidence scores and explanations.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <div class="feature-title">Privacy First</div>
                <div class="feature-desc">Your images are processed locally and never stored on any server.</div>
            </div>
        </div>
        
        <div class="tech-section">
            <h2 class="tech-title">Technology Stack</h2>
            <div class="tech-grid">
                <div class="tech-item">
                    <div class="tech-name">TensorFlow</div>
                    <div class="tech-desc">Deep Learning</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">Keras</div>
                    <div class="tech-desc">Model API</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">Streamlit</div>
                    <div class="tech-desc">Web Framework</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">Python</div>
                    <div class="tech-desc">Backend</div>
                </div>
            </div>
        </div>
        
        <div class="tech-section" style="margin-top: 2rem;">
            <h2 class="tech-title">Training Data</h2>
            <p style="text-align: center; color: var(--text-secondary); margin-bottom: 1.5rem;">
                Our model was trained on the Tiny GenImage dataset, which includes images from:
            </p>
            <div class="tech-grid">
                <div class="tech-item">
                    <div class="tech-name">Midjourney</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">Stable Diffusion</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">DALL-E</div>
                </div>
                <div class="tech-item">
                    <div class="tech-name">BigGAN</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============== HISTORY PAGE ==============
elif st.session_state.current_page == "history":
    st.markdown("""
    <div class="main-content">
        <div class="about-hero">
            <h1 class="about-title">Analysis History</h1>
            <p class="about-desc">View all your recently analyzed images and their detection results.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        col1, col2 = st.columns([2, 1])
        with col1:
            for item in st.session_state.history:
                badge_class = "ai" if item.get("is_ai") else "real"
                badge_text = "AI Generated" if item.get("is_ai") else "Real Image"
                st.markdown(f"""
                <div class="history-item" style="margin-bottom: 12px;">
                    <img src="data:image/jpeg;base64,{item.get('thumbnail', '')}" class="history-thumb"/>
                    <div class="history-info">
                        <div class="history-label">{item.get('label', 'Unknown')}</div>
                        <div class="history-meta">{item.get('confidence', 0):.1f}% confidence • {item.get('time', '')}</div>
                    </div>
                    <span class="history-badge {badge_class}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            total = len(st.session_state.history)
            ai_count = sum(1 for h in st.session_state.history if h.get("is_ai"))
            real_count = total - ai_count
            
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-title" style="margin-bottom: 1rem;">Statistics</div>
                <div class="stats-grid" style="grid-template-columns: 1fr;">
                    <div class="stat-item">
                        <div class="stat-value">{total}</div>
                        <div class="stat-label">Total Analyzed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="color: var(--danger);">{ai_count}</div>
                        <div class="stat-label">AI Detected</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" style="color: var(--success);">{real_count}</div>
                        <div class="stat-label">Real Images</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("No images analyzed yet. Go to Analyze to get started!")

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem 0; color: var(--text-muted); font-size: 0.8125rem;">
    Authentic © 2026 • <a href="https://github.com/Gechen989898/AI_Art_vs_Human_Art" style="color: var(--accent);">GitHub</a>
</div>
""", unsafe_allow_html=True)
