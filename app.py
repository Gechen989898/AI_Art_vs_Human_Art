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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS - Theme aware
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
is_dark = st.session_state.theme == "dark"

# Theme colors
bg_primary = "#0a0a0a" if is_dark else "#fafafa"
bg_card = "#171717" if is_dark else "#ffffff"
bg_hover = "#262626" if is_dark else "#f4f4f5"
border_color = "#262626" if is_dark else "#e4e4e7"
text_primary = "#fafafa" if is_dark else "#09090b"
text_secondary = "#a1a1aa" if is_dark else "#52525b"
text_muted = "#71717a" if is_dark else "#a1a1aa"

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
    background: {text_primary};
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: {bg_primary};
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
    background: {text_primary} !important;
    color: {bg_primary} !important;
    border: none !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: {text_secondary} !important;
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
    background: {bg_hover};
    border: 1px solid {border_color};
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    color: {text_secondary};
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
    font-size: 16px;
    color: {text_secondary};
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.6;
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
    border-color: {text_muted} !important;
    background: {bg_hover} !important;
}}

[data-testid="stFileUploader"] button {{
    background: {text_primary} !important;
    color: {bg_primary} !important;
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
}}

.sample-btn > button:hover {{
    background: {bg_hover} !important;
    border-color: {text_muted} !important;
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
    {"name": "Mountain", "icon": "🏔", "url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400"},
    {"name": "Dog", "icon": "🐕", "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400"},
    {"name": "Flower", "icon": "🌷", "url": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400"},
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
            st.rerun()
        
        # Check if this is a multiclass result
        is_multiclass = "all_probs" in res
        
        if is_multiclass:
            # Multiclass result display - shows AI generator type
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
                        🎨 {res["label"].upper()}
                    </div>
                </div>
                <div class="result-body">
                    <div class="result-header">
                        <div>
                            <div class="result-title">AI Generator Identified</div>
                            <div class="result-subtitle">Detected: <strong style="text-transform: capitalize;">{res["label"]}</strong></div>
                        </div>
                        <div class="confidence-display">
                            <div class="confidence-number">{res["confidence"]:.1f}%</div>
                            <div class="confidence-label">Confidence</div>
                        </div>
                    </div>
                    <div style="margin-top: 16px;">
                        <div style="font-size: 12px; font-weight: 600; color: {text_muted}; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">All Generators Probability</div>
                        """ + probs_html + f"""
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
            """
            st.markdown(full_html, unsafe_allow_html=True)
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
    
    # Show upload form
    else:
        st.markdown("""
        <div class="hero">
            <div class="hero-label">◉ AI Detection</div>
            <h1 class="hero-title">Real or AI?</h1>
            <p class="hero-subtitle">Upload an image and know instantly if it was created by AI or captured with a camera.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selector
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 12px; font-weight: 600; color: {text_muted}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Select Detection Mode</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_mode1, col_mode2 = st.columns(2)
        with col_mode1:
            if st.button(
                "🔍 AI vs Real", 
                key="mode_binary",
                type="primary" if st.session_state.model_type == "binary" else "secondary",
                use_container_width=True,
                help="Detect if an image is AI-generated or real"
            ):
                st.session_state.model_type = "binary"
                st.rerun()
        
        with col_mode2:
            if st.button(
                "🎨 Identify Generator", 
                key="mode_multiclass",
                type="primary" if st.session_state.model_type == "multiclass" else "secondary",
                use_container_width=True,
                help="Identify which AI generator created the image"
            ):
                st.session_state.model_type = "multiclass"
                st.rerun()
        
        # Show mode description
        if st.session_state.model_type == "binary":
            st.markdown(f'<p style="text-align: center; font-size: 13px; color: {text_muted}; margin: 12px 0 20px;">Classifies images as AI-generated or real photographs</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="text-align: center; font-size: 13px; color: {text_muted}; margin: 12px 0 20px;">Identifies the AI generator: Midjourney, Stable Diffusion, GLIDE, and more</p>', unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        
        if uploaded:
            img = Image.open(uploaded)
            with st.spinner("Analyzing..."):
                # Use ViT models based on selection
                tensor = preprocess_vit(img, vit_transform)
                if st.session_state.model_type == "binary":
                    res = predict_vit_binary(vit_binary_model, tensor)
                else:
                    res = predict_vit_multiclass(vit_multiclass_model, tensor)
            st.session_state.analyzed_image = img
            st.session_state.result = res
            # Add to history
            thumb = img.copy()
            thumb.thumbnail((80, 80), Image.Resampling.LANCZOS)
            st.session_state.history.insert(0, {
                "thumb": img_to_b64(thumb, 80),
                "label": res["label"],
                "is_ai": res["is_ai"],
                "confidence": res["confidence"]
            })
            st.session_state.history = st.session_state.history[:8]  # Keep last 8
            st.rerun()
        
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
                        # Use ViT models based on selection
                        tensor = preprocess_vit(img, vit_transform)
                        if st.session_state.model_type == "binary":
                            res = predict_vit_binary(vit_binary_model, tensor)
                        else:
                            res = predict_vit_multiclass(vit_multiclass_model, tensor)
                        st.session_state.analyzed_image = img
                        st.session_state.result = res
                        # Add to history
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
        
        # History section
        if st.session_state.history:
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
            <p>State-of-the-art transformer architecture fine-tuned for AI image detection. Two modes: Binary (AI vs Real) and Multiclass (identify the specific AI generator).</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature 2
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎨</div>
        <div class="feature-content">
            <h3>Multi-Generator Detection</h3>
            <p>Identifies images from Midjourney, Stable Diffusion, GLIDE, BigGAN, ADM, VQDM, and Wukong generators.</p>
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
