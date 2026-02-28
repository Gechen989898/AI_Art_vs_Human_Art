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
    page_title="AI Art vs Human Art",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state for history
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
    st.session_state.processed_images = {}  # hash -> {label, confidence, time}

# Load history from localStorage on first run (st_javascript is async)
if not st.session_state.history_loaded:
    stored_history = st_javascript("localStorage.getItem('ai_classifier_history')")
    
    # st_javascript returns 0 on first render, then actual data on rerender
    if stored_history == 0:
        # First render - JS hasn't executed yet, increment attempt counter
        st.session_state.load_attempt += 1
        if st.session_state.load_attempt < 3:
            time.sleep(0.1)
            st.rerun()
    else:
        # JS has returned - either data or null
        if stored_history and stored_history != "null" and isinstance(stored_history, str):
            try:
                loaded_history = json.loads(stored_history)
                if isinstance(loaded_history, list) and len(loaded_history) > 0:
                    st.session_state.history = loaded_history
            except (json.JSONDecodeError, TypeError):
                pass
        st.session_state.history_loaded = True

# Sample images for quick testing
SAMPLE_IMAGES = {
    "🌄 Landscape": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400",
    "🐕 Dog": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400",
    "🌸 Flower": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400",
}

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .ai-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    
    .human-result {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    
    /* Stats styling */
    .stat-box {
        background: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Upload area styling - Enhanced drop zone for dark mode */
    [data-testid="stFileUploader"] {
        border: 3px dashed #667eea !important;
        border-radius: 1.5rem !important;
        padding: 2rem !important;
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2 !important;
        background: linear-gradient(135deg, #2d2d3d 0%, #3d3d4d 100%) !important;
        transform: scale(1.01);
    }
    
    /* Fix file uploader text contrast */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: #ffffff !important;
    }
    
    /* File name in uploader */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: #e0e0e0 !important;
    }
    
    /* Uploaded file info */
    .uploadedFile {
        color: #ffffff !important;
    }
    
    .uploadedFileName {
        color: #ffffff !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #868e96;
        font-size: 0.8rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animated gauge */
    .gauge-container {
        position: relative;
        width: 200px;
        height: 200px;
        margin: 0 auto;
    }
    
    .gauge-bg {
        fill: none;
        stroke: #e9ecef;
        stroke-width: 20;
    }
    
    .gauge-fill {
        fill: none;
        stroke-width: 20;
        stroke-linecap: round;
        transform: rotate(-90deg);
        transform-origin: 50% 50%;
        animation: gaugeAnimation 1s ease-out forwards;
    }
    
    .gauge-fill-ai {
        stroke: url(#gradient-ai);
    }
    
    .gauge-fill-real {
        stroke: url(#gradient-real);
    }
    
    @keyframes gaugeAnimation {
        from { stroke-dashoffset: 283; }
    }
    
    .gauge-text {
        font-size: 2.5rem;
        font-weight: bold;
        fill: #495057;
    }
    
    .gauge-label {
        font-size: 0.9rem;
        fill: #868e96;
    }
    
    /* Sample image buttons */
    .sample-btn {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        background: white;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .sample-btn:hover {
        background: #f8f9fa;
        border-color: #667eea;
    }
    
    /* History item */
    .history-item {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-left: 4px solid;
    }
    
    .history-ai {
        border-left-color: #ff6b6b;
    }
    
    .history-real {
        border-left-color: #51cf66;
    }
    
    /* Smooth animations */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE = (128, 128)
MODEL_PATH = "models/basic_cnn.keras"


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
    # Calculate stroke-dashoffset based on actual radius
    radius = 42
    circumference = 2 * 3.14159 * radius  # ≈ 263.89
    offset = circumference - (confidence / 100 * circumference)
    
    color = "#ff6b6b" if is_ai else "#51cf66"
    
    svg = f"""
    <div style="position: relative; width: 180px; height: 180px; margin: 0 auto;">
        <svg width="180" height="180" viewBox="0 0 100 100" style="transform: rotate(-90deg);">
            <!-- Background circle -->
            <circle cx="50" cy="50" r="{radius}" fill="none" stroke="#3a3a3a" stroke-width="8"/>
            <!-- Progress circle -->
            <circle cx="50" cy="50" r="{radius}" fill="none" stroke="{color}" 
                    stroke-width="8" stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"/>
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
            <div style="font-size: 2.2rem; font-weight: bold; color: {color};">
                {confidence:.0f}%
            </div>
            <div style="font-size: 0.75rem; color: #888; margin-top: -5px;">confidence</div>
        </div>
    </div>
    """
    return svg


def get_last_conv_layer_name(model):
    """Find the name of the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam(model, img_array: np.ndarray, pred_class: int = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for the given image.
    
    Args:
        model: The trained Keras model
        img_array: Preprocessed image array (1, 128, 128, 3)
        pred_class: Class index to explain (None = use predicted class)
    
    Returns:
        Heatmap as numpy array (128, 128) with values 0-1
    """
    try:
        # Find the last conv layer
        last_conv_layer_name = get_last_conv_layer_name(model)
        if last_conv_layer_name is None:
            return None
        
        # Build the model if it hasn't been built yet (for Sequential models)
        if not model.built:
            model.build(input_shape=(None, 128, 128, 3))
        
        # For Sequential models, we need to create explicit input
        inputs = keras.Input(shape=(128, 128, 3))
        x = inputs
        
        # Rebuild the model with explicit input to get intermediate outputs
        last_conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                last_conv_output = x
        
        if last_conv_output is None:
            return None
        
        # Create a model that outputs both the conv layer output and final prediction
        grad_model = keras.Model(inputs=inputs, outputs=[last_conv_output, x])
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_class is None:
                # For binary classification: class 0 if prediction < 0.5, class 1 otherwise
                pred_class = 1 if predictions[0][0] >= 0.5 else 0
            
            # For binary sigmoid output, we use the prediction directly
            if pred_class == 1:
                class_output = predictions[0][0]  # Real photo (score towards 1)
            else:
                class_output = 1 - predictions[0][0]  # AI generated (score towards 0)
        
        # Compute gradients of the class output with respect to the conv layer
        grads = tape.gradient(class_output, conv_outputs)
        
        if grads is None:
            return None
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + keras.backend.epsilon())
        
        return heatmap.numpy()
    
    except Exception as e:
        # If Grad-CAM fails for any reason, return None gracefully
        print(f"Grad-CAM error: {e}")
        return None


def create_gradcam_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: Original PIL Image
        heatmap: Grad-CAM heatmap (H, W) with values 0-1
        alpha: Transparency for the heatmap overlay
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize image to match our model input for consistent overlay
    img = image.copy().convert("RGB")
    original_size = img.size
    
    # Resize heatmap to match original image size
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        original_size, Image.Resampling.BILINEAR
    )) / 255.0
    
    # Apply colormap (jet)
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored)
    
    # Blend with original image
    overlaid = Image.blend(img, heatmap_pil, alpha)
    
    return overlaid


def create_result_image(image: Image.Image, label: str, confidence: float) -> bytes:
    """Create a downloadable result image with prediction overlay"""
    # Create a copy and resize for consistent output
    img = image.copy()
    img = img.convert("RGB")
    
    # Resize to max 800px width while maintaining aspect ratio
    max_width = 800
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    draw = ImageDraw.Draw(img)
    
    # Create overlay banner at bottom
    banner_height = 60
    banner_y = img.height - banner_height
    
    # Semi-transparent banner
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    banner_color = (255, 107, 107, 220) if "AI" in label else (81, 207, 102, 220)
    overlay_draw.rectangle([0, banner_y, img.width, img.height], fill=banner_color)
    
    # Composite
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    
    draw = ImageDraw.Draw(img)
    
    # Add text
    text = f"{label} • {confidence:.1f}% confidence"
    # Use default font (works across systems)
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (img.width - text_width) // 2
    text_y = banner_y + (banner_height - 20) // 2
    draw.text((text_x, text_y), text, fill="white")
    
    # Convert to bytes
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
    # Keep only last 10 items
    st.session_state.history = st.session_state.history[:10]
    
    # Save to localStorage
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
    
    # Save to JSON file for persistence
    feedback_file = "feedback_log.json"
    try:
        import os
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                existing = json.load(f)
        else:
            existing = []
        
        # Don't save thumbnail to file (too large)
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
    # Escape single quotes for JavaScript
    history_json_escaped = history_json.replace("'", "\\'")
    st_javascript(f"localStorage.setItem('ai_classifier_history', '{history_json_escaped}')")


# ============== MAIN APP ==============

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 AI vs Real Image Classifier</h1>
    <p style="font-size: 1.2rem; color: #868e96;">
        Detect whether an image was generated by AI or is a real photograph
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    model_loaded = False

if model_loaded:
    # Create three columns for layout
    left_spacer, main_col, right_spacer = st.columns([1, 3, 1])
    
    with main_col:
        # Tabs for input method
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Image", "🔗 From URL", "🖼️ Try Examples", "📦 Batch Processing"])
        
        image = None
        
        with tab1:
            st.markdown("""
            <p style="text-align: center; color: #868e96; margin-bottom: 1rem;">
                📤 Drag & drop your image here or click to browse
            </p>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload",
                type=["jpg", "jpeg", "png", "webp"],
                help="Supported formats: JPG, JPEG, PNG, WEBP",
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        
        with tab2:
            col_url, col_btn = st.columns([4, 1])
            with col_url:
                url = st.text_input(
                    "Image URL",
                    placeholder="https://example.com/image.jpg",
                    label_visibility="collapsed"
                )
            with col_btn:
                load_btn = st.button("Load", use_container_width=True, type="primary")
            
            if url and load_btn:
                try:
                    with st.spinner("Loading image..."):
                        image = load_image_from_url(url)
                        st.success("✓ Image loaded!")
                except Exception as e:
                    st.error(f"Failed to load: {e}")
        
        with tab3:
            st.markdown("**Click a sample image to test the classifier:**")
            sample_cols = st.columns(len(SAMPLE_IMAGES))
            
            for idx, (name, sample_url) in enumerate(SAMPLE_IMAGES.items()):
                with sample_cols[idx]:
                    if st.button(name, use_container_width=True, key=f"sample_{idx}"):
                        try:
                            image = load_image_from_url(sample_url)
                            st.success(f"✓ Loaded {name}")
                        except Exception as e:
                            st.error(f"Failed: {e}")
        
        with tab4:
            st.markdown("""
            <p style="text-align: center; color: #868e96; margin-bottom: 1rem;">
                📦 Upload multiple images to analyze them all at once
            </p>
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
                
                col_process, col_clear = st.columns([1, 1])
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
                            
                            # Check for duplicate
                            img_bytes = img.tobytes()
                            img_hash = hashlib.md5(img_bytes).hexdigest()
                            is_duplicate = img_hash in st.session_state.processed_images
                            
                            img_array = preprocess_image(img)
                            label, emoji, confidence, raw_score = predict(model, img_array)
                            is_ai = "AI" in label
                            
                            # Cache the result
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
                st.markdown("### 📊 Batch Results")
                
                # Summary stats
                total = len(st.session_state.batch_results)
                ai_count = sum(1 for r in st.session_state.batch_results if r["is_ai"])
                real_count = total - ai_count
                dup_count = sum(1 for r in st.session_state.batch_results if r.get("is_duplicate", False))
                
                stat_cols = st.columns(4)
                stat_cols[0].metric("Total Images", total)
                stat_cols[1].metric("🤖 AI Generated", ai_count)
                stat_cols[2].metric("📷 Real Photos", real_count)
                stat_cols[3].metric("🔄 Duplicates", dup_count)
                
                st.markdown("---")
                
                # Results table with thumbnails
                for result in st.session_state.batch_results:
                    border_color = "#ff6b6b" if result["is_ai"] else "#51cf66"
                    is_dup = result.get("is_duplicate", False)
                    
                    col_thumb, col_info, col_conf = st.columns([1, 3, 2])
                    
                    with col_thumb:
                        if result["thumbnail"]:
                            st.markdown(f'<img src="data:image/jpeg;base64,{result["thumbnail"]}" style="width: 60px; height: 60px; object-fit: cover; border-radius: 8px; border: 2px solid {border_color};"/>', unsafe_allow_html=True)
                    
                    with col_info:
                        dup_badge = " 🔄" if is_dup else ""
                        st.markdown(f"**{result['filename']}**{dup_badge}")
                        st.markdown(f"{result['emoji']} {result['prediction']}")
                    
                    with col_conf:
                        st.markdown(f"""<div style="text-align: right;"><span style="background: {border_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: bold;">{result['confidence']}%</span></div>""", unsafe_allow_html=True)
                    
                    st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #3a3a3a;'>", unsafe_allow_html=True)
                
                if dup_count > 0:
                    st.caption(f"🔄 = Previously analyzed image ({dup_count} duplicate{'s' if dup_count > 1 else ''} found)")
                
                # Export to CSV button
                st.markdown("### 💾 Export Results")
                
                # Create CSV data
                csv_data = "Filename,Prediction,Confidence (%),Raw Score,Duplicate\n"
                for r in st.session_state.batch_results:
                    is_dup = "Yes" if r.get("is_duplicate", False) else "No"
                    csv_data += f"{r['filename']},{r['prediction']},{r['confidence']},{r['raw_score']},{is_dup}\n"
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Results section
    if image is not None:
        st.markdown("---")
        
        # Compute image hash for duplicate detection
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        
        # Check if image was already processed
        is_duplicate = img_hash in st.session_state.processed_images
        
        if is_duplicate:
            prev = st.session_state.processed_images[img_hash]
            st.warning(f"""
            ⚠️ **Duplicate Image Detected!**  
            This image was already analyzed at **{prev['time']}**  
            Previous result: **{prev['label']}** ({prev['confidence']:.1f}% confidence)
            """)
            
            reanalyze = st.checkbox("🔄 Re-analyze anyway", key=f"reanalyze_{img_hash}")
            if not reanalyze:
                st.info("👆 Check the box above to re-analyze this image")
                st.stop()
        
        # Analyze with animation
        with st.spinner("🔍 Analyzing image..."):
            img_array = preprocess_image(image)
            time.sleep(0.5)  # Brief pause for effect
            label, emoji, confidence, raw_score = predict(model, img_array)
            is_ai = "AI" in label
            
            # Generate Grad-CAM heatmap
            gradcam_heatmap = generate_gradcam(model, img_array)
            gradcam_overlay = None
            if gradcam_heatmap is not None:
                gradcam_overlay = create_gradcam_overlay(image, gradcam_heatmap)
            
            # Store in processed images cache
            st.session_state.processed_images[img_hash] = {
                "label": label,
                "confidence": confidence,
                "time": time.strftime("%H:%M:%S")
            }
            
            # Add to history with image thumbnail (only if not a re-analysis)
            if not is_duplicate:
                add_to_history(label, confidence, is_ai, image)
        
        # Results in columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 📷 Input Image")
            st.image(image, use_container_width=True)
            
            # Image info
            w, h = image.size
            st.caption(f"Size: {w} × {h} px")
            
            # Download button
            result_img = create_result_image(image, label, confidence)
            st.download_button(
                label="💾 Download Result",
                data=result_img,
                file_name=f"ai_detection_result.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col2:
            st.markdown("### 🎯 Prediction")
            
            # Large result display
            if is_ai:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%); 
                            padding: 2rem; border-radius: 1rem; text-align: center; color: white;
                            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);">
                    <div style="font-size: 4rem;">{emoji}</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #51cf66 0%, #40c057 100%); 
                            padding: 2rem; border-radius: 1rem; text-align: center; color: white;
                            box-shadow: 0 4px 15px rgba(81, 207, 102, 0.3);">
                    <div style="font-size: 4rem;">{emoji}</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### 📊 Confidence")
            
            # Animated gauge
            gauge_svg = create_gauge_svg(confidence, is_ai)
            st.markdown(gauge_svg, unsafe_allow_html=True)
            
            # Certainty indicator
            certainty = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
            cert_color = "#51cf66" if confidence > 80 else "#fcc419" if confidence > 60 else "#ff6b6b"
            st.markdown(f"""
            <p style="text-align: center; margin-top: 0.5rem;">
                <span style="background: {cert_color}; color: white; padding: 0.25rem 0.75rem; 
                             border-radius: 1rem; font-size: 0.9rem;">{certainty} certainty</span>
            </p>
            """, unsafe_allow_html=True)
            
            # Raw score details
            with st.expander("🔬 Technical Details"):
                st.markdown(f"""
                - **Raw Score:** `{raw_score:.4f}`
                - **Threshold:** `0.5`
                - **Distance from threshold:** `{abs(raw_score - 0.5):.4f}`
                """)
                
                # Mini bar chart
                st.markdown("**Score Distribution:**")
                ai_pct = (1 - raw_score) * 100
                real_pct = raw_score * 100
                st.progress(ai_pct / 100, text=f"AI: {ai_pct:.1f}%")
                st.progress(real_pct / 100, text=f"Real: {real_pct:.1f}%")
        
        # User Feedback Section
        st.markdown("---")
        st.markdown("### 💬 Was this prediction correct?")
        st.markdown("<p style='color: #868e96; font-size: 0.9rem;'>Your feedback helps us improve the model!</p>", unsafe_allow_html=True)
        
        # Generate unique prediction ID based on image hash and time
        prediction_id = f"{hash(image.tobytes()) % 10000}_{int(time.time())}"
        current_thumbnail = image_to_base64_thumbnail(image)
        
        # Store current prediction for feedback
        if st.session_state.current_prediction_id != prediction_id:
            st.session_state.current_prediction_id = prediction_id
            st.session_state.feedback_given = False
        
        if not st.session_state.feedback_given:
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])
            
            with fb_col1:
                if st.button("👍 Correct", use_container_width=True, type="primary"):
                    save_feedback(prediction_id, label, confidence, True, current_thumbnail)
                    st.session_state.feedback_given = True
                    st.rerun()
            
            with fb_col2:
                if st.button("👎 Wrong", use_container_width=True):
                    save_feedback(prediction_id, label, confidence, False, current_thumbnail)
                    st.session_state.feedback_given = True
                    st.rerun()
        else:
            st.success("✅ Thanks for your feedback!")
            
            # Show feedback stats
            if st.session_state.feedback_log:
                total_fb = len(st.session_state.feedback_log)
                correct_fb = sum(1 for f in st.session_state.feedback_log if f["is_correct"])
                accuracy = (correct_fb / total_fb * 100) if total_fb > 0 else 0
                st.caption(f"📊 Session accuracy: {accuracy:.0f}% ({correct_fb}/{total_fb} correct)")
        
        # Grad-CAM Explainability Section
        if gradcam_overlay is not None:
            st.markdown("---")
            st.markdown("### 🔥 Model Explainability (Grad-CAM)")
            st.markdown("""
            <p style="color: #868e96; font-size: 0.9rem;">
                The heatmap shows which regions the model focused on to make its prediction. 
                <b style="color: #ff6b6b;">Red/yellow</b> areas had the most influence.
            </p>
            """, unsafe_allow_html=True)
            
            gcam_col1, gcam_col2 = st.columns(2)
            
            with gcam_col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            with gcam_col2:
                st.markdown("**Attention Heatmap**")
                st.image(gradcam_overlay, use_container_width=True)
            
            with st.expander("💡 How to interpret Grad-CAM"):
                st.markdown("""
                **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizes which parts 
                of the image were most important for the model's prediction.
                
                - **Red/Yellow regions**: High importance - the model focused heavily on these areas
                - **Blue/Green regions**: Lower importance - less influential for the prediction
                - **Concentrated heat**: Model found specific discriminative features
                - **Diffuse heat**: Model looked at the overall image structure
                
                **Common patterns:**
                - AI images often show uniform attention or focus on texture artifacts
                - Real photos typically show attention on semantic objects (faces, animals, etc.)
                """)

    # Footer section
    st.markdown("---")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
            This classifier uses a **Convolutional Neural Network (CNN)** trained on the 
            [Tiny GenImage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) dataset.
            
            **AI Generators in training data:**
            - BigGAN, VQDM, Stable Diffusion v5
            - Wukong, ADM, Glide, Midjourney
            
            **Architecture:** 4 Conv2D layers → Dense → Sigmoid
            """)
    
    with col_info2:
        with st.expander("⚠️ Limitations"):
            st.markdown("""
            **Current model limitations:**
            - Trained on **nature/outdoor images** as the "real" class
            - May misclassify portraits, indoor photos, or artwork
            - Best accuracy with landscape & nature photography
            
            *Future versions: transfer learning + multi-class AI model detection*
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ by the AI Art Detection Team | 
        <a href="https://github.com/Gechen989898/AI_Art_vs_Human_Art">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


# ============== SIDEBAR - History (rendered last for fresh data) ==============
with st.sidebar:
    st.markdown("## 📜 Prediction History")
    
    if st.session_state.history:
        for item in st.session_state.history:
            # Ensure proper types (especially when loaded from localStorage)
            is_ai = item.get("is_ai", False)
            confidence = float(item.get("confidence", 0))
            label = item.get("label", "Unknown")
            item_time = item.get("time", "")
            thumbnail = item.get("thumbnail", "")
            
            border_color = "#ff6b6b" if is_ai else "#51cf66"
            emoji = "🤖" if is_ai else "📷"
            
            # Build thumbnail HTML if available
            thumbnail_html = ""
            if thumbnail:
                thumbnail_html = f'<img src="data:image/jpeg;base64,{thumbnail}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 4px; margin-right: 10px; float: left;"/>'
            
            st.markdown(f"""
            <div style="padding: 0.75rem; border-radius: 0.5rem; margin: 0.5rem 0; 
                        background: #2d2d2d; border-left: 4px solid {border_color}; overflow: hidden;">
                {thumbnail_html}
                <div style="overflow: hidden;">
                    <strong style="color: white;">{emoji} {label}</strong><br>
                    <small style="color: #aaa;">{confidence:.1f}% • {item_time}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.history_loaded = True  # Prevent reloading from localStorage
            # Clear localStorage too
            st_javascript("localStorage.removeItem('ai_classifier_history')")
            st.rerun()
    else:
        st.caption("No predictions yet. Upload an image to get started!")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")
    if st.session_state.history:
        ai_count = sum(1 for h in st.session_state.history if h["is_ai"])
        real_count = len(st.session_state.history) - ai_count
        col1, col2 = st.columns(2)
        col1.metric("🤖 AI", ai_count)
        col2.metric("📷 Real", real_count)
