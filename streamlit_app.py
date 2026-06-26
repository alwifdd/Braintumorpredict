import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import base64
import io

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,300;1,400&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,700&display=swap');

/* ── Force Light Mode ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stSidebarContent"], .main, .block-container,
[class*="st-"], [class*="css"] {
    color-scheme: light only !important;
}

:root {
    --background-color: #f0fafa !important;
    --text-color: #1a3a3a !important;
    --secondary-background-color: rgba(255,255,255,0.6) !important;
}

html { filter: none !important; }

[data-testid="stAppViewContainer"] {
    background: #f0fafa !important;
    color: #1a3a3a !important;
}

[data-testid="stApp"] {
    background: #f0fafa !important;
}

/* Force semua text tetap gelap */
p, span, div, label, h1, h2, h3, h4, h5, h6 {
    color: inherit;
}

/* Selectbox dark mode override */
[data-baseweb="select"] * {
    background-color: rgba(255,255,255,0.9) !important;
    color: #1a3a3a !important;
}
[data-baseweb="popover"] * {
    background-color: #ffffff !important;
    color: #1a3a3a !important;
}

/* File uploader dark mode override */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.45) !important;
    color: #1a3a3a !important;
}

/* Progress bar */
[data-testid="stProgress"] > div {
    background: rgba(46,196,182,0.15) !important;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: #f0fafa;
    color: #1a3a3a;
}

body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: 
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(46,196,182,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(160,196,255,0.22) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 60% 30%, rgba(255,214,165,0.15) 0%, transparent 55%),
        linear-gradient(160deg, #e8fafa 0%, #f5f8ff 50%, #fff8f0 100%);
    z-index: -1;
    pointer-events: none;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding-top: 1.5rem !important; 
    padding-bottom: 2rem !important;
    max-width: 1200px !important; 
    margin: 0 auto; 
}

.hero {
    background: rgba(255,255,255,0.45);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1.5px solid rgba(255,255,255,0.75);
    border-radius: 24px;
    padding: 32px 48px 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(46,196,182,0.06);
    margin-bottom: 16px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(46,196,182,0.12);
    border: 1.5px solid rgba(46,196,182,0.35);
    color: #1a9e96;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 100px;
    margin-bottom: 14px;
    position: relative; z-index: 2;
}

.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 8px;
    color: #0d3333;
    line-height: 1.1;
    position: relative; z-index: 2;
}

.hero h1 span {
    background: linear-gradient(135deg, #2EC4B6, #A0C4FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero p {
    font-size: 14px;
    color: #4a7070;
    margin: 0;
    font-weight: 400;
    max-width: 460px;
    line-height: 1.65;
    position: relative; z-index: 2;
}

.brain-scan-container {
    position: absolute; 
    top: 50%; 
    right: 48px; 
    transform: translateY(-50%);
    width: 120px;
    height: 120px;
    z-index: 3;
    display: flex;
    justify-content: center;
    align-items: center;
}

.brain-icon {
    font-size: 88px; 
    opacity: 0.45; 
    animation: pulseGlow 4s ease-in-out infinite;
    filter: drop-shadow(0 0 15px rgba(46,196,182,0.5));
}

.scan-line {
    position: absolute;
    width: 120%;
    height: 3px;
    background: #2EC4B6;
    box-shadow: 0 0 12px 2px rgba(46,196,182,0.8);
    border-radius: 50%;
    animation: scanning 2.5s ease-in-out infinite;
    z-index: 4;
}

@keyframes pulseGlow {
    0%, 100% { transform: scale(1); opacity: 0.45; }
    50% { transform: scale(1.05); opacity: 0.6; }
}

@keyframes scanning {
    0% { top: 10%; opacity: 0; }
    15% { opacity: 1; }
    85% { opacity: 1; }
    100% { top: 90%; opacity: 0; }
}

.status-row {
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(255,255,255,0.4);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid rgba(46,196,182,0.15);
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.status-loaded {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(46,196,182,0.12);
    border: 1.5px solid rgba(46,196,182,0.3);
    color: #1a9e96;
    font-size: 12px; font-weight: 600;
    padding: 5px 14px; border-radius: 100px;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #2EC4B6;
    box-shadow: 0 0 8px #2EC4B6; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 4px #2EC4B6; }
    50% { box-shadow: 0 0 12px #2EC4B6; }
}
.status-row .chip {
    font-size: 12px; color: #7aabab; font-weight: 500;
    background: rgba(160,196,255,0.15); border: 1px solid rgba(160,196,255,0.35);
    padding: 4px 12px; border-radius: 100px;
}

.card {
    background: rgba(255,255,255,0.60);
    backdrop-filter: blur(20px);
    border: 1.5px solid rgba(255,255,255,0.85);
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 8px 32px rgba(46,196,182,0.06);
}
.card-title {
    font-size: 11.5px; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; color: #8ab8b8; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.card-title::before {
    content: ''; display: inline-block; width: 4px; height: 16px;
    background: linear-gradient(180deg, #2EC4B6, #A0C4FF); border-radius: 3px;
}

.medical-warning {
    background: linear-gradient(135deg, rgba(255,200,50,0.18), rgba(255,160,30,0.12));
    border: 1.5px solid rgba(255,190,30,0.5);
    border-radius: 16px;
    padding: 14px 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.medical-warning-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.medical-warning-text { font-size: 12.5px; color: #7a5200; line-height: 1.6; font-weight: 500; }
.medical-warning-text strong { color: #5a3a00; font-weight: 700; font-size: 13px; }

.stSelectbox label { display: none !important; }
.stSelectbox > div > div {
    background: rgba(255,255,255,0.7) !important; 
    border: 1.5px solid rgba(46,196,182,0.25) !important;
    border-radius: 12px !important; 
    box-shadow: 0 2px 8px rgba(46,196,182,0.05) !important;
}
.model-desc-tag {
    display: inline-flex; gap: 6px; font-size: 12px; font-weight: 500; color: #4a8a8a;
    background: rgba(46,196,182,0.08); border: 1px solid rgba(46,196,182,0.2);
    padding: 4px 12px; border-radius: 100px; margin: -4px 0 16px;
}
.stFileUploader label { display: none !important; }
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.45) !important; 
    border: 2px dashed rgba(46,196,182,0.35) !important;
    border-radius: 16px !important; 
    padding: 20px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2EC4B6 0%, #A0C4FF 100%) !important; 
    color: white !important;
    border: none !important; 
    border-radius: 14px !important; 
    padding: 13px 32px !important;
    font-weight: 700 !important; 
    font-size: 15px !important;
    width: 100% !important; 
    box-shadow: 0 4px 15px rgba(46,196,182,0.3) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(46,196,182,0.45) !important;
}

.mri-scan-wrapper {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
    display: inline-block;
    width: 100%;
    background: #000;
}
.mri-scan-wrapper img { display: block; width: 100%; height: auto; border-radius: 12px; opacity: 0.88; }
.mri-scan-wrapper .scan-beam {
    position: absolute; left: 0; width: 100%; height: 4px;
    background: linear-gradient(90deg, transparent 0%, rgba(46,196,182,0.3) 10%, rgba(46,196,182,1) 45%, #7ffff4 50%, rgba(46,196,182,1) 55%, rgba(46,196,182,0.3) 90%, transparent 100%);
    box-shadow: 0 0 6px 2px rgba(46,196,182,0.9), 0 0 20px 6px rgba(46,196,182,0.5), 0 0 40px 10px rgba(46,196,182,0.2);
    animation: mriScan 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    z-index: 20; top: 0;
}
.mri-scan-wrapper .scan-trail {
    position: absolute; left: 0; width: 100%; height: 80px;
    background: linear-gradient(180deg, rgba(46,196,182,0.18) 0%, rgba(46,196,182,0.06) 60%, transparent 100%);
    animation: mriScanTrail 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    z-index: 19; top: 0; pointer-events: none;
}
.mri-scan-wrapper .scan-grid {
    position: absolute; inset: 0;
    background-image: repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(46,196,182,0.04) 3px, rgba(46,196,182,0.04) 4px);
    z-index: 18; border-radius: 12px; pointer-events: none;
}
.mri-scan-wrapper .scan-label {
    position: absolute; top: 12px; left: 12px;
    background: rgba(0,0,0,0.65); backdrop-filter: blur(8px);
    border: 1px solid rgba(46,196,182,0.5); color: #2EC4B6;
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 4px 10px; border-radius: 6px;
    z-index: 25; animation: blinkLabel 1s ease-in-out infinite;
}
.mri-scan-wrapper .corner {
    position: absolute; width: 18px; height: 18px;
    border-color: rgba(46,196,182,0.8); border-style: solid; z-index: 25;
}
.mri-scan-wrapper .corner-tl { top: 8px; left: 8px; border-width: 2px 0 0 2px; }
.mri-scan-wrapper .corner-tr { top: 8px; right: 8px; border-width: 2px 2px 0 0; }
.mri-scan-wrapper .corner-bl { bottom: 8px; left: 8px; border-width: 0 0 2px 2px; }
.mri-scan-wrapper .corner-br { bottom: 8px; right: 8px; border-width: 0 2px 2px 0; }

@keyframes mriScan {
    0%   { top: -1%; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 101%; opacity: 0; }
}
@keyframes mriScanTrail {
    0%   { top: -80px; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
@keyframes blinkLabel {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.result-box {
    margin-top: 14px; padding: 16px; 
    background: rgba(46,196,182,0.08); 
    border-radius: 14px; border: 1px solid rgba(46,196,182,0.2);
}

/* ── Probability Distribution Bar ── */
.prob-section {
    margin-top: 18px;
    padding: 18px 20px;
    background: rgba(255,255,255,0.55);
    border: 1.5px solid rgba(46,196,182,0.15);
    border-radius: 16px;
}
.prob-title {
    font-size: 11.5px; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; color: #8ab8b8; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.prob-title::before {
    content: '📊'; font-size: 14px; letter-spacing: 0;
}
.prob-row {
    display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
}
.prob-label {
    min-width: 130px; font-size: 12.5px; font-weight: 600; color: #2a5a5a;
    white-space: nowrap;
}
.prob-bar-bg {
    flex: 1; height: 10px; background: rgba(46,196,182,0.1);
    border-radius: 100px; overflow: hidden;
    border: 1px solid rgba(46,196,182,0.15);
}
.prob-bar-fill {
    height: 100%; border-radius: 100px;
    transition: width 0.6s ease;
}
.prob-pct {
    min-width: 46px; text-align: right;
    font-size: 13px; font-weight: 800; color: #1a7a72;
}
.prob-pct.highlight { color: #e05a5a; }

.img-hint { font-size: 11.5px; color: #9acaca; margin-top: -4px; font-weight: 500; }

.empty-state { text-align: center; padding: 48px 20px; opacity: 0.6; }
.empty-state .icon { font-size: 36px; }
.empty-state p { font-size: 13.5px; margin-top: 10px; color: #6a9090; }

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(46,196,182,0.2);
}
</style>
""", unsafe_allow_html=True)


# =========================
# CROP
# =========================
def crop_brain(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image_bgr
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return image_bgr[y:y+h, x:x+w]

# =========================
# BUILD MODELS
# =========================
def build_mobilenet():
    base = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)

def build_resnet():
    base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)

def build_effnet():
    base = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    mnet = build_mobilenet()
    mnet.load_weights("model/mobilenet_ft.weights.h5")
    resnet = build_resnet()
    resnet.load_weights("model/resnet_ft.weights.h5")
    effnet = build_effnet()
    effnet.load_weights("model/effnet_ft.weights.h5")
    return mnet, resnet, effnet

model_m, model_r, model_e = load_models()

# URUTAN SESUAI ABJAD
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

class_labels = {
    "glioma":     ("Glioma",         "#e05a5a"),
    "meningioma": ("Meningioma",     "#f0a050"),
    "pituitary":  ("Pituitary",      "#a070d0"),
    "notumor":    ("Tidak Ada Tumor","#2EC4B6"),
}

# Warna bar per kelas
class_bar_colors = {
    "glioma":     "linear-gradient(90deg, #e05a5a, #ff8a8a)",
    "meningioma": "linear-gradient(90deg, #f0a050, #ffd080)",
    "pituitary":  "linear-gradient(90deg, #a070d0, #c8a0f0)",
    "notumor":    "linear-gradient(90deg, #2EC4B6, #7ffff4)",
}

# =========================
# PREPROCESS & GRAD-CAM
# =========================
def preprocess(img_rgb, model_type):
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   
    img_bgr  = crop_brain(img_bgr)                         
    img_bgr  = cv2.resize(img_bgr, (224, 224))
    img_rgb2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    
    img_display = img_rgb2.copy()                           
    img_f = img_rgb2.astype(np.float32)
    if model_type == "mobilenet":
        img_f = tf.keras.applications.mobilenet_v2.preprocess_input(img_f)
    elif model_type == "resnet":
        img_f = tf.keras.applications.resnet50.preprocess_input(img_f)
    else:
        img_f = tf.keras.applications.efficientnet.preprocess_input(img_f)
    return np.expand_dims(img_f, axis=0), img_display

def make_gradcam_heatmap(img_input, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0: heatmap /= heatmap.max()
    return heatmap

def overlay_gradcam(img_rgb_uint8, heatmap, alpha=0.4):
    h, w = img_rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb_uint8, 1 - alpha, heatmap_color_rgb, alpha, 0)

def run_model(model, img, model_type, layer_name):
    img_input, img_display = preprocess(img, model_type)
    pred = model.predict(img_input, verbose=0)
    pred_class = np.argmax(pred)
    confidence = float(pred[0][pred_class])
    # Kembalikan seluruh prob distribution
    prob_dist = {class_names[i]: float(pred[0][i]) for i in range(len(class_names))}
    heatmap = make_gradcam_heatmap(img_input, model, layer_name)
    overlay = overlay_gradcam(img_display, heatmap)
    return overlay, class_names[pred_class], confidence, prob_dist

def img_to_b64(np_img):
    pil = Image.fromarray(np_img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def render_prob_distribution(prob_dist, top_class):
    """Render distribusi probabilitas 4 kelas pakai native Streamlit — no HTML."""
    st.markdown("**📊 Distribusi Probabilitas**")
    sorted_items = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_items:
        display_name, color = class_labels[cls]
        pct = prob * 100
        is_top = (cls == top_class)
        col_label, col_bar, col_pct = st.columns([2, 5, 1])
        with col_label:
            if is_top:
                st.markdown(f"**{display_name}**")
            else:
                st.markdown(display_name)
        with col_bar:
            st.progress(float(prob))
        with col_pct:
            if is_top:
                st.markdown(f"**{pct:.1f}%**")
            else:
                st.markdown(f"{pct:.1f}%")

MODEL_OPTIONS = {
    "EfficientNetB0": {"model": lambda: model_e, "type": "effnet",    "layer": "top_conv",           "desc": "EfficientNetB0 · Fine-tuned", "color": "#2EC4B6", "emoji": "✨"},
    "ResNet50":       {"model": lambda: model_r, "type": "resnet",    "layer": "conv5_block3_out",   "desc": "ResNet50 · Fine-tuned",       "color": "#FFD6A5", "emoji": "🔬"},
    "MobileNetV2":    {"model": lambda: model_m, "type": "mobilenet", "layer": "Conv_1",             "desc": "MobileNetV2 · Fine-tuned",    "color": "#A0C4FF", "emoji": "⚡"},
    "Semua Model":    {"model": None,            "type": None,        "layer": None,                 "desc": "Bandingkan ketiga model secara bersamaan", "color": "#2EC4B6", "emoji": "🧩"},
}

# ══════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 Deep Learning &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp; Analisis MRI Otak</div>
    <h1>Brain<span>Scan</span> XAI</h1>
    <p>Alat bantu skrining tumor otak dari citra MRI menggunakan deep learning dengan visualisasi Grad-CAM, dirancang untuk mendukung tenaga medis dan pasien.</p>
    <div class="brain-scan-container">
        <div class="brain-icon">🧠</div>
        <div class="scan-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="status-row">
    <div class="status-loaded">
        <div class="status-dot"></div>
        3 Model siap digunakan
    </div>
    <span class="chip">EfficientNetB0</span>
    <span class="chip">ResNet50</span>
    <span class="chip">MobileNetV2</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="medical-warning">
    <div class="medical-warning-icon">⚠️</div>
    <div class="medical-warning-text">
        <strong>Peringatan Medis:</strong> Hasil analisis ini bersifat bantuan skrining awal berbasis kecerdasan buatan dan <strong>bukan merupakan diagnosis medis</strong>. 
        Selalu konsultasikan hasil dengan dokter spesialis radiologi atau neurologi berpengalaman sebelum mengambil keputusan klinis apapun. 
        Sistem ini dibangun semata-mata untuk tujuan <strong>penelitian dan edukasi</strong>.
    </div>
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""<div class="card" style="margin-bottom: 14px;">
        <div class="card-title">Pilih Model Analisis</div>""", unsafe_allow_html=True)

    selected_model_name = st.selectbox("Model", options=list(MODEL_OPTIONS.keys()), index=0, label_visibility="collapsed")
    cfg = MODEL_OPTIONS[selected_model_name]
    
    st.markdown(f"""
        <div class="model-desc-tag">
            <span>{cfg['emoji']}</span>
            <span style="color:{cfg['color']}; font-weight:700;">{selected_model_name}</span>
            <span style="color:#9acaca;">·</span>
            <span>{cfg['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-bottom: 14px;"><div class="card-title">Upload Citra MRI</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("MRI", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    st.markdown('<p class="img-hint">Format JPG, PNG · Resolusi minimum 224×224 px</p>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        st.image(img_np, caption="📷 Citra MRI yang diunggah", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:4px;'>", unsafe_allow_html=True)
    
    if not uploaded_file:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #b8cccc 0%, #c5d5e8 100%);
            color: rgba(255,255,255,0.75);
            border: none; border-radius: 14px; padding: 13px 32px;
            font-weight: 700; font-size: 15px; width: 100%; text-align: center;
            cursor: not-allowed; box-shadow: none;
            font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: 0.01em;
        ">
            🔍 Analisis Sekarang
        </div>
        <p style="text-align:center; font-size:11.5px; color:#a0baba; margin-top:8px; font-weight:500;">
            Upload citra MRI terlebih dahulu
        </p>
        """, unsafe_allow_html=True)
        predict_clicked = False
    else:
        predict_clicked = st.button("🔍 Analisis Sekarang", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="card"><div class="card-title">Hasil Analisis Grad-CAM</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🔬</div>
            <p>Upload citra MRI di sebelah kiri untuk memulai analisis.</p>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file and not predict_clicked:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">⚡</div>
            <p>Klik <b>Analisis Sekarang</b> untuk menjalankan model deep learning.</p>
        </div>
        """, unsafe_allow_html=True)

    elif predict_clicked:
        img_b64 = img_to_b64(img_np)

        scan_placeholder = st.empty()
        scan_placeholder.markdown(f"""
        <div style="margin-bottom:8px; text-align:center;">
            <span style="font-size:11px; font-weight:700; color:#2EC4B6; 
                         letter-spacing:0.12em; text-transform:uppercase;">
                ● Memindai citra MRI...
            </span>
        </div>
        <div class="mri-scan-wrapper">
            <img src="data:image/png;base64,{img_b64}" />
            <div class="scan-beam"></div>
            <div class="scan-trail"></div>
            <div class="scan-grid"></div>
            <div class="scan-label">SCANNING</div>
            <div class="corner corner-tl"></div>
            <div class="corner corner-tr"></div>
            <div class="corner corner-bl"></div>
            <div class="corner corner-br"></div>
        </div>
        """, unsafe_allow_html=True)

        if selected_model_name == "Semua Model":
            inference_result = [
                ("EfficientNetB0", run_model(model_e, img_np, "effnet",    "top_conv")),
                ("ResNet50",       run_model(model_r, img_np, "resnet",    "conv5_block3_out")),
                ("MobileNetV2",    run_model(model_m, img_np, "mobilenet", "Conv_1")),
            ]
        else:
            inference_result = run_model(cfg["model"](), img_np, cfg["type"], cfg["layer"])

        scan_placeholder.empty()

        # ── Semua Model ──
        if selected_model_name == "Semua Model":
            st.markdown("✅ **Perbandingan 3 Model**")
            st.markdown("---")

            for name, (overlay, label, conf, prob_dist) in inference_result:
                display_label, label_color = class_labels[label]
                conf_color = "#2EC4B6" if label == "notumor" else "#e05a5a"

                st.markdown(f"#### 🔬 {name}")

                img_col, pred_col = st.columns([1, 1])
                with img_col:
                    st.image(overlay, caption="🌡️ Grad-CAM Heatmap", use_container_width=True)
                with pred_col:
                    st.markdown(f"**Prediksi:** {display_label}")
                    st.markdown(f"**Confidence:**")
                    st.markdown(f"<span style='font-size:28px; font-weight:900; color:{conf_color};'>{conf*100:.1f}%</span>", unsafe_allow_html=True)

                # Distribusi full width di bawah gambar
                render_prob_distribution(prob_dist, label)
                st.markdown("---")

        # ── Single Model ──
        else:
            overlay, label, conf, prob_dist = inference_result
            display_label, label_color = class_labels[label]
            conf_color = "#2EC4B6" if label == "notumor" else "#e05a5a"
            
            st.markdown("""
            <div style="font-size:12px; font-weight:600; color:#2EC4B6; letter-spacing:0.06em; 
                        text-transform:uppercase; margin-bottom:10px; text-align:center;">
                ✅ Hasil Analisis
            </div>
            """, unsafe_allow_html=True)
            
            img_col, gcam_col = st.columns(2)
            with img_col:
                st.image(img_np, caption="📷 Citra Asli", use_container_width=True)
            with gcam_col:
                st.image(overlay, caption="🌡️ Grad-CAM Heatmap", use_container_width=True)
            
            # Hasil prediksi
            st.markdown(f"""
            <div class="result-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:11px; color:#4a7070; font-weight:700; 
                                    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
                            Hasil Prediksi
                        </div>
                        <div style="display:inline-block; padding:7px 18px; border-radius:100px; 
                                    font-size:14px; font-weight:700; background:rgba(46,196,182,0.15); 
                                    border:1.5px solid rgba(46,196,182,0.35); color:#1a7a72;">
                            {display_label}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px; color:#4a7070; font-weight:700; 
                                    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
                            Confidence
                        </div>
                        <div style="font-size:32px; font-weight:800; color:{conf_color}; line-height:1;">
                            {conf*100:.1f}%
                        </div>
                    </div>
                </div>
                <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(46,196,182,0.15);
                            font-size:11.5px; color:#6a9090; line-height:1.6;">
                    🌡️ Heatmap Grad-CAM menampilkan area yang paling berpengaruh terhadap keputusan prediksi model.
                    Area merah/kuning menunjukkan fokus utama model.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── DISTRIBUSI PROBABILITAS 4 KELAS ──
            render_prob_distribution(prob_dist, label)

    st.markdown("</div>", unsafe_allow_html=True)