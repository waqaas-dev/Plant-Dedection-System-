import os
import pickle
import numpy as np
import opencv-python-headless
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ---------------------------------------------------------------
# PAGE CONFIGURATION & BRANDING
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Plant Leaf Disease Detection System | Aror University",
    page_icon="🌿",
    layout="wide"
)

# Top Banner / Header with University Logo
col_logo, col_title = st.columns([1, 4])

with col_logo:
    if os.path.exists("aror_logo.png"):
        st.image("aror_logo.png", use_container_width=True)
    else:
        st.write("🏛️")

with col_title:
    st.markdown("### **Aror University of Art, Architecture, Design & Heritage, Sukkur, Sindh**")
    st.markdown("#### **Digital Image Processing (DIP) – Semester V Project**")
    st.caption("Department of Computer Science & Information Technology / Multimedia")

st.markdown("---")
st.title("🌿 Automated Plant Leaf Disease Detection System")

# ---------------------------------------------------------------
# SIDEBAR WITH ACADEMIC DETAILS
# ---------------------------------------------------------------
with st.sidebar:
    if os.path.exists("aror_logo.png"):
        st.image("aror_logo.png", width=180)
    
    st.markdown("### **Project Credentials**")
    st.markdown("""
    * **Course:** Digital Image Processing
    * **Semester:** 5th Semester
    * **Institute:** Aror University, Sukkur, Sindh
    * **Domain:** Agricultural Computer Vision
    """)
    st.markdown("---")
    
    st.markdown("### **Pipeline Architecture**")
    st.markdown("""
    1. **Pre-processing:** Resize ($256 \\times 256$), Median Blur
    2. **Contrast Enhancement:** Histogram Equalization
    3. **Color Space:** RGB $\\rightarrow$ HSV Transformation
    4. **Segmentation:** Otsu Binary Thresholding + Morphology
    5. **Feature Extraction:** Color Moments + GLCM Textures
    6. **Classification:** Machine Learning & Area Severity Ratio
    """)

# ---------------------------------------------------------------
# PIPELINE AND INFERENCE CODE
# ---------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    if os.path.exists("leaf_model.pkl") and os.path.exists("scaler.pkl"):
        with open("leaf_model.pkl", "rb") as f:
            model_dict = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model_dict["model"], model_dict["classes"], scaler
    return None, None, None

model, class_names, scaler = load_trained_model()

def process_pipeline(img_rgb):
    img = cv2.resize(img_rgb, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(img, 5)
    
    blur_gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(blur_gray)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    _, sat_mask = cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY)
    leaf_raw = cv2.bitwise_and(sat_mask, val_mask)
    
    kernel_k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf = cv2.morphologyEx(leaf_raw, cv2.MORPH_CLOSE, kernel_k5)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, kernel_k5)
    
    healthy_green = cv2.inRange(hsv, (30, 40, 40), (85, 255, 255))
    disease_raw = cv2.bitwise_and(leaf, cv2.bitwise_not(healthy_green))
    kernel_k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    disease = cv2.morphologyEx(disease_raw, cv2.MORPH_OPEN, kernel_k3)
    
    segmented = cv2.bitwise_and(img, img, mask=leaf)
    
    return {
        "img": img, "gray": gray, "blur": blur, "eq": eq,
        "hsv": hsv, "leaf": leaf, "disease": disease, "segmented": segmented
    }

def extract_features(p):
    leaf_px = p["leaf"] > 0
    total_leaf = max(1, int(leaf_px.sum()))
    if total_leaf < 100:
        leaf_px = np.ones_like(p["leaf"], dtype=bool)
        total_leaf = leaf_px.size

    hsv_data = p["hsv"][leaf_px].astype(float)
    color_feats = list(hsv_data.mean(axis=0)) + list(hsv_data.std(axis=0))

    quantized_gray = (p["eq"] // 8).astype(np.uint8)
    glcm = graycomatrix(quantized_gray, distances=[1], angles=[0, np.pi/4, np.pi/2], 
                        levels=32, symmetric=True, normed=True)
    texture_feats = [
        float(graycoprops(glcm, "contrast").mean()),
        float(graycoprops(glcm, "energy").mean()),
        float(graycoprops(glcm, "homogeneity").mean()),
        float(graycoprops(glcm, "correlation").mean())
    ]

    disease_pixels = int((p["disease"] > 0).sum())
    infection_pct = (disease_pixels / total_leaf) * 100.0

    return np.array(color_feats + texture_feats + [infection_pct]), infection_pct

uploaded_file = st.file_uploader("Upload Leaf Sample Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = np.array(Image.open(uploaded_file).convert("RGB"))
    pipeline = process_pipeline(raw_img)
    feats, infection_pct = extract_features(pipeline)

    st.subheader("1. Pipeline Visualization")
    c1, c2, c3, c4 = st.columns(4)
    c1.image(pipeline["img"], caption="Input Sample", use_container_width=True)
    c2.image(pipeline["gray"], caption="Grayscale", use_container_width=True)
    c3.image(pipeline["blur"], caption="Median Filtered", use_container_width=True)
    c4.image(pipeline["eq"], caption="Histogram Equalized", use_container_width=True)

    c5, c6, c7 = st.columns(3)
    c5.image(pipeline["leaf"], caption="Binary Leaf Mask", use_container_width=True)
    c6.image(pipeline["segmented"], caption="Isolated Leaf ROI", use_container_width=True)
    c7.image(pipeline["disease"], caption="Extracted Lesion Mask", use_container_width=True)

    st.subheader("2. Diagnostic Analysis")
    col_pct, col_result = st.columns([1, 2])
    col_pct.metric("Infected Surface Ratio", f"{infection_pct:.2f}%")

    if model and scaler:
        scaled_feats = scaler.transform([feats])
        pred_idx = model.predict(scaled_feats)[0]
        pred_label = class_names[pred_idx]
        probs = model.predict_proba(scaled_feats)[0]
        confidence = probs[pred_idx] * 100.0

        if "healthy" in pred_label.lower():
            col_result.success(f"**Classification: {pred_label}**\n\nModel Confidence: {confidence:.2f}%")
        else:
            col_result.error(f"**Classification: {pred_label}**\n\nModel Confidence: {confidence:.2f}%")
    else:
        if infection_pct < 3.0:
            col_result.success("**Diagnosis: Healthy**\n\nNo significant lesions detected.")
        else:
            col_result.warning(f"**Diagnosis: Diseased ({infection_pct:.1f}% affected)**\n\nPathogen presence identified.")
