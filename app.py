import os
import pickle
import datetime
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ---------------------------------------------------------------
# PAGE CONFIGURATION & BRANDING
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Plant Leaf Disease Detection System | Aror University",
    page_icon="🌿",
    layout="wide",
)

# ---- Session-state initialization (history tracking) ----------
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ---- Resolve university logo (handles multiple name variants) --
LOGO_FILE = None
for candidate in ["aror_logo.png", "Aror Logo.png", "aror logo.png"]:
    if os.path.exists(candidate):
        LOGO_FILE = candidate
        break

# ---------------------------------------------------------------
# TOP BANNER / HEADER
# ---------------------------------------------------------------
col_logo, col_title = st.columns([1, 4])

with col_logo:
    if LOGO_FILE:
        st.image(LOGO_FILE, use_container_width=True)
    else:
        st.write("🏛️")

with col_title:
    st.markdown(
        "### **Aror University of Art, Architecture, Design & Heritage, Sukkur, Sindh**"
    )
    st.markdown("#### **Digital Image Processing (DIP) – Semester V Project**")
    st.caption(
        "Department of Computer Science & Information Technology / Multimedia"
    )

st.markdown("---")
st.title("🌿 Automated Plant Leaf Disease Detection System")

# ---------------------------------------------------------------
# SIDEBAR WITH ACADEMIC DETAILS
# ---------------------------------------------------------------
with st.sidebar:
    if LOGO_FILE:
        st.image(LOGO_FILE, width=180)

    st.markdown("### **Project Credentials**")
    st.markdown(
        """
    * **Course:** Digital Image Processing
    * **Semester:** 5th Semester
    * **Institute:** Aror University, Sukkur, Sindh
    * **Domain:** Agricultural Computer Vision
    """
    )
    st.markdown("---")

    st.markdown("### **Pipeline Architecture**")
    st.markdown(
        """
    1. **Pre-processing:** Resize ($256 \\times 256$), Median Blur
    2. **Contrast Enhancement:** Histogram Equalization
    3. **Color Space:** RGB $\\rightarrow$ HSV Transformation
    4. **Segmentation:** Otsu Binary Thresholding + Morphology
    5. **Feature Extraction:** Color Moments + GLCM Textures
    6. **Classification:** Machine Learning & Area Severity Ratio
    """
    )

    st.markdown("---")
    st.markdown("### **🆕 Features**")
    st.markdown(
        """
    * 🔍 **Multi-Leaf Detection** — Detects & analyzes each leaf
    * 💊 **Remediation Suggestions** — Treatment advice
    * 📊 **History Tracking** — Session analysis log & CSV export
    """
    )

# ---------------------------------------------------------------
# REMEDIATION DATABASE
# ---------------------------------------------------------------
REMEDIATION_DB = {
    "healthy": {
        "status": "✅ Healthy",
        "advice": [
            "Continue regular watering and fertilization schedule.",
            "Monitor periodically for early signs of disease.",
            "Maintain proper spacing between plants for air circulation.",
        ],
    },
    "mild": {
        "status": "⚠️ Mild Infection (< 15 %)",
        "advice": [
            "Remove affected leaves carefully to prevent spread.",
            "Apply organic neem oil spray as a preventive measure.",
            "Ensure proper drainage to avoid waterlogging.",
            "Increase air circulation around the plant.",
            "Monitor closely over the next 5–7 days.",
        ],
    },
    "moderate": {
        "status": "🟠 Moderate Infection (15 – 40 %)",
        "advice": [
            "Prune all visibly infected leaves and branches immediately.",
            "Apply appropriate fungicide (copper-based recommended).",
            "Isolate the plant from healthy specimens if possible.",
            "Reduce overhead watering — use drip irrigation instead.",
            "Test soil pH and adjust if outside optimal range (6.0–7.0).",
            "Consider applying a balanced fertilizer to boost plant immunity.",
        ],
    },
    "severe": {
        "status": "🔴 Severe Infection (> 40 %)",
        "advice": [
            "⚠️ Immediate action required — heavy pruning of all infected parts.",
            "Apply systemic fungicide / bactericide as appropriate.",
            "Quarantine the plant completely from other crops.",
            "Disinfect all tools used on this plant (10 % bleach solution).",
            "Consider soil treatment or replacement if root rot is suspected.",
            "Consult a local agricultural extension officer for targeted guidance.",
            "If infection persists after 2 weeks of treatment, consider removing the plant.",
        ],
    },
}

DISEASE_SPECIFIC_REMEDIES = {
    "bacterial_spot": "Apply copper-based bactericide. Avoid overhead irrigation.",
    "early_blight": "Apply chlorothalonil or mancozeb fungicide. Practice crop rotation.",
    "late_blight": "Apply metalaxyl-based fungicide immediately. Remove all infected debris.",
    "leaf_mold": "Improve ventilation. Reduce humidity. Apply appropriate fungicide.",
    "septoria": "Remove infected leaves. Apply fungicide containing chlorothalonil.",
    "spider_mite": "Apply insecticidal soap or neem oil. Increase humidity around the plant.",
    "target_spot": "Apply fungicide. Ensure proper plant spacing for air circulation.",
    "yellow_leaf_curl": "Control whitefly vectors. Remove and destroy infected plants.",
    "mosaic_virus": "No cure — remove and destroy infected plants. Control aphid vectors.",
    "powdery_mildew": "Apply sulfur-based fungicide. Improve air circulation.",
    "rust": "Apply fungicide containing triadimefon. Remove infected leaves.",
    "downy_mildew": "Apply metalaxyl or copper-based fungicide. Avoid wet foliage.",
    "anthracnose": "Apply copper-based fungicide. Prune infected areas. Avoid overhead watering.",
    "scab": "Apply captan or myclobutanil fungicide. Remove fallen infected leaves.",
    "black_rot": "Prune infected canes. Apply mancozeb fungicide. Improve drainage.",
    "cedar_apple_rust": "Apply fungicide in spring. Remove nearby cedar trees if possible.",
    "leaf_scorch": "Ensure adequate watering. Provide shade during extreme heat. Mulch base.",
}


def _severity_level(infection_pct: float) -> str:
    if infection_pct < 3.0:
        return "healthy"
    elif infection_pct < 15.0:
        return "mild"
    elif infection_pct < 40.0:
        return "moderate"
    return "severe"


def get_remediation(infection_pct: float, pred_label: str | None = None):
    """Return (severity_key, info_dict, disease_specific_tip_or_None)."""
    severity = _severity_level(infection_pct)
    info = REMEDIATION_DB[severity]

    specific = None
    if pred_label:
        lbl = pred_label.lower().replace(" ", "_").replace("-", "_")
        for key, remedy in DISEASE_SPECIFIC_REMEDIES.items():
            if key in lbl or lbl in key:
                specific = remedy
                break

    return severity, info, specific


# ---------------------------------------------------------------
# PIPELINE & INFERENCE HELPERS
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


def process_pipeline(img_rgb: np.ndarray) -> dict:
    """Run the full DIP pipeline on a single leaf image."""
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
        "img": img,
        "gray": gray,
        "blur": blur,
        "eq": eq,
        "hsv": hsv,
        "leaf": leaf,
        "disease": disease,
        "segmented": segmented,
    }


def extract_features(p: dict):
    """Return (feature_vector, infection_percentage)."""
    leaf_px = p["leaf"] > 0
    total_leaf = max(1, int(leaf_px.sum()))
    if total_leaf < 100:
        leaf_px = np.ones_like(p["leaf"], dtype=bool)
        total_leaf = leaf_px.size

    hsv_data = p["hsv"][leaf_px].astype(float)
    color_feats = list(hsv_data.mean(axis=0)) + list(hsv_data.std(axis=0))

    quantized_gray = (p["eq"] // 8).astype(np.uint8)
    glcm = graycomatrix(
        quantized_gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2],
        levels=32,
        symmetric=True,
        normed=True,
    )
    texture_feats = [
        float(graycoprops(glcm, "contrast").mean()),
        float(graycoprops(glcm, "energy").mean()),
        float(graycoprops(glcm, "homogeneity").mean()),
        float(graycoprops(glcm, "correlation").mean()),
    ]

    disease_pixels = int((p["disease"] > 0).sum())
    infection_pct = (disease_pixels / total_leaf) * 100.0

    return np.array(color_feats + texture_feats + [infection_pct]), infection_pct


# ---------------------------------------------------------------
# MULTI-LEAF DETECTION
# ---------------------------------------------------------------
def detect_multiple_leaves(img_rgb: np.ndarray, min_area_ratio: float = 0.01):
    """Detect individual leaf regions via contour analysis.

    Returns (list_of_leaf_dicts, annotated_image).
    """
    img = cv2.resize(img_rgb, (512, 512))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Broad leaf mask
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    _, sat_mask = cv2.threshold(s_ch, 25, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(v_ch, 35, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(sat_mask, val_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = img.shape[0] * img.shape[1]
    min_area = total_area * min_area_ratio

    leaves: list[dict] = []
    annotated = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

        leaves.append(
            {"crop": img[y1:y2, x1:x2], "bbox": (x1, y1, x2, y2), "area": area}
        )

    # Sort largest-first and assign IDs
    leaves.sort(key=lambda l: l["area"], reverse=True)
    for idx, leaf in enumerate(leaves):
        leaf["id"] = idx + 1
        x1, y1, x2, y2 = leaf["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Leaf {leaf['id']}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return leaves, annotated


# ---------------------------------------------------------------
# HELPER: run single-leaf diagnosis & render remediation
# ---------------------------------------------------------------
def _diagnose_and_remediate(
    pipeline: dict, feats: np.ndarray, infection_pct: float, source_name: str
):
    """Render diagnostic results + remediation; return a history record dict."""
    pred_label: str | None = None
    confidence: float | None = None

    col_pct, col_result = st.columns([1, 2])
    col_pct.metric("Infected Surface Ratio", f"{infection_pct:.2f} %")

    if model and scaler:
        scaled_feats = scaler.transform([feats])
        pred_idx = int(model.predict(scaled_feats)[0])
        pred_label = class_names[pred_idx]
        probs = model.predict_proba(scaled_feats)[0]
        confidence = float(probs[pred_idx]) * 100.0

        if "healthy" in pred_label.lower():
            col_result.success(
                f"**Classification: {pred_label}**\n\nModel Confidence: {confidence:.2f} %"
            )
        else:
            col_result.error(
                f"**Classification: {pred_label}**\n\nModel Confidence: {confidence:.2f} %"
            )
    else:
        if infection_pct < 3.0:
            col_result.success(
                "**Diagnosis: Healthy**\n\nNo significant lesions detected."
            )
            pred_label = "Healthy"
        else:
            col_result.warning(
                f"**Diagnosis: Diseased ({infection_pct:.1f} % affected)**\n\n"
                "Pathogen presence identified."
            )
            pred_label = f"Diseased ({infection_pct:.1f} %)"

    # ---- Remediation ----
    severity, sev_info, specific = get_remediation(infection_pct, pred_label)

    st.markdown(f"#### 💊 {sev_info['status']}")
    st.progress(min(infection_pct / 100.0, 1.0), text=f"Infection Level: {infection_pct:.1f} %")

    st.markdown("**Recommended Actions:**")
    for tip in sev_info["advice"]:
        st.markdown(f"- {tip}")

    if specific:
        st.info(f"🎯 **Disease-Specific Treatment:** {specific}")

    # Build history record
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": source_name,
        "infection_pct": round(infection_pct, 2),
        "diagnosis": pred_label,
        "severity": severity,
        "confidence": round(confidence, 2) if confidence is not None else "N/A",
    }


def _append_history(record: dict):
    """Append record if it differs from the last entry (prevents dups on rerun)."""
    hist = st.session_state.analysis_history
    if not hist or (
        hist[-1].get("timestamp") != record["timestamp"]
        or hist[-1].get("filename") != record["filename"]
    ):
        hist.append(record)


# ---------------------------------------------------------------
# MAIN UI — THREE TABS
# ---------------------------------------------------------------
tab_single, tab_multi, tab_history = st.tabs(
    ["🔬 Single Leaf Analysis", "🍃 Multi-Leaf Detection", "📊 Analysis History"]
)

# ======================== SINGLE LEAF ========================= #
with tab_single:
    uploaded_file = st.file_uploader(
        "Upload Leaf Sample Image", type=["jpg", "jpeg", "png"], key="single"
    )

    if uploaded_file is not None:
        raw_img = np.array(Image.open(uploaded_file).convert("RGB"))
        pipeline = process_pipeline(raw_img)
        feats, infection_pct = extract_features(pipeline)

        # ---- Pipeline visualisation ----
        st.subheader("1️⃣ Pipeline Visualization")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(pipeline["img"], caption="Input Sample", use_container_width=True)
        c2.image(pipeline["gray"], caption="Grayscale", use_container_width=True)
        c3.image(pipeline["blur"], caption="Median Filtered", use_container_width=True)
        c4.image(pipeline["eq"], caption="Histogram Equalized", use_container_width=True)

        c5, c6, c7 = st.columns(3)
        c5.image(pipeline["leaf"], caption="Binary Leaf Mask", use_container_width=True)
        c6.image(
            pipeline["segmented"], caption="Isolated Leaf ROI", use_container_width=True
        )
        c7.image(
            pipeline["disease"], caption="Extracted Lesion Mask", use_container_width=True
        )

        # ---- Diagnosis + remediation ----
        st.subheader("2️⃣ Diagnostic Analysis & Remediation")
        record = _diagnose_and_remediate(
            pipeline, feats, infection_pct, uploaded_file.name
        )
        _append_history(record)

# ====================== MULTI-LEAF ============================ #
with tab_multi:
    st.markdown(
        "Upload an image containing **multiple leaves** and the system will "
        "detect, isolate, and analyze each leaf independently."
    )
    multi_file = st.file_uploader(
        "Upload Multi-Leaf Image", type=["jpg", "jpeg", "png"], key="multi"
    )

    if multi_file is not None:
        raw_multi = np.array(Image.open(multi_file).convert("RGB"))
        leaves, annotated_img = detect_multiple_leaves(raw_multi)

        st.subheader(f"Detected {len(leaves)} Leaf Region(s)")
        st.image(annotated_img, caption="Detected Leaf Regions", use_container_width=True)

        if len(leaves) == 0:
            st.warning(
                "No leaf regions detected. Try a clearer photo or different background."
            )
        else:
            for leaf_info in leaves:
                with st.expander(
                    f"🌿 Leaf #{leaf_info['id']} — Area: {leaf_info['area']:.0f} px²",
                    expanded=(leaf_info["id"] == 1),
                ):
                    lp = process_pipeline(leaf_info["crop"])
                    lf, li = extract_features(lp)

                    lc1, lc2, lc3 = st.columns(3)
                    lc1.image(
                        leaf_info["crop"],
                        caption="Cropped Leaf",
                        use_container_width=True,
                    )
                    lc2.image(
                        lp["segmented"],
                        caption="Segmented",
                        use_container_width=True,
                    )
                    lc3.image(
                        lp["disease"],
                        caption="Lesion Mask",
                        use_container_width=True,
                    )

                    rec = _diagnose_and_remediate(
                        lp,
                        lf,
                        li,
                        f"{multi_file.name} (Leaf #{leaf_info['id']})",
                    )
                    _append_history(rec)

# ====================== HISTORY =============================== #
with tab_history:
    st.subheader("📊 Analysis History")

    if not st.session_state.analysis_history:
        st.info("No analyses performed yet. Upload a leaf image to get started!")
    else:
        df = pd.DataFrame(st.session_state.analysis_history)
        df.columns = [
            "Timestamp",
            "Filename",
            "Infection %",
            "Diagnosis",
            "Severity",
            "Confidence",
        ]

        # Summary cards
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Analyses", len(df))
        healthy_n = int((df["Severity"] == "healthy").sum())
        sc2.metric("Healthy", healthy_n)
        sc3.metric("Diseased", len(df) - healthy_n)
        sc4.metric("Avg Infection %", f"{df['Infection %'].mean():.2f} %")

        st.dataframe(df, use_container_width=True, hide_index=True)

        # CSV download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download History as CSV",
            data=csv_bytes,
            file_name=f"leaf_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
