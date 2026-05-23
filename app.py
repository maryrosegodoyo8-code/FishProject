import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
# Use forward slashes (cross-platform compatible)
MODEL_PATH = "runs/detect/fish_detector/weights/best.pt"

# Verify the model exists
if not os.path.exists(MODEL_PATH):
    # Try alternative path if running from different directory
    alt_path = Path(__file__).parent / MODEL_PATH
    if alt_path.exists():
        MODEL_PATH = str(alt_path)

# --- PAGE SETUP ---
st.set_page_config(page_title="Fish Detection App", layout="wide")
st.title("🐟 Fish Detection System")
st.write("Upload an image of a fish to detect its location and class.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

# --- MAIN APP LOGIC ---
if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1. Read and Convert Image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # 2. Create Columns
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(image, width=400)

        # 3. Run Prediction
        with st.spinner("Detecting fish..."):
            results = model.predict(source=image_np, conf=confidence_threshold)

        # 4. Display Results
        with col2:
            st.header("Detection Result")
            annotated_image = results[0].plot(line_width=2)
            st.image(annotated_image, channels="BGR", width=400)

        # 5. Details
        with st.expander("🔍 View Detection Details"):
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"✅ **Detected:** {class_name} | **Confidence:** {confidence:.2f}")
            else:
                st.write("❌ No fish detected.")
else:
    st.error("Model could not be loaded. Check the path.")
