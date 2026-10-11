"""
PANCREAS HEALTH SCREENER - Streamlit Web App (2D Images)
Run with: streamlit run streamlit_app_2d.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path

st.set_page_config(
    page_title="Pancreas Health Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_image(image):
    """Normalize image to 0-1 range"""
    return image.astype(np.float32) / 255.0

class SegmentationAnalyzer:
    def analyze(self, image, prediction):
        """Analyze segmentation result"""
        
        pred_binary = prediction > 0.5
        pred_area = np.sum(pred_binary)
        total_area = image.size
        
        image_region = image[pred_binary]
        variance = np.var(image_region) if len(image_region) > 0 else 0
        
        coverage = (pred_area / total_area) * 100
        
        # Risk assessment
        risk_score = 0
        flags = []
        
        if coverage < 5:
            flags.append("🔴 Very low segmentation coverage")
            risk_score += 2
        elif coverage > 60:
            flags.append("🟡 High segmentation coverage")
            risk_score += 1
        
        if variance > 0.3:
            flags.append("🟡 High pixel variance in segmented region")
            risk_score += 1
        
        if risk_score >= 3:
            risk_level = "🔴 HIGH RISK"
        elif risk_score >= 1:
            risk_level = "🟡 MODERATE RISK"
        else:
            risk_level = "🟢 LOW RISK"
        
        return {
            'pred_area': pred_area,
            'coverage': coverage,
            'variance': variance,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'flags': flags if flags else ["✓ Normal segmentation"]
        }

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    model_path = './models/pancreas_2d_unet.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

# ============================================================================
# UI
# ============================================================================

st.title("🏥 Pancreas Health Screener")
st.markdown("""
**Automated Medical Image Segmentation using Deep Learning**

- 2D U-Net segmentation model
- Automated area & risk analysis
- Clinical summary with health assessment
""")

st.markdown("---")

# Load model
model = load_model()

if model is None:
    st.error("❌ Model not found!")
    st.info("Train the model first: Run in terminal: python pancreas_2d_unet.py")
    st.stop()

st.success("✓ Model loaded successfully")

# ============================================================================
# FILE UPLOAD
# ============================================================================

st.subheader("📤 Upload Medical Image")
uploaded_file = st.file_uploader(
    "Upload PNG image file",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is None:
    st.info("👈 Upload a medical image to begin analysis")
    st.stop()

# ============================================================================
# PROCESSING
# ============================================================================

st.subheader("⏳ Processing...")
progress = st.progress(0)

with tempfile.TemporaryDirectory() as tmpdir:
    temp_path = os.path.join(tmpdir, 'temp_image.png')
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    progress.progress(20)
    
    # Load image
    image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error("❌ Could not read image file")
        st.stop()
    
    original_shape = image.shape
    
    progress.progress(40)
    
    # Preprocess
    image_resized = cv2.resize(image, (192, 192))
    image_normalized = normalize_image(image_resized)
    
    progress.progress(60)
    
    # Predict
    input_batch = image_normalized[np.newaxis, ..., np.newaxis]
    prediction = model.predict(input_batch, verbose=0)
    segmentation = prediction[0, :, :, 0]
    
    progress.progress(80)
    
    # Analyze
    analyzer = SegmentationAnalyzer()
    results = analyzer.analyze(image_normalized, segmentation)
    
    progress.progress(100)

st.success("✓ Analysis complete")
st.markdown("---")

# ============================================================================
# RESULTS
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Segmented Area",
        f"{results['pred_area']} pixels",
        help="Number of pixels identified as pancreas"
    )

with col2:
    st.metric(
        "Coverage",
        f"{results['coverage']:.1f}%",
        help="Percentage of image segmented"
    )

with col3:
    st.metric(
        "Variance",
        f"{results['variance']:.3f}",
        help="Pixel intensity variance in segmented region"
    )

st.markdown("---")

# Risk Assessment
risk = results
st.markdown(f"### {risk['risk_level']}")
st.write(f"**Risk Score:** {risk['risk_score']:.1f}")

st.subheader("Clinical Assessment")
for flag in risk['flags']:
    st.write(flag)

st.markdown("---")

# ============================================================================
# VISUALIZATION
# ============================================================================

st.subheader("🔍 Segmentation Visualization")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#f0f2f6')

# Original image
axes[0].imshow(image_normalized, cmap='gray')
axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
axes[0].axis('off')

# Segmentation mask
axes[1].imshow(segmentation, cmap='RdYlGn')
axes[1].set_title("Segmentation Mask", fontsize=12, fontweight='bold')
axes[1].axis('off')

# Overlay
axes[2].imshow(image_normalized, cmap='gray')
axes[2].imshow(segmentation, cmap='Reds', alpha=0.6)
axes[2].set_title("Overlay (Red = Segmented)", fontsize=12, fontweight='bold')
axes[2].axis('off')

st.pyplot(fig)

st.markdown("---")

st.subheader("📊 Image Details")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Original Size:** {original_shape[0]}×{original_shape[1]} pixels")
    st.write(f"**Processed Size:** 192×192 pixels")

with col2:
    st.write(f"**Image Min Intensity:** {image_normalized.min():.3f}")
    st.write(f"**Image Max Intensity:** {image_normalized.max():.3f}")

st.markdown("---")

st.info("⚠️ **Disclaimer:** This is a research tool for educational purposes. Clinical decisions require qualified medical professionals.")