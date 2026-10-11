import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Pancreas Health Screener", layout="wide")

def normalize_image(image):
    return image.astype(np.float32) / 255.0

class SegmentationAnalyzer:
    def analyze(self, image, prediction):
        pred_binary = prediction > 0.5
        pred_area = np.sum(pred_binary)
        coverage = (pred_area / image.size) * 100
        image_region = image[pred_binary]
        variance = np.var(image_region) if len(image_region) > 0 else 0
        
        risk_score = 0
        flags = []
        
        if coverage < 5:
            flags.append("🔴 Very low segmentation coverage")
            risk_score += 2
        elif coverage > 60:
            flags.append("🟡 High segmentation coverage")
            risk_score += 1
        
        if variance > 0.3:
            flags.append("🟡 High pixel variance")
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
            'flags': flags if flags else ["✓ Normal"]
        }

@st.cache_resource
def load_model():
    model_path = './models/pancreas_2d_unet.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

st.title("🏥 Pancreas Health Screener")
st.markdown("**2D U-Net Medical Image Segmentation**")
st.markdown("---")

model = load_model()

if model is None:
    st.error("❌ Model not found!")
    st.stop()

st.success("✓ Model loaded")

st.subheader("📤 Upload Medical Image")
uploaded_file = st.file_uploader("Upload PNG image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is None:
    st.info("👈 Upload an image")
    st.stop()

st.subheader("⏳ Processing...")
progress = st.progress(0)

with tempfile.TemporaryDirectory() as tmpdir:
    temp_path = os.path.join(tmpdir, 'temp.png')
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    progress.progress(20)
    image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    original_shape = image.shape
    
    progress.progress(40)
    image_resized = cv2.resize(image, (192, 192))
    image_normalized = normalize_image(image_resized)
    
    progress.progress(60)
    input_batch = image_normalized[np.newaxis, ..., np.newaxis]
    prediction = model.predict(input_batch, verbose=0)
    segmentation = prediction[0, :, :, 0]
    
    progress.progress(80)
    analyzer = SegmentationAnalyzer()
    results = analyzer.analyze(image_normalized, segmentation)
    progress.progress(100)

st.success("✓ Complete")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Area", f"{results['pred_area']} px")
with col2:
    st.metric("Coverage", f"{results['coverage']:.1f}%")
with col3:
    st.metric("Variance", f"{results['variance']:.3f}")

st.markdown("---")
st.markdown(f"### {results['risk_level']}")
st.write(f"Score: {results['risk_score']:.1f}")

st.subheader("Flags")
for flag in results['flags']:
    st.write(flag)

st.markdown("---")
st.subheader("Visualization")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_normalized, cmap='gray')
axes[0].set_title("Image")
axes[0].axis('off')
axes[1].imshow(segmentation, cmap='RdYlGn')
axes[1].set_title("Mask")
axes[1].axis('off')
axes[2].imshow(image_normalized, cmap='gray')
axes[2].imshow(segmentation, cmap='Reds', alpha=0.6)
axes[2].set_title("Overlay")
axes[2].axis('off')
st.pyplot(fig)

st.info("⚠️ Educational tool only")
