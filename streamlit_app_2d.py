
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Pancreas Health Screener",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Pancreas Health Screener")
st.subheader("2D U-Net Medical Image Segmentation")

st.write(
    "Upload a 2D CT image to generate a segmentation prediction "
    "using your trained U-Net model."
)


# -----------------------------
# Find the trained model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_EXTENSIONS = {".h5", ".hdf5", ".keras"}


def find_model():
    if not MODEL_DIR.exists():
        return None

    model_files = [
        path for path in MODEL_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
    ]

    if not model_files:
        return None

    # Prefer .h5 files if multiple models are present.
    model_files.sort(
        key=lambda path: (
            path.suffix.lower() != ".h5",
            path.name.lower()
        )
    )

    return model_files[0]


@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path, compile=False)


model_file = find_model()
model = None

if model_file is None:
    st.error("❌ Model not found!")
    st.info(
        "Place your trained .h5 model inside the models folder "
        "and push that folder to GitHub."
    )
    st.code(
        "pancreas-ct-segmentation/\n"
        "├── streamlit_app_2d.py\n"
        "├── requirements.txt\n"
        "└── models/\n"
        "    └── your_model.h5"
    )
    st.stop()

try:
    model = load_trained_model(str(model_file))
    st.success(f"✅ Model loaded: {model_file.name}")
except Exception as error:
    st.error("❌ The model file was found but could not be loaded.")
    st.code(str(error))
    st.stop()


# -----------------------------
# Display model input information
# -----------------------------
st.write("Model input shape:", model.input_shape)
st.write("Model output shape:", model.output_shape)


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image, model):
    """
    Resize a grayscale CT image to the model's expected
    spatial input dimensions.

    This assumes a single-channel 2D U-Net input.
    """
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        raise ValueError(
            "This app expects a model with a single input."
        )

    if len(input_shape) != 4:
        raise ValueError(
            f"Expected a 2D image model with 4 input dimensions; "
            f"received {input_shape}."
        )

    height = input_shape[1] or 256
    width = input_shape[2] or 256
    channels = input_shape[3]

    if channels not in (1, None):
        raise ValueError(
            f"This app expects a grayscale model, but the model "
            f"expects {channels} channels."
        )

    gray = np.array(image.convert("L"))

    resized = cv2.resize(
        gray,
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA
    )

    # Common normalization for grayscale images.
    normalized = resized.astype(np.float32) / 255.0

    # Shape: (1, height, width, 1)
    batch = normalized[np.newaxis, ..., np.newaxis]

    return batch, resized


# -----------------------------
# Segmentation prediction
# -----------------------------
def generate_prediction(image_array, model):
    prediction = model.predict(image_array, verbose=0)

    # Handle a single-image output.
    if isinstance(prediction, list):
        prediction = prediction[0]

    prediction = np.asarray(prediction)

    if prediction.ndim == 4:
        prediction = prediction[0]

    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[..., 0]

    # Binary segmentation: threshold probabilities.
    # Multiclass segmentation: choose the class with the
    # highest predicted probability.
    if prediction.ndim == 2:
        mask = (prediction >= 0.5).astype(np.uint8)
    elif prediction.ndim == 3:
        mask = np.argmax(prediction, axis=-1).astype(np.uint8)
    else:
        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    return prediction, mask


# -----------------------------
# Upload CT image
# -----------------------------
st.divider()
st.header("📤 Upload CT Image")

uploaded_file = st.file_uploader(
    "Choose a 2D CT image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original CT Image")
            st.image(image, use_container_width=True)

        with st.spinner("Generating segmentation..."):
            image_array, resized_image = preprocess_image(
                image, model
            )
            prediction, mask = generate_prediction(
                image_array, model
            )

        with col2:
            st.subheader("Predicted Segmentation")
            st.image(
                mask,
                caption="Model prediction mask",
                clamp=True,
                use_container_width=True
            )

        # Overlay the predicted mask on the resized CT image.
        display_mask = (
            mask > 0
        ).astype(np.uint8) * 255

        overlay = cv2.cvtColor(
            resized_image, cv2.COLOR_GRAY2RGB
        )

        overlay[display_mask > 0] = [255, 0, 0]

        st.subheader("CT Image with Segmentation Overlay")
        st.image(
            overlay,
            caption="Red indicates pixels classified as foreground",
            use_container_width=True
        )

        # Allow downloading the prediction.
        mask_image = Image.fromarray(display_mask)

        from io import BytesIO

        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")

        st.download_button(
            label="⬇️ Download Segmentation Mask",
            data=buffer.getvalue(),
            file_name="pancreas_segmentation.png",
            mime="image/png"
        )

    except Exception as error:
        st.error("Unable to process this image.")
        st.exception(error)

else:
    st.info("Upload a 2D CT image to start segmentation.")

st.divider()
st.caption(
    "Educational demonstration only. Predictions are not a "
    "medical diagnosis and must not guide treatment decisions."
)
