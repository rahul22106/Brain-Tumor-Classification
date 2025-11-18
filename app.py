import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Update path to your new model folder!
MODEL_PATH = "model/model.keras"
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

@st.cache_resource
def load_classifier():
    return load_model(MODEL_PATH)

model = load_classifier()

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))  # Adjust if your model uses another size
    arr = np.array(img) / 255.0   # Normalize pixel values
    arr = arr[np.newaxis, ...]    # Add batch dimension
    return arr

st.title("Brain Tumor Classification (MRI Image)")
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    arr = preprocess(image)
    pred_probs = model.predict(arr)[0]
    pred_idx = int(np.argmax(pred_probs))
    pred_label = CLASS_LABELS[pred_idx]
    confidence = float(pred_probs[pred_idx])
    st.markdown(f"### Prediction: `{pred_label}`")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
    st.bar_chart(dict(zip(CLASS_LABELS, pred_probs)))