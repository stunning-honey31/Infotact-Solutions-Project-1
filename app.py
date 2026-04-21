import streamlit as st
import torch
import os
import sys
import json
from PIL import Image
import torchvision.transforms as transforms

# Fix path
sys.path.append(os.path.abspath("."))

from models.model import CustomCNN

def format_class_name(name):
    # Replace ___ with separator
    name = name.replace("___", " - ")

    # Replace underscores with spaces
    name = name.replace("_", " ")

    # Capitalize nicely
    name = name.title()

    return name


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")

# Load class names
class_names = []
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

checkpoint = None
state_dict = None
checkpoint_num_classes = len(class_names) if class_names else 0

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

if "fc.3.weight" in state_dict:
    checkpoint_num_classes = int(state_dict["fc.3.weight"].shape[0])

if checkpoint_num_classes <= 0:
    st.error("Could not infer number of classes from model checkpoint.")
    st.stop()

# Align labels with checkpoint output size.
if len(class_names) != checkpoint_num_classes:
    if os.path.isdir(TRAIN_DIR):
        train_classes = sorted([
            name for name in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, name))
        ])
        if len(train_classes) == checkpoint_num_classes:
            class_names = train_classes
            st.warning("classes.json count mismatched model outputs. Using classes discovered from data/train.")
        else:
            class_names = [f"Class_{i}" for i in range(checkpoint_num_classes)]
            st.warning("classes.json count mismatched model outputs. Using generic class names.")
    else:
        class_names = [f"Class_{i}" for i in range(checkpoint_num_classes)]
        st.warning("classes.json count mismatched model outputs. Using generic class names.")

# Load model with checkpoint class count
model = CustomCNN(num_classes=checkpoint_num_classes)

try:
    model.load_state_dict(state_dict)
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model checkpoint: {e}")
    st.stop()

# 🎯 UI
st.title(" Crop Disease Detection System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)

    # Prediction with spinner
    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probabilities, 1)

    # Main prediction
    class_name = class_names[predicted.item()].replace("___", " - ")
    class_name = format_class_name(class_names[predicted.item()])
    confidence_score = confidence.item()

    st.markdown(f"### Prediction: **{class_name}**")
    st.markdown(f"### Confidence: **{confidence_score * 100:.2f}%**")

    # Confidence bar
    st.progress(float(confidence_score))

    # Top-3 Predictions
    st.markdown("### Top 3 Predictions")
    top_k = min(3, probabilities.shape[1], len(class_names))
    probs, indices = torch.topk(probabilities, top_k)

    for i in range(top_k):
        name = class_names[indices[0][i]].replace("___", " - ")
        score = probs[0][i].item() * 100
        st.write(f"{i+1}. {name} — {score:.2f}%")