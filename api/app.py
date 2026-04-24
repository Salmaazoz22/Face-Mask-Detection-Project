"""
Face Mask Detection API
-----------------------
FastAPI backend that exposes a /predict endpoint for classifying
whether a person in an uploaded image is wearing a face mask or not.

Model: MobileNetV2 trained on 224x224 RGB images

Classes:
    0 -> "mask_on"
    1 -> "without_mask"

Run with:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "mask_detector.pth"

CLASS_NAMES = ["mask_on", "without_mask"]
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """
    Rebuild MobileNetV2 with the same classifier used during training.
    """

    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES),
    )

    return model


def load_model() -> nn.Module:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = build_model()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("✅ Loaded model_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully")

    return model


# Load model once
model: nn.Module = load_model()

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

])


def preprocess_image(image_bytes: bytes):

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:
        raise ValueError(f"Cannot read image: {exc}")

    tensor = TRANSFORM(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    return tensor


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Face Mask Detection API",
    version="2.0.0"
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():

    return {
        "status": "ok",
        "message": "API is running successfully"
    }


# ---------------------------------------------------------------------------
# Model Info Endpoint ⭐ (Professional addition)
# ---------------------------------------------------------------------------

@app.get("/info", tags=["Info"])
def model_info():

    return {

        "model": "MobileNetV2",

        "image_size": 224,

        "classes": CLASS_NAMES,

        "device": str(DEVICE)

    }


# ---------------------------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------------------------

@app.post("/predict", tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Upload image file")
):

    # Validate file type
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp"
    ):
        raise HTTPException(
            status_code=415,
            detail="Upload a valid image file."
        )

    image_bytes = await file.read()

    try:
        input_tensor = preprocess_image(image_bytes)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

    with torch.no_grad():

        logits = model(input_tensor)

    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    class_name = CLASS_NAMES[predicted_idx.item()]

    confidence_score = round(
        confidence.item(),
        4
    )

    # ⭐ Add Action Logic (Required in project)

    if class_name == "mask_on":
        action = "Allow entry"
    else:
        action = "Deny entry"

    # Logging (optional but useful)

    print(
        f"Prediction: {class_name} | "
        f"Confidence: {confidence_score}"
    )

    return JSONResponse(

        content={

            "status": class_name,

            "action": action,

            "confidence": confidence_score

        }

    )