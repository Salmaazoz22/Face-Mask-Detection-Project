# Face Mask Detection

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-nginx--alpine-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.60%25-brightgreen)

An end-to-end deep learning system for real-time face mask detection. Built with **MobileNetV2 Transfer Learning**, served through a **FastAPI** REST backend, and paired with a **static HTML/CSS/JS** frontend that supports both live webcam streaming and image upload.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation Results](#evaluation-results)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Running the Frontend](#running-the-frontend)
- [API Reference](#api-reference)
- [Project Workflow](#project-workflow)
- [Team Members](#team-members)

---

## Project Overview

This project addresses a real-world computer vision problem: automatically detecting whether a person is wearing a face mask. The system is designed to be lightweight, accurate, and deployable — making it suitable for access control systems, surveillance pipelines, or edge deployments.

The pipeline spans the full ML lifecycle:

1. Data collection and preparation
2. Exploratory data analysis and augmentation
3. Transfer learning with MobileNetV2
4. REST API with FastAPI
5. Interactive frontend with webcam and upload support
6. Containerized frontend deployment via Docker

---

## Features

- **Real-time webcam detection** — captures frames every second and classifies them via the API
- **Image upload / drag-and-drop** — supports JPEG, PNG, WEBP, BMP
- **Confidence score** — visual progress bar and percentage shown per prediction
- **Action decision** — API returns `Allow entry` or `Deny entry` based on prediction
- **Dockerized frontend** — nginx-based static server for easy deployment
- **Interactive API docs** — auto-generated Swagger UI available at `/docs`
- **GPU/CPU auto-selection** — model runs on CUDA if available, falls back to CPU

---

## Project Structure

```
Face-Mask-Detection-Project/
│
├── api/
│   └── app.py                    # FastAPI application (all endpoints and model logic)
│
├── frontend/
│   ├── index.html                # Single-page UI (webcam + upload)
│   ├── style.css                 # Dark-themed responsive stylesheet
│   ├── script.js                 # Webcam capture, upload, API communication
│   ├── Dockerfile                # nginx:alpine static server for frontend
│   └── .dockerignore
│
├── insights/
│   ├── classification_report.txt # Per-class precision, recall, F1
│   ├── evaluation_summary.txt    # Full evaluation summary with error analysis
│   ├── summary of results.txt    # Dataset counts and resize statistics
│   └── test_accuracy.txt         # Final test accuracy (99.60%)
│
├── Models/
│   └── mask_detector.pth         # Trained model weights (not tracked in git)
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb # Dataset loading, split verification, balance check
│   ├── 02_EDA.ipynb              # Exploratory data analysis and visualizations
│   ├── 03_Augmentation.ipynb     # Augmentation strategies and visual samples
│   ├── 04_Model_Training.ipynb   # MobileNetV2 fine-tuning, training loop, checkpointing
│   └── 05_evaluation_report.ipynb# Confusion matrix, misclassification analysis
│
├── .gitignore
├── requirements.txt              # Python dependencies
└── README.md
```

> **Note:** The `Dataset/` and `Dataset_Resized/` directories are excluded from version control via `.gitignore`. See the [Dataset](#dataset) section for the expected folder layout.

---

## Dataset

The model was trained on approximately **11,792 face images** organized into two classes: `WithMask` and `WithoutMask`.

### Split Summary

| Split      | Images | Ratio   |
|:-----------|-------:|--------:|
| Train      | 10,000 | 84.80%  |
| Validation |    800 |  6.78%  |
| Test       |    992 |  8.41%  |
| **Total**  | **11,792** | **100%** |

### Expected Folder Layout

```
Dataset/
├── Train/
│   ├── WithMask/
│   └── WithoutMask/
├── Validation/
│   ├── WithMask/
│   └── WithoutMask/
└── Test/
    ├── WithMask/
    └── WithoutMask/
```

### Preprocessing

- Original images varied in size (34×34 to 139×139 pixels)
- All images resized to **224 × 224** pixels
- All images converted to RGB; zero corrupt or non-RGB images found
- Normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

### Training Augmentations

Applied to the training split to improve generalization:

- `RandomHorizontalFlip`
- `RandomRotation(15°)`
- `ColorJitter` (brightness, contrast, saturation)

---

## Model Architecture

The model uses **MobileNetV2** (pretrained on ImageNet) as a frozen feature extractor, with a custom classifier head trained on the mask dataset.

### Classifier Head

```
MobileNetV2 (frozen backbone)
    └── features → Global Average Pooling → [1280]
            ↓
        Dropout(p=0.2)
        Linear(1280 → 256)
        ReLU()
        Dropout(p=0.2)
        Linear(256 → 2)
```

### Training Configuration

| Parameter       | Value                              |
|:----------------|:-----------------------------------|
| Optimizer       | Adam                               |
| Learning Rate   | 1e-3                               |
| Weight Decay    | 1e-4                               |
| LR Scheduler    | ReduceLROnPlateau (factor=0.5, patience=2) |
| Loss Function   | CrossEntropyLoss                   |
| Max Epochs      | 15                                 |
| Early Stopping  | Patience = 4 (on validation loss)  |
| Input Size      | 224 × 224                          |
| Batch Loader    | `torchvision.datasets.ImageFolder` |

### Class Mapping

| Index | Folder Name   | API Label       | Decision     |
|:-----:|:--------------|:----------------|:-------------|
| 0     | `WithMask`    | `mask_on`       | Allow entry  |
| 1     | `WithoutMask` | `without_mask`  | Deny entry   |

---

## Evaluation Results

The final model was evaluated on **992 held-out test images**.

### Summary

| Metric          | Value    |
|:----------------|:---------|
| Test Accuracy   | **99.60%** |
| Misclassified   | 4 images |

### Classification Report

| Class          | Precision | Recall | F1-Score | Support |
|:---------------|:---------:|:------:|:--------:|--------:|
| WithMask       | 0.99      | 1.00   | 1.00     | 483     |
| WithoutMask    | 1.00      | 0.99   | 1.00     | 509     |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **992** |

### Error Analysis

The 4 misclassified images were attributed to:

- Low lighting conditions
- Side-facing or partially visible faces
- Incorrect/improper mask usage (mask pulled down or covering only chin)

> Detailed confusion matrices and per-epoch loss/accuracy curves are available in `notebooks/05_evaluation_report.ipynb` and the `insights/` folder.

---

## Tech Stack

| Layer         | Technology                          |
|:--------------|:------------------------------------|
| Deep Learning | PyTorch 2.0+, torchvision 0.15+     |
| Model         | MobileNetV2 (Transfer Learning)     |
| Image I/O     | Pillow 10.0+                        |
| API Backend   | FastAPI 0.110+, Uvicorn 0.29+       |
| File Uploads  | python-multipart 0.0.9+             |
| Frontend      | HTML5, CSS3, Vanilla JavaScript     |
| Containerization | Docker (nginx:alpine)            |
| Notebooks     | Jupyter (data prep, EDA, training, eval) |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) CUDA-compatible GPU

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Face-Mask-Detection-Project.git
cd Face-Mask-Detection-Project

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Place the trained model weights
#    Copy mask_detector.pth into the Models/ folder:
#    Face-Mask-Detection-Project/Models/mask_detector.pth
```

---

## Running the API

Start the FastAPI server from the **project root** directory:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

| URL | Description |
|:----|:------------|
| `http://localhost:8000/` | Health check |
| `http://localhost:8000/info` | Model information |
| `http://localhost:8000/predict` | Prediction endpoint |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/redoc` | ReDoc documentation |

---

## Running the Frontend

### Option 1 — VS Code Live Server (recommended for development)

1. Open the `frontend/` folder in VS Code.
2. Right-click `index.html` → **Open with Live Server**.
3. The page opens at `http://127.0.0.1:5500` (matches the API's CORS configuration).
4. Ensure the FastAPI backend is running at `http://localhost:8000`.

### Option 2 — Docker (nginx static server)

```bash
cd frontend

# Build the image
docker build -t mask-frontend .

# Run on port 80
docker run -p 80:80 mask-frontend
```

The frontend will be served at `http://localhost:80`.

> **CORS note:** The backend is configured to accept requests from `http://127.0.0.1:5500` by default. If you serve the frontend from a different origin (e.g., Docker on port 80), update `allow_origins` in `api/app.py` accordingly.

---

## API Reference

### `GET /`

Health check endpoint.

**Response**

```json
{
  "status": "ok",
  "message": "API is running successfully"
}
```

---

### `GET /info`

Returns metadata about the loaded model.

**Response**

```json
{
  "model": "MobileNetV2",
  "image_size": 224,
  "classes": ["mask_on", "without_mask"],
  "device": "cpu"
}
```

---

### `POST /predict`

Classifies an uploaded image.

**Request**

| Type | Field | Value |
|:-----|:------|:------|
| Content-Type | `multipart/form-data` | — |
| Form field | `file` | Image file (JPEG, PNG, BMP, WEBP) |

**Success Response — `200 OK`**

```json
{
  "status": "mask_on",
  "action": "Allow entry",
  "confidence": 0.9987
}
```

| Field | Type | Description |
|:------|:-----|:------------|
| `status` | `string` | `mask_on` or `without_mask` |
| `action` | `string` | `Allow entry` or `Deny entry` |
| `confidence` | `float` | Model confidence score (0.0 – 1.0, 4 decimal places) |

**Error Responses**

| Status | Condition |
|:-------|:----------|
| `415 Unsupported Media Type` | File is not a supported image format |
| `422 Unprocessable Entity` | Image cannot be decoded or read |

---

## Project Workflow

```
Raw Dataset
    │
    ▼
01_Data_Preparation.ipynb
    │  • Verify class balance
    │  • Check image integrity
    │  • Split into Train / Validation / Test
    ▼
02_EDA.ipynb
    │  • Visualize class distribution
    │  • Sample image inspection
    ▼
03_Augmentation.ipynb
    │  • Apply and visualize augmentation strategies
    │  • Prepare Dataset_Resized for training
    ▼
04_Model_Training.ipynb
    │  • Load pretrained MobileNetV2
    │  • Freeze backbone, train custom classifier head
    │  • Early stopping, LR scheduling
    │  • Save mask_detector.pth
    ▼
05_evaluation_report.ipynb
    │  • Load saved checkpoint
    │  • Compute accuracy, precision, recall, F1
    │  • Confusion matrix and misclassification analysis
    │  • Export results to insights/
    ▼
api/app.py  ←──────────────────────── Models/mask_detector.pth
    │  • Load model on startup
    │  • Expose POST /predict endpoint
    ▼
frontend/
    │  • Webcam stream → frame capture → POST /predict
    │  • Image upload / drag-and-drop → POST /predict
    │  • Display status, action, and confidence
    ▼
Docker (frontend/Dockerfile)
       • Serve static frontend via nginx:alpine
```

---

## Team Members

| Name | Role |
|:-----|:-----|
| Sameh Maged Ahmed | Data Manager |
| Malak Tarek Ahmed | EDA & Visualizer |
| Ahmed Khaled Sayed | Augmentation Designer |
| Rawan Essam El-Din | Model Trainer |
| Mohamed Saied Ahmed | Evaluator |
| Salma Mohamed Abdelaziz | Team Leader, API Developer & Frontend Developer |
| Shahd Mohamed Abdelhay | Deployer |
