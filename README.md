# 😷 Face Mask Detection Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

An end-to-end deep learning project for detecting face mask usage in images using **Transfer Learning**, deployed via **FastAPI** and **Docker**.

---

## 📌 Project Overview

The goal of this project is to build a robust computer vision system that predicts whether a person is wearing a face mask with high confidence.

The system is designed to be:
- ⚡ Fast and lightweight  
- 🎯 Highly accurate  
- 🚀 Easily deployable in real-world scenarios  

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (MobileNetV2, ResNet18)
* **Data Science:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Backend:** FastAPI, Uvicorn
* **DevOps:** Docker

---

## 📂 Dataset & Preprocessing
The model was trained using the **Face Mask Dataset** (~12K images) from Kaggle.

### Data Split
| Split | Percentage | Purpose |
| :--- | :--- | :--- |
| **Train** | 70% | Model parameter optimization |
| **Validation** | 15% | Hyperparameter tuning & overfitting check |
| **Test** | 15% | Final unbiased evaluation |

### Preprocessing Pipeline
1.  **Resize:** All images scaled to $224 \times 224$ pixels.
2.  **Normalization:** Applied ImageNet mean and standard deviation.
3.  **Augmentation:** Applied Horizontal Flips, Rotations, and Brightness Adjustments to improve generalization. *(Note: Vertical flips were excluded as faces are naturally upright).*

---

## 🏗️ Model Architecture & Training
We utilized **Transfer Learning** to leverage features from models pre-trained on ImageNet.

-   **The Model:** **MobileNetV2** (Selected for its excellent speed/accuracy tradeoff on edge devices).

### Strategy
- Freeze the backbone feature extractor.
- Replace the final fully connected layer with a custom classifier head.
- Fine-tune selected top layers to adapt to facial features.
- Implement **Early Stopping** to prevent overfitting.

---

## 📊 Evaluation & Insights
The model achieved high performance, particularly on clear frontal faces.

### Metrics
- **Accuracy:** ~96%
- **Error Analysis:** Most misclassifications occurred in scenarios with extreme low lighting or heavy occlusion (e.g., hands over face).

> [!TIP]
> Check `notebooks/evaluation.ipynb` for detailed confusion matrices and loss/accuracy curves.

---

## 🚀 Deployment

### ⚡ FastAPI Implementation
The application exposes a REST API for real-time predictions.

**Run Locally:**
```bash
uvicorn api.app:app --reload
```

----

## 🐳 Docker

Containerize the application for seamless deployment:

```Bash
docker build -t mask-app .
docker run -p 8000:8000 mask-app
```

---

## 📁 Project Structure

```text
/

Face-Mask-Detection-Project/

├── insights/             # Reports and insights from the training and evaluation
├── notebooks/            # Data Preparation, EDA, Training, and Evaluation Jupyters
├── model/                # Saved .pth weights
├── api/                  # FastAPI application
├── requirements.txt      # Project dependencies
└── README.md             
```

---

## 🔮 Future Improvements

- [ ] Implement real-time webcam detection using OpenCV.

- [ ] Expand dataset to include more "MaskIncorrect" samples.

- [ ] Deploy to Cloud (AWS App Runner or Google Cloud Run).

- [ ] Build a React-based frontend dashboard.

---

## ⭐ Support
### If you find this project helpful for your AI/ML journey, please consider giving it a Star!
