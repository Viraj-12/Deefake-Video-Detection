# 🎭 AI-Powered Deepfake Video Detection

> **A robust deep learning system to detect manipulated video content.**

![Project Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)

## 📖 Overview

**AI-Powered Deepfake Video Detection** is a deep learning–based system designed to analyze videos, extract frames, and predict whether the content is **REAL** or **MANIPULATED**.

Leveraging a custom-trained model (CNN architecture), this project focuses on high accuracy, robustness against compression, and fast inference times. It is capable of processing video files, extracting key frames, and aggregating predictions to provide a final confidence score.

---

## ✨ Core Features

* **🎬 Smart Frame Extraction:** Automatically extracts key frames at fixed intervals for reliable prediction, reducing redundancy.
* **🧠 Deepfake Classification Model:** Advanced architecture (CNN) trained on diverse real vs. fake datasets.
* **📊 Confidence Scores:** Provides a clear probability percentage indicating if the video is FAKE or REAL.
* **🖼️ Visual Frame Output:** Displays the specific frames analyzed during the inference process.
* **⚡ Optimized Pipeline:** Engineered for speed with fast preprocessing and efficient embedding extraction.
* **🔍 Explainability (XAI):** Includes an optional module for simple text-based explanations of the model's decision.

---

## 🚀 Tech Stack & Tools

### Machine Learning & Backend
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

### Utilities & Processing
![XAI](https://img.shields.io/badge/XAI-Explainability-000000?style=for-the-badge&logoColor=white)

---

## 🧠 How It Works

The system follows a streamlined pipeline to process video data:

1.  **Input:** User uploads a video file (MP4, AVI, etc.) via the API or Interface.
2.  **Extraction:** The system uses `MTCNN` to extract frames at specific intervals.
3.  **Preprocessing:** Faces are detected and cropped; frames are normalized for the model.
4.  **Inference:** The preprocessed frames are passed to the PyTorch model.
5.  **Aggregation:** The model outputs probabilities for individual frames, which are aggregated (averaged/weighted) to form a final verdict.
6.  **Output:** The system returns a **REAL** or **FAKE** label along with a confidence score and (optional) XAI explanation.

---

