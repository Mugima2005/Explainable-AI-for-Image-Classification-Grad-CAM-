## Explainable AI for Image Classification using Grad-CAM
# 📌 Project Overview

This project implements a CNN-based image classification system for crop disease detection and enhances it with Explainable AI using Grad-CAM. The model classifies leaf images into Healthy and Blight categories and visualizes the image regions that influence its predictions.

# 🎯 Objectives

Build an image classification model using a Convolutional Neural Network (ResNet18)

Apply Grad-CAM to explain model predictions

Improve transparency and trust in deep learning models

Validate that predictions are based on disease-affected regions

# 🧠 Model & Techniques

Model: ResNet18 (Pre-trained)

Framework: PyTorch

Explainability: Gradient-weighted Class Activation Mapping (Grad-CAM)

Hardware Acceleration: CUDA-enabled GPU

# 📂 Dataset Structure
dataset/
 ├── train/
 │    ├── Healthy/
 │    ├── Blight/
 ├── test/
 │    ├── Healthy/
 │    ├── Blight/


Folder names represent class labels

Images are resized to 224×224 for model input

# ⚙️ Workflow

Load and preprocess image dataset

Train CNN model on training data

Evaluate model performance on test data

Apply Grad-CAM to visualize important image regions

Save Grad-CAM heatmaps for analysis

# 📊 Output

Predicted class label (Healthy / Blight)

Grad-CAM heatmap highlighting influential regions

Saved explanation images for reporting and visualization

# 🧪 Results

The model successfully classifies crop leaf images

Grad-CAM highlights disease-affected regions, confirming correct model learning

Explainability improves confidence in model predictions

# 🚀 How to Run
# Train the model
python load_model.py

# Generate Grad-CAM explanation
python gradcam.py

# 🛠 Technologies Used

Python

PyTorch

Torchvision

OpenCV

NumPy

CUDA

# 📌 Key Learnings

CNN-based image classification

Model explainability using Grad-CAM

GPU-accelerated deep learning

Importance of interpretable AI in real-world applications

# 👤 Author

Mugima S
B.Tech Artificial Intelligence and Data Science
