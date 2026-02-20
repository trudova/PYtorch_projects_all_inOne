# 🚀 Project1: Loan Classification Neural Network

End-to-end deep learning project that predicts loan approval status using a custom-built Neural Network in PyTorch.

## Overview
This project demonstrates a complete ML workflow:
- Data cleaning & preprocessing (Pandas, NumPy)
- Exploratory Data Analysis (Jupyter Notebook)
- Custom Neural Network architecture (PyTorch)
- Model training, evaluation & testing pipeline
- Performance metrics (accuracy, classification report)

## Tech Stack
Python · PyTorch · NumPy · Pandas · Scikit-learn · Matplotlib

## Highlights
- Modular project structure (separate model, device, and testing scripts)
- Structured training loop with loss tracking
- Reproducible experimentation setup
- Clean, production-style ML codebase

## Goal
Showcase practical deep learning skills including feature engineering, model development, debugging, and performance evaluation — aligned with real-world ML engineering workflows.

# 🖼️ Project 2: Concrit Image Classification CNN (PyTorch)

Deep learning project implementing a Convolutional Neural Network (CNN) for multi-class image classification using PyTorch.

## Overview
This project demonstrates a structured computer vision pipeline:
- Image preprocessing & transformations
- Custom CNN architecture implementation
- GPU/CPU device management
- Model training & evaluation loops
- Testing & inference workflow

## Tech Stack
Python · PyTorch · TorchVision · NumPy · Matplotlib

## Highlights
- Custom CNN model (`ImmageClassificationNet.py`)
- Separate training & testing scripts
- Modular device detection for GPU acceleration
- Reproducible training setup with performance tracking
- Clean, scalable project structure

## Goal
Showcase practical computer vision skills including CNN design, data transformation pipelines, model optimization, and production-style deep learning workflows.

# 🐶 Project 3: Dog Breed Classification CNN (PyTorch)

Multi-class deep learning project that classifies dog breeds from images using a custom Convolutional Neural Network (CNN) built in PyTorch.

## Overview
This project demonstrates a complete computer vision workflow:
- Image preprocessing & augmentation pipeline
- Multi-class CNN architecture implementation
- GPU/CPU device handling
- Structured training & validation loops
- Model evaluation & inference testing

## Tech Stack
Python · PyTorch · TorchVision · NumPy · Matplotlib · Scikit-learn

## Highlights
- Custom multi-class CNN (`ImageMultiClassCNN.py`)
- Modular training & testing scripts
- Data transformation pipeline for image normalization
- Device-aware training for GPU acceleration
- Clean, production-style deep learning structure

## Goal
Showcase applied computer vision and deep learning skills including CNN design, multi-class classification, model optimization, and scalable ML engineering practices.

# 🐾 Project 4: Dog Breed Classification — ResNet50 (Custom Training Pipeline)

Multi-class dog breed classification project using **transfer learning** with a pretrained **ResNet50** backbone, built with a clean, modular PyTorch training pipeline.

## Overview
This project fine-tunes ResNet50 for dog breed prediction and includes an end-to-end workflow:
- Dataset loading & preprocessing (`Data.py`)
- Model definition + custom classifier head (`Model.py`)
- GPU/CPU device detection (`Device.py`)
- Training loop with validation (`train.py`)
- Evaluation & inference testing (`test.py`)
- Image transforms / augmentation (`transformer.py`)

## Tech Stack
Python · PyTorch · TorchVision · ResNet50 · NumPy · Scikit-learn

## Highlights
- Pretrained ResNet50 + custom final layer for multi-class output
- Modular codebase (data, model, train, test separated)
- Device-aware training for GPU acceleration
- Reproducible experimentation structure

## Goal
Showcase real-world transfer learning, model fine-tuning, and production-style deep learning engineering practices.


# 🛞 Project 5: Tire Binary Classification — Transfer Learning (ResNet50)

Binary image classification project using **transfer learning** with a pretrained **ResNet50** model in PyTorch (e.g., tire defect vs. normal / worn vs. good).

## Overview
This project fine-tunes ResNet50 for a 2-class prediction task with a clean, modular pipeline:
- Dataset loading & preprocessing (`Data.py`)
- Model definition + custom binary head (`Model.py`)
- GPU/CPU device detection (`Device.py`)
- Training loop with validation (`train.py`)
- Evaluation & inference testing (`test.py`)
- Image transforms / augmentation (`transformer.py`)

## Tech Stack
Python · PyTorch · TorchVision · ResNet50 · NumPy · Scikit-learn

## Highlights
- Pretrained ResNet50 backbone + custom binary classifier head
- Modular code structure (data / model / train / test separated)
- Device-aware training for GPU acceleration
- Reproducible experiments and scalable CV workflow

## Goal
Showcase applied transfer learning and production-style deep learning engineering for real-world computer vision classification.

# ❤️ Project 6: Heartbeat Sound Classification — Audio→Image CNN (PyTorch)

Multi-class heartbeat classification project that converts raw heart sound recordings into **image-like representations** (e.g., spectrograms) and trains a **custom CNN** to predict cardiac sound categories.

## Overview
Pipeline includes:
- Audio preprocessing + feature extraction (sound → spectrogram images)
- Data loading & batching (`Data_loading.py`)
- Audio/image processing utilities (`data_processing.py`, `ploting_audio.py`, `transformer.py`)
- Custom CNN architecture (`SoundCNNModel.py`)
- Training & evaluation loops (`train.py`, `test.py`) with device support (`Device.py`)

## Highlights
- End-to-end “audio as vision” approach for classification
- Clean, modular codebase with reproducible training
- Strong performance demonstrated via confusion matrix (near-perfect separation across classes)

## Tech Stack
Python · PyTorch · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn 