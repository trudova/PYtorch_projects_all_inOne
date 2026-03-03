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

# 🐱🐶 Project 7: Cat vs Dog Classification — DenseNet121 + Custom Head (PyTorch)

Binary image classification project using **transfer learning** with a pretrained **DenseNet121** backbone and a custom fully connected classifier head in PyTorch.

## Overview
This project fine-tunes DenseNet121 for a 2-class image classification task (Cat vs Dog) using a structured, modular training pipeline:

- Dataset loading via `ImageFolder` (`Data.py`)
- Image preprocessing & augmentation pipeline (`transformer.py`)
- Custom classifier head with BatchNorm + Dropout (`Model.py`)
- Device-aware training (CUDA / MPS / CPU support) (`Device.py`)
- BCEWithLogitsLoss for stable binary training
- Training & validation loops with performance tracking (`train.py`)
- Evaluation with accuracy, F1 score & confusion matrix (`test.py`)

## Tech Stack
Python · PyTorch · TorchVision · DenseNet121 · NumPy · Scikit-learn · Matplotlib · Seaborn

## Highlights
- Pretrained **DenseNet121 (ImageNet weights)** with frozen backbone
- Custom fully connected classifier (progressive dimensional reduction + regularization)
- Binary classification using **BCEWithLogitsLoss**
- Automatic device detection (GPU / Apple MPS / CPU fallback)
- Model checkpointing based on accuracy + loss thresholds
- Confusion matrix visualization with Seaborn

## Architecture
- DenseNet121 feature extractor (frozen)
- Custom FC head:
  - Linear → BatchNorm → ReLU → Dropout
  - Progressive dimensional reduction
  - Final single-neuron output for sigmoid-based binary prediction

## Goal
Demonstrate practical transfer learning, modular ML engineering practices, and production-style deep learning workflow design for real-world computer vision tasks.

# 👁️ Project 8: Corneal Infection Classification — ResNet152 + Custom FC Head (PyTorch)

Multi-class medical image classification project using **transfer learning** with a pretrained **ResNet152** backbone and a custom fully connected classifier head to predict corneal infection types.

## Overview
This project fine-tunes ResNet152 for 4-class corneal epithelium pathology classification:

- Dataset loading via `ImageFolder` (`get_data.py`)
- Medical image preprocessing & normalization (`transformer.py`)
- Pretrained ResNet152 feature extractor (ImageNet weights)
- Custom fully connected classifier head (`fc_resnet_model.py`)
- Device-aware training (CUDA / MPS / CPU fallback) (`Device.py`)
- Structured training & validation pipeline (`train_m_rest.py`)
- Evaluation with accuracy, macro F1-score & confusion matrix (`test_rest.py`)

## Classes
- No_ulcer_of_the_corneal_epithelium  
- Micro_punctate  
- Macro_punctate  
- Coalescent_macro_punctate  

## Tech Stack
Python · PyTorch · TorchVision · ResNet152 · NumPy · Scikit-learn · Matplotlib · Seaborn

## Highlights
- Pretrained **ResNet152 (ImageNet weights)** with frozen backbone
- Custom deep fully connected head with:
  - Progressive dimensional reduction (2048 → 64)
  - Batch Normalization
  - Dropout regularization
- Multi-class classification using **CrossEntropyLoss**
- Separate feature extractor + classifier training (transfer learning best practice)
- Automatic device detection (GPU / Apple MPS / CPU)
- Confusion matrix visualization for clinical interpretability
- Model checkpointing based on loss + accuracy thresholds

## Architecture
- ResNet152 feature extractor (final FC removed via `Identity`)
- Custom FC head:
  - Linear → BatchNorm → ReLU → Dropout blocks
  - Final 4-neuron output layer for multi-class prediction

## Goal
Demonstrate applied deep learning in medical computer vision, structured transfer learning workflows, and production-style ML engineering practices for real-world healthcare image classification tasks.

# 👁️ Project 9: Corneal Infection Classification — DenseNet121 + Custom Head (PyTorch)

Multi-class medical image classification project using **transfer learning** with a pretrained **DenseNet121** backbone and a custom fully connected classifier head to predict corneal infection severity types.

## Overview
This project fine-tunes DenseNet121 for 5-class corneal epithelium pathology classification using a clean, modular deep learning pipeline:

- Dataset loading via `ImageFolder` (`Data.py`)
- Image preprocessing & augmentation pipeline (`transformer.py`)
- Pretrained DenseNet121 feature extractor (ImageNet weights)
- Custom fully connected classifier head (`Model.py`)
- Device-aware training (CUDA / MPS / CPU fallback) (`Device.py`)
- Structured training & validation loops (`train_densNet.py`)
- Evaluation with accuracy, macro F1-score & confusion matrix visualization (`test_densnet.py`)

## Classes
- No_ulcer_of_the_corneal_epithelium  
- Micro_punctate  
- Macro_punctate  
- Coalescent_macro_punctate  
- Patch_ge_1mm  

## Tech Stack
Python · PyTorch · TorchVision · DenseNet121 · NumPy · Scikit-learn · Matplotlib · Seaborn

## Highlights
- Pretrained **DenseNet121 (ImageNet weights)** with frozen backbone
- Custom deep classifier head:
  - Progressive dimensional reduction (1024 → 12 → 5)
  - Batch Normalization for training stability
  - Dropout regularization to reduce overfitting
- Multi-class classification using **CrossEntropyLoss**
- Data augmentation (rotation, inversion, horizontal flip)
- Automatic device detection (GPU / Apple MPS / CPU support)
- Model checkpointing based on accuracy and loss thresholds
- Confusion matrix heatmap for interpretability in medical context

## Architecture
- DenseNet121 feature extractor (classifier replaced)
- Custom FC head:
  - Linear → BatchNorm → ReLU → Dropout blocks
  - Final 5-neuron output layer for multi-class prediction

## Goal
Demonstrate applied deep learning in medical computer vision, structured transfer learning workflows, and scalable ML engineering practices for healthcare image classification problems.

# 👁️ Project 10: Corneal Infection Classification — Custom CNN (Perfect Classification)

Multi-class medical image classification project using a fully **custom-built Convolutional Neural Network (CNN)** architecture that achieved **perfect classification performance** on the evaluation set.

## Overview
This project implements a custom end-to-end CNN (no transfer learning) for 5-class corneal infection severity classification:

- Dataset loading via `ImageFolder` (`Data_loading.py`)
- Image preprocessing & augmentation (`transformer.py`)
- Fully custom CNN architecture (`CNNModel.py`)
- Structured training pipeline with 100 epochs (`train.py`)
- Evaluation using accuracy, macro F1-score & confusion matrix (`test.py`)
- Device-aware training (CUDA / MPS / CPU fallback) (`Device.py`)

## Classes
- No_ulcer_of_the_corneal_epithelium  
- Micro_punctate  
- Macro_punctate  
- Coalescent_macro_punctate  
- Patch_ge_1mm  

## Tech Stack
Python · PyTorch · NumPy · Scikit-learn · Matplotlib · Seaborn

## Highlights
- Fully custom CNN architecture (no pretrained backbone)
- 3 convolutional blocks with MaxPooling
- Deep fully connected classifier (256 → 128 → 64 → 5)
- CrossEntropyLoss for multi-class learning
- Achieved **100% classification accuracy**
- Perfect confusion matrix (no misclassifications)
- Stable convergence with near-zero training & testing loss
- Model checkpointing based on strict loss + accuracy thresholds

## Architecture
**Feature Extractor**
- Conv2D (3 → 6) → ReLU → MaxPool  
- Conv2D (6 → 16) → ReLU → MaxPool  
- Conv2D (16 → 32) → ReLU → MaxPool  

**Classifier**
- Flatten  
- Linear → ReLU (256)  
- Linear → ReLU (128)  
- Linear → ReLU (64)  
- Linear → 5-class output  
- LogSoftmax activation  

## Training Setup
- 100 epochs
- Adam optimizer
- Learning rate: 0.0009
- CrossEntropyLoss
- Real-time tracking of:
  - Training accuracy
  - Testing accuracy
  - Training loss
  - Testing loss

## Results
- 100% accuracy on evaluation set
- Macro F1-score = 1.00
- Perfect diagonal confusion matrix
- Near-zero loss convergence
- 
<p align="center">
  <img src="acc.png" width="45%"/>
  <img src="loss.png" width="45%"/>
</p>

<p align="center">
  <img src="CM_eye.png" width="60%"/>
</p>

## Goal
Demonstrate the ability to design, train, and optimize a fully custom CNN architecture for medical image classification — achieving complete class separation without relying on transfer learning.


