# Image Classification using Deep Learning

## Overview

This repository contains implementations of image classification using different deep learning approaches.

Current approaches:

- **Custom CNN using TensorFlow/Keras** 
- **Custom CNN using PyTorch** 
- **Transfer Learning using a Pretrained CNN** 

The current TensorFlow/Keras model is trained for up to **20 epochs** with a **batch size of 32** and uses **Early Stopping** to reduce overfitting.

## What is Image Classification?

Image classification is the task of assigning an image to one of a set of predefined classes based on its visual features.

The dataset contains six classes:

`buildings` • `forest` • `glacier` • `mountain` • `sea` • `street`

## Approaches

### 1. Custom CNN

A **Convolutional Neural Network (CNN)** learns visual features directly from images using convolution and pooling layers, followed by fully connected layers for classification.

### 2. PyTorch CNN

The same image classification task will be implemented using **PyTorch** to understand model definition, training, backpropagation, and optimization.

### 3. Transfer Learning

Transfer learning uses a **pretrained CNN**, such as ResNet, and adapts it to a new classification task instead of learning all visual features from scratch.

## Features

- Image loading and preprocessing
- Data augmentation
- Custom CNN architecture
- Training and validation
- Early Stopping
- Accuracy and loss tracking
- Confusion matrix
- Image prediction
- Saving the trained model

## Results

### TensorFlow/Keras CNN

- **Best Validation Accuracy:** 82.32%
- **Test Accuracy:** ~82.8%
- **Training:** Configured for 20 epochs; stopped after 6 epochs using Early Stopping

### PyTorch CNN

- **Training Accuracy:** 90.86%
- **Test Accuracy:** 86.10%
- **Test Loss:** 0.4746
- **Training:** Completed for 20 epochs

### Transfer Learning

`Planned`

## Technologies

`Python` • `TensorFlow` • `Keras` • `PyTorch` • `NumPy` • `Matplotlib` • `Scikit-learn`
