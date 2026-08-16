# Image Classification using Deep Learning

## Overview

This repository contains implementations of **image classification using different deep learning approaches**.

My goal is to understand how Convolutional Neural Networks (CNNs) learn visual representations, compare CNN implementations using **TensorFlow/Keras** and **PyTorch**, and study **transfer learning using a pretrained ResNet18 model**.

Current implementations:

- **Custom CNN using TensorFlow/Keras** 
- **Custom CNN using PyTorch** 
- **Transfer Learning using ResNet18 (PyTorch)** 

All implementations use the same six-class scene classification task so that the models and training workflows can be compared.

---

## Image Classification

Image classification is a computer vision task in which a model assigns an input image to one predefined class based on its visual content.

The dataset used in this repository contains six scene categories:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

Unlike object detection, which also identifies the location of objects, image classification predicts a class for the image as a whole.

---

## Convolutional Neural Networks

A **Convolutional Neural Network (CNN)** is a deep learning architecture designed to learn spatial patterns from image data.

Instead of manually defining image features, CNNs learn filters directly during training. Earlier convolutional layers generally capture low-level patterns such as edges and textures, while deeper layers combine these representations into more complex visual features.

Main components:

- **Convolutional Layers** — learn local visual patterns using trainable filters
- **ReLU Activation** — introduces non-linearity
- **Max Pooling** — reduces the spatial dimensions of feature maps
- **Flattening** — converts feature maps into a feature vector
- **Dense / Linear Layers** — perform classification
- **Dropout** — helps reduce overfitting
- **Softmax / Logits** — represent output scores used for class prediction

<img width="1536" height="1024" alt="go6MKktRG9keRRDfhY3UyMbSUJeseINb1spcwxz-xrjWbbuto1q1p0lP4_mEX839zXh-ZKQkiAQXVSfqWLRGIeSE13GG12IC0DZc1BKRmSYArThgMdTPILP9O95bbgN7odEmFjm6VLDFGSBQE3NTO8e1KXFayuz1tulHEnNaMbuZoodFCjA68svc8pyClUVZ" src="https://github.com/user-attachments/assets/021f9e01-d93f-4ed9-8f1b-2aba9a17757f" />


---

# Approach 1: CNN from Scratch

Training a CNN from scratch means that the network starts with randomly initialized parameters and learns both visual features and the final classification function directly from the target dataset.

The same CNN-based image classification task is implemented using both **TensorFlow/Keras** and **PyTorch**.

## TensorFlow / Keras CNN

**TensorFlow** is a deep learning framework, while **Keras** provides a higher-level API for defining and training neural networks.

The implementation includes:

- Image resizing and normalization
- Data augmentation
- Convolution and pooling layers
- Dense layers and dropout
- Adam optimizer
- Sparse Categorical Cross-Entropy
- Training and validation
- Early Stopping
- Test evaluation
- Confusion matrix
- Image prediction
- Model saving

Keras handles much of the training process through:

```python
model.fit(...)
```

Internally, neural-network training still performs:

- Forward propagation
- Loss computation
- Backpropagation
- Gradient computation
- Parameter updates

**Early Stopping** monitors validation loss and stops training when additional epochs no longer improve validation performance.

<img width="1856" height="676" alt="1*7K4ZTTfZb-hbjoADbisHAg" src="https://github.com/user-attachments/assets/039c88c5-4830-4110-b553-7e01b946490e" />


---

## PyTorch CNN

**PyTorch** provides a more explicit neural-network training workflow.

The network is defined by extending `nn.Module`, while the `forward()` method specifies how input tensors propagate through the network.

The implementation uses:

- `ImageFolder`
- `Dataset` and `DataLoader`
- `torchvision.transforms`
- `Conv2d`
- `MaxPool2d`
- `Linear`
- `Dropout`
- `CrossEntropyLoss`
- Adam optimizer

Unlike the higher-level Keras training interface, PyTorch explicitly performs the major optimization steps:

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Here:

- `optimizer.zero_grad()` clears gradients from the previous iteration
- `model(images)` performs the forward pass
- `criterion(outputs, labels)` calculates the loss
- `loss.backward()` performs backpropagation and computes gradients
- `optimizer.step()` updates the model parameters

This makes the mechanics of neural-network training more visible.

<img width="1400" height="749" alt="1*A8cX1eHqGI7dIuH3awcjlw" src="https://github.com/user-attachments/assets/1e5ac389-8246-4f16-b97c-d3d452e1a6ff" />


---

## TensorFlow/Keras vs PyTorch

TensorFlow/Keras and PyTorch are **deep learning frameworks**, not different CNN architectures.

The underlying CNN concepts remain the same. The main difference is how each framework exposes the model-building and training workflow.

| TensorFlow / Keras | PyTorch |
|---|---|
| Higher-level training API | More explicit training workflow |
| `model.fit()` manages training | Training loop is written manually |
| Backpropagation is largely abstracted | `loss.backward()` explicitly computes gradients |
| Parameter updates are handled internally | `optimizer.step()` explicitly updates parameters |

---

# Approach 2: Transfer Learning

**Transfer learning** reuses representations learned by a model that has already been trained on a large source dataset.

Instead of initializing every model parameter randomly, a pretrained network provides useful visual representations that can be transferred to a new image-classification task.

CNNs trained on large image datasets learn general-purpose features such as edges, textures, shapes, and object-level patterns. These representations can often be reused instead of being learned again from scratch.

In this repository, transfer learning is implemented using **ResNet18**.

<img width="1334" height="751" alt="iwnOvgoKUVAwUIuIbMRVv3t8bLb6P3GGRtg3QqWpvR2rW-nlQvfVFb5MyV4U9IK3OvXX0LzBibkKV-8UO1HJM_EdYuSgqieIdpkINvS2jLxX00F8ooh0ds5CWZJ0DYZjkO8NpRMDSuz-co8vE-DEUP5W3UEsMyyhouwpGfVHH7_HubIhQIEzBMmOud_4vOr_" src="https://github.com/user-attachments/assets/89693aa7-7349-4ce8-8a43-68f01e50d008" />

---

## ResNet18

Residual Networks, or **ResNets**, were introduced by He et al. in *Deep Residual Learning for Image Recognition*.

As neural networks become deeper, simply stacking additional layers can make optimization difficult. ResNet addresses this using **residual or skip connections**.

Instead of requiring a group of layers to directly learn a mapping `H(x)`, a residual block learns:

`F(x) = H(x) - x`

The desired mapping can therefore be represented as:

`H(x) = F(x) + x`

The original input `x` is passed through a skip connection and added to the output of the convolutional layers.

These residual connections improve information and gradient flow through deep networks and make deeper architectures easier to optimize.

<img width="1123" height="487" alt="ResNet" src="https://github.com/user-attachments/assets/6971ba2d-fb13-4c93-85fb-abdccd7e3a1c" />

### Transfer Learning Setup

For the current implementation:

1. Load a pretrained **ResNet18**
2. Freeze the pretrained convolutional parameters
3. Replace the original final classification layer
4. Create a new fully connected layer for the six target classes
5. Train the new classification layer

ResNet18 therefore acts as a **pretrained feature extractor**, while the new final layer learns the six-class scene classification task.

---

## CNN from Scratch vs Transfer Learning

| CNN from Scratch | Transfer Learning |
|---|---|
| Starts with randomly initialized weights | Starts with pretrained weights |
| Learns visual representations from the target dataset | Reuses previously learned representations |
| Entire network is trained | Pretrained backbone can be frozen |
| Usually requires more optimization | Can converge with fewer training epochs |
| Useful for studying CNN fundamentals | Useful for adapting pretrained models to new tasks |

---

# Results

## TensorFlow / Keras CNN

- **Best Validation Accuracy:** 82.32%
- **Test Accuracy:** ~82.8%
- **Training:** Configured for 20 epochs; stopped after 6 epochs using Early Stopping

## PyTorch CNN

- **Training Accuracy:** 90.86%
- **Test Accuracy:** 86.10%
- **Test Loss:** 0.4746
- **Training:** 20 epochs

## ResNet18 Transfer Learning

- **Training Accuracy:** 89.35%
- **Training Loss:** 0.2937
- **Test Accuracy:** 89.60%
- **Test Loss:** 0.2817
- **Training:** 10 epochs

---

## Model Comparison

| Approach | Training Accuracy | Test Accuracy | Test Loss |
|---|---:|---:|---:|
| TensorFlow/Keras Custom CNN | — | ~82.8% | — |
| PyTorch Custom CNN | 90.86% | 86.10% | 0.4746 |
| ResNet18 Transfer Learning | 89.35% | **89.60%** | **0.2817** |

Among the current experiments, **ResNet18 Transfer Learning achieved the highest test accuracy of 89.60%**.

---

## Dependencies

- Python
- TensorFlow
- Keras
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

TensorFlow and PyTorch can be run in separate virtual environments to keep framework-specific dependencies isolated.

---

## Repository Purpose

The purpose of this repository is to **study and implement image-classification and deep-learning concepts rather than use neural-network models only as black boxes**.

Topics covered include:

- Image Classification
- Convolutional Neural Networks
- Convolutional Feature Extraction
- TensorFlow/Keras
- PyTorch
- Forward Propagation
- Backpropagation
- Gradient-Based Optimization
- Regularization
- Model Evaluation
- Transfer Learning
- Residual Networks
- ResNet18
- Pretrained Feature Extraction

Additional image-classification architectures and experiments may be added as the study progresses.
