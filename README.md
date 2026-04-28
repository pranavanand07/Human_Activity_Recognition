# Human Activity Recognition (HAR) using Bi-Directional GRUs

This repository contains a deep learning project that classifies human physical activities based on smartphone sensor data. By utilizing a **Bi-Directional Gated Recurrent Unit (Bi-GRU)** architecture, the model captures temporal dependencies in both forward and backward directions, significantly improving the recognition of complex motion patterns.

## 🚀 Highlights
* **High Accuracy:** Achieved a test accuracy of **92%** on the UCI HAR dataset.
* **Bi-Directional Logic:** Leveraged bi-directional layers to distinguish between similar activities like "Walking Upstairs" and "Walking Downstairs" by analyzing the complete temporal context of each step.
* **Robust Performance:** Outperformed standard Bi-LSTM models in generalization, maintaining a smaller gap between training and validation accuracy.

## 📊 Dataset: UCI HAR
The project uses the **UCI Human Activity Recognition Dataset**. It includes recordings of 30 subjects performing daily activities while wearing a waist-mounted smartphone.
* **Sensors:** Tri-axial Accelerometer and Gyroscope.
* **Input Data:** 3D Tensors of shape `(Samples, 128, 9)` representing 2.56-second windows of motion.
* **Classes:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.

## 🧠 Model Architecture
The final model is a stacked Bi-Directional GRU built with TensorFlow/Keras:
1. **Input Layer:** `(128, 9)`
2. **Bi-GRU Layer 1:** 128 units + Dropout (0.5)
3. **Bi-GRU Layer 2:** 64 units + Dropout (0.5)
4. **Dense Layer:** 64 units (ReLU)
5. **Output Layer:** 6 units (Softmax)

## 📈 Evaluation
### Classification Report
| Activity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Walking** | 0.97 | 0.95 | 0.96 |
| **Walking Upstairs** | 0.90 | 0.94 | 0.92 |
| **Walking Downstairs**| 0.91 | 0.97 | 0.94 |
| **Sitting** | 0.89 | 0.78 | 0.83 |
| **Standing** | 0.85 | 0.91 | 0.88 |
| **Laying** | 0.99 | 0.97 | 0.98 |

### 📈 Model Performance Analysis
The Bi-GRU model achieved a strong balance between training and validation accuracy. 

**Key Observation:** The model demonstrates high precision in dynamic activities (Walking, Stairs) but shows a slight bottleneck in distinguishing **Sitting** from **Standing**. This is a known challenge in HAR datasets where the static gravity vector is nearly identical for both activities. Future iterations could include a "Magnitude" feature calculation to improve static state separation.
