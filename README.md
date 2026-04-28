# Human Activity Recognition (HAR) using Bi-Directional RNNs

This repository contains a deep learning project that classifies human physical activities based on smartphone sensor data. The project benchmarks two advanced architectures—**Bi-Directional GRU** and **Bi-Directional LSTM**—to capture temporal dependencies in movement patterns.

## 🚀 Highlights
* **High Accuracy:** Achieved a peak test accuracy of **92%** using the Bi-GRU model.
* **Comparative Analysis:** Evaluated both Bi-LSTM and Bi-GRU architectures to determine the most efficient model for time-series sensor data.
* **Bi-Directional Logic:** Leveraged bi-directional layers to distinguish between similar activities like "Walking Upstairs" and "Walking Downstairs" by analyzing the complete temporal context of each movement window.

## 📊 Dataset: UCI HAR
The project utilizes the **UCI Human Activity Recognition Dataset**, which contains recordings of 30 subjects performing daily activities with a waist-mounted smartphone.
* **Sensors:** Tri-axial Accelerometer and Gyroscope.
* **Input Data:** 3D Tensors of shape `(Samples, 128, 9)` representing 2.56-second windows of motion.
* **Classes:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.

## 🧠 Model Architecture
The project implements a stacked Bi-Directional approach built with TensorFlow/Keras:
1. **Input Layer:** `(128, 9)`
2. **Bi-RNN Layer 1:** 128 units + Dropout (0.5)
3. **Bi-RNN Layer 2:** 64 units + Dropout (0.5)
4. **Dense Layer:** 64 units (ReLU)
5. **Output Layer:** 6 units (Softmax)

## 📈 Evaluation & Model Comparison

I benchmarked two architectures to find the optimal balance between accuracy and generalization. While both models performed well, the **Bi-GRU** slightly outperformed the **Bi-LSTM** in overall test accuracy.

### 1. Bi-Directional GRU (Top Performer)
**Overall Accuracy: 92%**

| Activity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Walking** | 0.97 | 0.95 | 0.96 |
| **Walking Upstairs** | 0.90 | 0.94 | 0.92 |
| **Walking Downstairs**| 0.91 | 0.97 | 0.94 |
| **Sitting** | 0.89 | 0.78 | 0.83 |
| **Standing** | 0.85 | 0.91 | 0.88 |
| **Laying** | 0.99 | 0.97 | 0.98 |

### 2. Bi-Directional LSTM
**Overall Accuracy: 91%**

| Activity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Walking** | 0.98 | 0.93 | 0.96 |
| **Walking Upstairs** | 0.88 | 0.96 | 0.92 |
| **Walking Downstairs**| 0.91 | 0.95 | 0.93 |
| **Sitting** | 0.87 | 0.75 | 0.81 |
| **Standing** | 0.83 | 0.89 | 0.86 |
| **Laying** | 0.99 | 0.97 | 0.98 |

---

### 📉 Model Performance Analysis

**Why Bi-GRU?**
The **Bi-GRU** was selected as the final model because it achieved **92% accuracy** with a simpler architecture. It exhibited a smaller gap between training and validation accuracy, suggesting it is more robust against sensor noise compared to the Bi-LSTM.

**Key Observations:**
* **Dynamic Success:** Both models excelled at identifying **Laying** (98% F1) and **Walking** (96% F1).
* **Static Challenge:** Both models faced challenges distinguishing **Sitting** from **Standing** due to nearly identical gravity vectors. This highlights a path for future improvement using magnitude-based feature engineering.
