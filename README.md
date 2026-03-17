# Mental Attention States Classification Using EEG Data

> **Course Project - Pattern Recognition and Machine Learning (PRML)**  
> University of Science · Faculty of Mathematics and Computer Science

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Pipeline](#project-pipeline)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Feature Engineering](#feature-engineering)
4. [Models](#models)
5. [Results](#results)
6. [Challenges & Future Work](#challenges--future-work)
7. [Tech Stack](#tech-stack)
8. [Team](#team)

---

## Overview

This project tackles the problem of automatically classifying human mental attention states from EEG (Electroencephalography) signals. The three states of interest are:

| Label | State | Description |
|-------|-------|-------------|
| 0 | **Focused** | Actively supervising a task with full concentration |
| 1 | **Unfocused** | Alert but no longer paying attention |
| 2 | **Drowsy** | Explicitly drowsing / losing consciousness |

Understanding these states in real time has direct applications in driver monitoring, cognitive workload assessment, and brain-computer interface (BCI) systems. EEG is a particularly attractive modality for this task due to its non-invasive nature and high temporal resolution.

---

## Dataset

**Source:** [EEG Data for Mental Attention State Detection](https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection) (Kaggle)

- **Participants:** 5 subjects, each controlling a simulated train along a featureless route for 35–55 minutes per session.
- **Experiments:** 7 sessions per participant; the first 2 were for habituation, the last 5 were used for data collection.
- **Files:** 24 MATLAB (`.mat`) files in total (one participant completed 4 out of 5 sessions).
- **EEG Channels:** 14 electrodes - `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`
- **Sampling Rate:** 128 Hz

**State Segmentation (per session):**
- Minutes 0-10 → Focused
- Minutes 10-20 → Unfocused
- Minutes 20-30 → Drowsy *(truncated to 10 min to balance classes)*

Only the first 30 minutes of each recording are used, yielding a balanced three-class distribution.

---

## Project Pipeline

### Data Preprocessing

Raw EEG signals go through a two-stage cleaning pipeline before any analysis:

**1. Bandpass Filtering (0.5-45 Hz)**  
A 4th-order Butterworth bandpass filter is applied to each channel. This range covers all clinically relevant brainwave bands while removing:
- Baseline drift (< 0.5 Hz)
- High-frequency muscle artifacts and electrical noise (> 45 Hz)

**2. Independent Component Analysis (ICA)**  
`FastICA` is used to decompose the multichannel EEG into statistically independent components. Artifact components are identified using two criteria:
- **Kurtosis** - components with abnormally high kurtosis (spiky, non-Gaussian behavior) are flagged.
- **Power** - components whose power exceeds the 95th percentile are flagged.

Identified artifact components are zeroed out before reconstructing the cleaned signal. This process is parallelised across all 24 recordings using `joblib`.

---

### Exploratory Data Analysis

Several visualizations were produced to build intuition about the data:

- **Raw vs. filtered EEG traces** - side-by-side channel plots illustrating the effect of the bandpass filter.
- **ICA component plots** - independent components colored by artifact status (pink background = potential artifact).
- **FFT + Welch dual-axis spectral plots** - frequency-domain representation of each channel with annotated brainwave bands (Delta, Theta, Alpha, Beta).
- **Per-state spectral comparison** - frequency analysis of channels `F3` and `F7` in each mental state, revealing:
  - **Focused:** dominant Beta activity (13–30 Hz) - active concentration.
  - **Unfocused:** dominant Alpha/Theta activity (4–13 Hz) - relaxed, inattentive.
  - **Drowsy:** dominant Theta/Delta activity (0.5–8 Hz) - transition toward sleep.

---

### Feature Engineering

Features are extracted using a **sliding window** approach on the ICA-cleaned signals:

| Parameter | Value |
|-----------|-------|
| Window (epoch) length | 128 samples (1 second) |
| Overlap | 64 samples (50%) |
| PSD method | Welch's method (`nperseg=128`) |

For each window and each of the 14 channels, **10 features** are computed:

| Feature | Description |
|---------|-------------|
| **Delta power** | Band power 1-4 Hz |
| **Theta power** | Band power 4-8 Hz |
| **Alpha power** | Band power 8-12 Hz |
| **Beta power** | Band power 13-30 Hz |
| **IWMF** | Intensity Weighted Mean Frequency |
| **IWBW** | Intensity Weighted Bandwidth |
| **SEF (95%)** | Spectral Edge Frequency at 95% cumulative power |
| **Variance** | Signal variance |
| **Energy** | Total signal energy |
| **Entropy** | Approximate spectral entropy |

This yields **140 features per window** (14 channels × 10 features). Feature extraction is also parallelised across all trials via `joblib`.

---

## Models

Four model families were evaluated on this dataset.

### 1. Support Vector Machine - RBF Kernel (GPU-accelerated)

Implemented with **RAPIDS cuML** for GPU-accelerated training on standardised features.

- **Kernel:** Radial Basis Function (RBF)
- **Train/Test split:** 80/20 (stratified)
- **Preprocessing:** `StandardScaler` (cuML)

**Decision function:**
$$f(x) = \sum_{i=1}^{N} \alpha_i y_i K(x_i, x) + b, \quad K(x_i, x) = e^{-\gamma \|x_i - x\|^2}$$

---

### 2. Support Vector Machine - Linear Kernel (GPU-accelerated)

Same setup as above, with a linear kernel:

$$f(x) = w \cdot x + b, \quad \hat{y} = \text{sign}(f(x))$$

---

### 3. 1D Convolutional Neural Network

A lightweight 1D CNN trained directly on the 140-dimensional feature vectors.

| Layer | Configuration |
|-------|---------------|
| Conv1d | 1 → 32 channels, kernel 3 |
| MaxPool1d | stride 2 |
| Linear | 32·F → 128 |
| Dropout | p = 0.5 |
| Linear | 128 → 3 |

- **Optimizer:** Adam (lr = 0.001)  
- **Loss:** Cross-Entropy  
- **Epochs:** 10, batch size 64

---

### 4. 2D CNN on FFT Spectrograms

A more expressive model that first transforms windowed EEG signals to the frequency domain via FFT, then classifies the resulting spectrograms.

**FFT magnitude:** $|X[k]| = \sqrt{\text{Re}(X[k])^2 + \text{Im}(X[k])^2}$

**Data preparation:**
- 1-second windows with 50% overlap → shape `[N, 128, 14]`
- FFT applied per channel → magnitude spectrum retained
- Reshaped to `[N, 14, 128, 1]` for 2D convolution

**Architecture (CNNModel):**

| Block | Layers |
|-------|--------|
| Conv Block 1 | Conv2d(14→32, 3×3) → ReLU → MaxPool(2×1) |
| Conv Block 2 | Conv2d(32→64, 3×3) → ReLU → MaxPool(2×1) |
| Conv Block 3 | Conv2d(64→128, 3×3) → ReLU → AdaptiveAvgPool(1×1) |
| FC Block | Flatten → Linear(128→64) → ReLU → Dropout(0.5) → Linear(64→3) |

- **Optimizer:** Adam (lr = 0.001)  
- **Loss:** Cross-Entropy  
- **Epochs:** 20, batch size 32

---

### 5. Random Forest

A classic ensemble baseline using the same tabular feature set to benchmark against the deep learning approaches.

- **Trees:** 100 estimators  
- **Train/Test split:** 70/30 (stratified)  
- **Aggregation:** Majority voting across trees

**Gini impurity (split criterion):**
$$G = 1 - \sum_{k=1}^{K} p_k^2$$

---

## Results

### Classification Performance

| Model | Accuracy | Macro F1 | AUC (Class 0) | AUC (Class 1) | AUC (Class 2) |
|-------|----------|----------|---------------|---------------|---------------|
| SVM — RBF | **97.19%** | **0.97** | 0.99 | 0.99 | 1.00 |
| 2D CNN (FFT) | 95.83%* | — | — | — | — |
| 1D CNN | 82.31% | 0.82 | 0.96 | 0.93 | 0.95 |
| SVM — Linear | 79.77% | 0.80 | 0.93 | 0.88 | 0.92 |

*Validation accuracy after 20 epochs (Training accuracy: 96.15%)

### Highlights

- The **RBF SVM** achieves the best overall performance, with near-perfect AUC scores across all three classes. Its strong results reflect the high separability of the engineered frequency-domain features in a kernel-induced space.
- The **2D CNN on FFT spectrograms** approaches RBF SVM performance while requiring no manual feature engineering, demonstrating the viability of end-to-end spectral learning.
- The **1D CNN** performs reasonably well considering its simplicity, though it lags behind the SVM in all metrics.
- The **Linear SVM** underperforms relative to the RBF kernel, suggesting the class boundaries in the feature space are not linearly separable.
- Across all models, **Class 1 (Unfocused)** is the hardest to classify, likely because it represents a transitional state between focused and drowsy rather than a physiologically distinct condition.

---

## Challenges & Future Work

### Challenges

- **Signal noise:** EEG is highly susceptible to artifacts from eye blinks, muscle activity, and electrode movement. Even after ICA, residual noise can affect model performance.
- **Subtle class boundaries:** The `Unfocused` state is physiologically close to both `Focused` and `Drowsy`, making precise classification inherently difficult.
- **High dimensionality:** The 140-feature space may contain redundant or irrelevant information, potentially slowing convergence and increasing overfitting risk.
- **Subject variability:** EEG patterns vary significantly across individuals; a model trained on data from all subjects may not generalize well to unseen participants.

### Potential Improvements

- **Advanced artifact removal:** Incorporate wavelet-based denoising or adaptive filtering alongside ICA.
- **Class imbalance handling:** Apply SMOTE or class-weighting strategies if extending to longer recording durations.
- **Feature selection:** Use Recursive Feature Elimination (RFE) or mutual information-based selection to reduce the feature space.
- **Subject-independent evaluation:** Implement leave-one-subject-out (LOSO) cross-validation to better assess cross-subject generalization.
- **Hyperparameter tuning:** Apply Bayesian optimization for systematic model tuning beyond grid search.
- **Transformer-based models:** Explore attention-based architectures (e.g., EEGNet, Conformer) that have shown strong results on EEG classification benchmarks.

---

## Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Language | Python 3 |
| Data I/O | `scipy.io`, `kagglehub`, `pandas` |
| Signal Processing | `scipy.signal`, `numpy`, `torch.fft` |
| Machine Learning | `scikit-learn`, `cuml` (RAPIDS AI) |
| Deep Learning | `PyTorch` |
| Visualization | `matplotlib`, `seaborn` |
| Parallelism | `joblib` |
| Environment | Google Colab (GPU runtime) |

---

## Team

| Name | Student ID | University |
|------|------------|------------|
| Ngô Thị Mỹ Duyên | 22280017 | University of Science, HCMC |
| Lê Hoàng Uyên Thư | 22280090 | University of Science, HCMC |
| Tô Gia Bảo | 22280006 | University of Science, HCMC |
| Bùi Trung Hiếu | 21110082 | University of Science, HCMC |

**Instructors:** Ngô Minh Mẫn · Lê Hoàng Đức
