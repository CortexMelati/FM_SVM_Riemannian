
# EEG-Based Fibromyalgia Classification Pipeline

**A Methodological Replication of Li et al. (2026) # https://doi.org/10.3389/fpain.2025.1704444**

Dataset 1: https://osf.io/srpbg/files/osfstorage

Dataset 2: TDBrain DOI = 10.1038/s41597-022-01409-z

Dataset 3: https://osf.io/m45j2

## Overview

This repository contains a machine learning pipeline designed to identify neurophysiological biomarkers for Fibromyalgia (FM) using resting-state EEG data. The methodology replicates the feature extraction and classification framework proposed by Li et al. (2026), utilizing spectral connectivity (Coherence) and an optimized Support Vector Machine (SVM) classifier.

To maintain high academic standards and prevent data leakage, the pipeline is structured into four isolated, modular stages.

## Pipeline Architecture

### 1. Data Preprocessing & Feature Extraction (`preprocess_pipeline.py`)

This script processes the raw EEG signals and extracts the foundational connectivity features.

* **Signal Cleaning:** Applies a 50 Hz notch filter and a 0.5–44 Hz bandpass filter. The initial 10 seconds of data are discarded to eliminate recording artifacts.
* **Segmentation:** The continuous EEG data is divided into 30-second macro-segments.
* **Micro-Epoching:** Each 30-second segment is further subdivided into 1-second micro-epochs. This captures phase-locked connectivity without introducing temporal artifacts.
* **Feature Engineering:** Calculates Spectral Connectivity (Coherence) using the multitaper method across 171 channel pairs over 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma), yielding 855 features per segment.

### 2. Dataset Aggregation & Balancing (`build_dataset.py`)

This script aggregates individual subject features into a master dataset and structures the train/test splits.

* **Subject-Level Stratification:** Implements an 80/20 train/test split grouped strictly by `Subject` to prevent identity memorization and cross-segment data leakage.
* **Data Balancing (Training):** Addresses class imbalance by applying a defined sampling ratio: 5 segments per FM patient and 4 segments per Healthy Control (HC).
* **Hold-Out Validation (Testing):** Enforces an inclusion criterion where only subjects with exactly 5 available segments are admitted to the hold-out test set, ensuring a structurally sound evaluation matrix.

### 3. Feature Selection & Model Training (`train_svm.py`)

This script executes the machine learning logic exclusively on the 80% training set.

* **ROI Isolation:** Restricts the feature space to the central 9-channel Region of Interest (ROI) or skip this by changing USE_ROI = False
* **Cross-Validation:** Utilizes `StratifiedGroupKFold` (grouped by subject) to ensure reliable internal validation.
* **mSFFS:** Employs Modified Sequential Floating Forward Selection to identify the optimal feature subset (ranging from 1 to 20 features).
* **Hyperparameter Tuning:** Optimizes the SVM (RBF kernel) parameters (**$C$** and **$\gamma$**) using `GridSearchCV`.
* **Statistical Validation:** Computes a 1000-shuffle non-parametric permutation test to verify the statistical significance of the internal accuracy.
* **Artifact Generation:** Exports the trained model, the selected features, and the `StandardScaler` as frozen `.pkl` files.

### 4. External Validation (`test_unseen_data.py`)

This script serves as the unbiased evaluation layer.

* **Strict Isolation:** Loads the isolated 20% hold-out test set and applies the frozen `.pkl` artifacts (model and scaler) generated during the training phase.
* **Evaluation:** Predicts the diagnostic class (HC vs. FM) on unseen data and calculates generalized performance metrics (Accuracy, ROC-AUC, Precision, Recall).
* **Visualization:** Generates Confusion Matrices for each frequency band to assess the predictive validity of the identified biomarkers.

## Methodological Safeguards

* **Zero Data Leakage:** The `StandardScaler` is fitted solely on the training data. The hold-out set is transformed exclusively using these frozen parameters.
* **Artifact Prevention:** By epoching the data prior to concatenation, the pipeline actively prevents phase-discontinuity artifacts that could otherwise lead to severe overfitting in lower frequency bands.
