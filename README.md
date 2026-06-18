
# EEG-Based Fibromyalgia Classification Pipeline

**A Methodological Replication and Extension of Li et al. (2026)** *Paper DOI:* [10.3389/fpain.2025.1704444](https://doi.org/10.3389/fpain.2025.1704444)

### Datasets

* **Primary Dataset (SRPBG):** [https://osf.io/srpbg/files/osfstorage](https://osf.io/srpbg/files/osfstorage)*(Note: Due to hardware/epoch limitations in external datasets like TDBrain, cross-domain validation is strictly performed via intra-dataset cross-cohort splitting on the SRPBG dataset, leveraging its multi-study composition of FM, CBP, and NCCP cohorts).*
* **Secondary References:** TDBrain (DOI: 10.1038/s41597-022-01409-z) | GyK89 ([https://osf.io/m45j2](https://osf.io/m45j2))

---

## Overview

This repository contains an advanced machine learning pipeline designed to identify neurophysiological biomarkers for Fibromyalgia (FM) using resting-state EEG data. The methodology replicates the feature extraction and classification framework proposed by Li et al. (2026), utilizing spectral connectivity (Coherence) and an optimized Support Vector Machine (SVM).

To comprehensively validate these findings, the repository extends the original methodology by introducing a parallel **Geometry-Based Riemannian Pipeline** (evaluating Spatial Covariance Matrices via Tangent Space mapping), along with robust modules for **Algorithmic Fairness (Bias)**, **Transfer Learning (TrAdaBoost)**, and **Post-Hoc Interpretability (SHAP & MNE Topography)**.

To maintain high academic standards and prevent data leakage, the pipeline is controlled by a centralized `config.py` and structured into isolated, modular stages.

---

## Pipeline Architecture

### Phase 0: Configuration (`config.py`)

All global variables, random states, channel selections (10-20 system), and the central ablation switch (`USE_ROI: True/False`) are managed here to guarantee architectural synchronization across all scripts.

### Phase 1: Feature-Based Baseline Pipeline (SVM)

* **1. Data Preprocessing & Feature Extraction (`preprocess_pipeline.py`):** Applies a 50 Hz notch filter and 0.5–44 Hz bandpass filter. Discards the first 10s of data. Divides data into 30s macro-segments and 1s micro-epochs. Extracts 855 Spectral Coherence features across 5 frequency bands using the multitaper method.
* **2. Dataset Aggregation & Balancing (`build_dataset.py`):**
  Aggregates features and implements a strict 80/20 train/test split grouped by `Subject` to prevent data leakage. Dynamically balances the training set via data-density calculation and locks exactly 5 segments per subject for the hold-out test set.
* **3. Feature Selection & Model Training (`train_svm.py`):**
  Executes exclusively on the 80% training set. Uses mSFFS (Modified Sequential Floating Forward Selection) to isolate the optimal connectivity features. Optimizes RBF-SVM parameters via Grid Search and validates significance via a 1000-shuffle permutation test. *(Replicates Paper Figure 3).*
* **4. External Validation (`test_unseen_data.py`):**
  Predicts diagnostic classes on the unseen 20% hold-out set using frozen artifacts. Computes AUPRC, AUROC, Precision, Recall, Brier Score, and Expected Calibration Error (ECE) to assess diagnostic utility and clinical reliability.
* **5. Post-Hoc Interpretability (`shap_analysis.py`):**
  Generates SHAP values to quantify biomarker contributions. Exports the Mean Absolute SHAP Bar Chart *(Figure 6A)*, the Beeswarm Summary *(Figure 6B)*, and maps the top 5 functional connections onto an MNE-Python topographical head map *(Figure 4)*.

### Phase 2: Geometry-Based Riemannian Pipeline

* **1. Strict Preprocessing (`1_preprocess_riemann.py`):**
  Inherits the exact 80/20 subject split from the SVM pipeline. Processes raw arrays into Spatial Covariance Matrices (SCMs) for both Whole-Brain (19 channels) and Central ROI (9 channels) layouts across all 5 bands.
* **2. Tangent Space Classification (`2_train_riemann.py`):**
  Projects SCMs onto a flat tangent space originating at the Fréchet mean. Trains both TS-SVM and MDM (Minimum Distance to Mean) classifiers via Stratified Group K-Fold CV.
* **3. Riemann Validation & Topography (`4_evaluate_testset.py` & `3_plot_topo.py`):**
  Evaluates the frozen geometry models on the hold-out set, generating equivalent clinical metrics (AUPRC, ECE). Reconstructs Tangent Space weights back into anatomical electrode pairs to plot geometric network connections.
* **4. Feature Ranking (`5_riemann_feature_ranking.py`):**
  Extracts and ranks global feature weights from the Riemannian pipeline to generate a mathematical comparison against the SVM mSFFS selection.

### Phase 3: Clinical Fairness & Generalizability

* **Algorithmic Bias Evaluation (`evaluate_bias.py`):**
  Merges blind test-set predictions from both the SVM and Riemannian models with metadata (`participants.tsv`) to evaluate hardware-specific and demographic fairness (e.g., accuracy across Sex and Age groups).
* **Cross-Domain Validation (`cross_domain_validation.py`):**
  Tests hardware-independence by evaluating frozen models on isolated target cohorts (Zero-Shot Direct Testing). Subsequently applies **TrAdaBoost** (Instance-Weighted Domain Adaptation) to calibrate models against hardware-specific distribution shifts. *(Replicates Paper Figure 7).*

---

## Methodological Safeguards

* **Zero Data Leakage:** The `StandardScaler` (SVM) and Fréchet Mean reference states (Riemann) are fitted solely on the training data. The 20% hold-out sets are transformed exclusively using these frozen parameters.
* **Blind Demographics:** Demographic data (Age, Sex) are strictly excluded from the training feature space to prevent the algorithms from establishing non-physiological diagnostic shortcuts. Bias is evaluated strictly post-hoc.
* **Artifact Prevention:** By epoching the data prior to concatenation, the pipeline actively prevents phase-discontinuity artifacts that could otherwise lead to severe overfitting in lower frequency bands.
