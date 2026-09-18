# EEG-Based Fibromyalgia Classification Pipeline

**A Methodological Replication and Extension of Li et al. (2026)**
*Paper DOI:* [10.3389/fpain.2025.1704444](https://doi.org/10.3389/fpain.2025.1704444)

### Datasets

* **Primary Dataset (SRPBG):** [https://osf.io/srpbg/files/osfstorage](https://osf.io/srpbg/files/osfstorage) *(Note: Due to hardware/epoch limitations in external datasets like TDBrain, cross-domain validation is strictly performed via intra-dataset cross-cohort splitting on the SRPBG dataset, leveraging its multi-study composition of FM, CBP, and NCCP cohorts).*

## Overview

This repository contains an advanced machine learning pipeline designed to identify neurophysiological biomarkers for Fibromyalgia (FM) using resting-state EEG data. The methodology replicates the feature extraction and classification framework proposed by Li et al. (2026), utilizing spectral connectivity (Coherence) and an optimized Support Vector Machine (SVM).

To comprehensively validate these findings, the repository extends the original methodology by introducing a parallel **Geometry-Based Riemannian Pipeline** (evaluating Spatial Covariance Matrices via Tangent Space mapping), along with robust modules for  **Algorithmic Fairness (Bias)** ,  **Transfer Learning (TrAdaBoost)** , and  **Post-Hoc Interpretability (SHAP & MNE Topography)** .

To maintain high academic standards and prevent data leakage, the pipeline is controlled by a centralized `config.py` and structured into isolated, modular stages.

---

## Execution Order & Pipeline Structure
To ensure full reproducibility and prevent data leakage, the repository is strictly modular. The pipeline requires specific manual interventions where hyperparameters and target bands must be updated based on intermediate results.

### CRITICAL WORKFLOW: When to update settings

1. **Initial Setup:** Before running anything, configure your dataset paths, sample rates (`SFREQ_MAP`), and standard 10-20 channels in `config.py`.
2. **First Run (Exploration):** Run Phase 1, followed by `1_SVM_feature_ranking.py` and `2_riemann_whole_brain.py`. These scripts evaluate the *entire* spatial and spectral space to find the optimal frequency bands.
3. **Manual Intervention (Update Config):** Stop the pipeline. Review the outputs of the exploratory scripts and update `config.py`:
   * Set `FOCUS_BAND = 'your_best_band'` (e.g., `'gamma'`) for the downstream SVM pipeline.
   * Set `BEST_BANDS = ['band_1', 'band_2']` (e.g., `['Theta', 'Gamma']`) for the downstream Riemannian ablation and SVM Fusion scripts.
4. **Resume Pipeline:** Run the remaining scripts in Phase 2, 3, and 4. 
*(Note: Some Riemannian scripts, like `3_riemann_roi_ablation.py`, contain an internal boolean toggle `RUN_AS_WHOLE_BRAIN = False/True` that must be set manually depending on the desired spatial layout).*

## Pipeline Architecture

### Phase 0: Central Configuration
* **`config.py`**
  The central nervous system of the repository. Manages global variables, channel selections, label mappings, and the dynamic `FOCUS_BAND` / `BEST_BANDS` routing.

### Phase 1: Preprocessing & Data Aggregation
* **`preprocess_pipeline.py`:** Filters raw `.vhdr` EEG data, epochs into 1-second segments, and extracts spectral coherence. Generates QC plots via **`prep_plot.py`**.
* **`build_dataset.py`:** Aggregates features and strictly enforces an 80/20 subject-level Train/Test split.

### Phase 2: Feature-Based Baseline Pipeline (SVM)
* **`1_SVM_feature_ranking.py`:** Evaluates the full feature space to calculate global SHAP values and identify the optimal `FOCUS_BAND`.
* **`2_SVM_roi_feature_screening.py`:** Restricts data to the central 9-channel ROI and identifies the Top 10 candidate connectivity features.
* **`3_SVM_feature_selection_msffs.py`:** Runs the mSFFS algorithm to find the optimal minimal feature subset.
* **`4_SVM_final_model_training.py`:** Trains and tunes the final SVM on the selected subset. Freezes the model as a `.pkl` artifact.
* **`5_SVM_model_evaluation.py`:** Evaluates the frozen model on the unseen test set, generating clinical metrics, final SHAP plots, and t-SNE distributions.

### Phase 3: Geometry-Based Riemannian Pipeline
* **`1_preprocess_riemann.py`:** Generates Spatial Covariance matrices strictly respecting the Phase 1 train/test split.
* **`2_riemann_whole_brain.py`:** Exploratory analysis testing all architectures on the 19-channel layout to identify the `BEST_BANDS`.
* **`3_riemann_roi_ablation_singlecv_rep10.py`:** Evaluates the identified top bands using the central ROI and freezes the winning geometric models.
* **`4_plot_riemann_results.py`:** Trains linear surrogate models to extract and plot paper-ready Riemannian network topographies.
* **`5_Riemann_Model_Evaluation.py`:** Evaluates the frozen models on the test set using Subject-Level Majority Voting.

### Phase 4: Fairness & Cross-Domain Generalizability
* **`6_Riemann_Bias_Evaluation.py` & `7_SVM_female_sensitivity_analysis.py`:** Evaluates model robustness against demographic confounders.
* **`7_Riemann_cross_domain_Tradaboost.py` & `8_SVM_cross_domain_validation.py`:** Performs Transfer Learning (TrAdaBoost / Riemannian Alignment) on external cohorts (e.g., NCCP).
* **`9_SVM_cross_frequency_fusion_msffs.py`:** Tests for complementarity between the two `BEST_BANDS` using a combined mSFFS search space.

## Methodological Safeguards

* **Zero Data Leakage:** The `StandardScaler` (SVM) and Fréchet Mean reference states (Riemann) are fitted solely on the training data. The 20% hold-out sets are transformed exclusively using these frozen parameters.
* **Blind Demographics:** Demographic data (Age, Sex) are strictly excluded from the training feature space to prevent the algorithms from establishing non-physiological diagnostic shortcuts. Bias is evaluated strictly post-hoc.

## 🧪 Alternative: LOSOCV Evaluation Pipeline (Folder: `/LOSOCV`)

To overcome the statistical limitations of small clinical hold-out sets and prevent optimization leakage, this repository includes an alternative pipeline utilizing **Leave-One-Subject-Out Cross-Validation (LOSOCV)**. This pipeline replaces the standard 80/20 Train/Test split evaluation.

> **⚠️ CRITICAL WARNING: DESTRUCTIVE OVERWRITE**
> Executing the scripts within the `LOSOCV/` directory **will overwrite** the intermediate feature files (e.g., `top_10_roi_features.csv`), scoreboards, and frozen `.pkl` model artifacts generated by the main pipeline. 
> 
> *Only run these scripts if you have backed up/moved the results from the main baseline pipeline, or if you explicitly intend to overwrite them to establish the LOSOCV models as your definitive evaluation standard.*

### LOSOCV Execution Order

If you choose to run the LOSOCV validation, execute the scripts in the `LOSOCV/` folder in the following order:

1. **`build_dataset.py`:** Generates a unified `final_dataset_master.csv` (bypassing the traditional train/test split) while isolating the target domain.
2. **`1_SVM_feature_ranking.py` & `2_SVM_roi_feature_screening.py`:** Performs exploratory SHAP ranking on the full master dataset to define the feature space.
3. **`3_SVM_feature_selection_msffs.py`:** Runs the mSFFS algorithm evaluated strictly via LOSOCV to find the optimal minimal feature subset.
4. **`4_SVM_final_model_training.py`:** The core LOSOCV evaluation engine. Computes final clinical metrics (AUROC, Brier, ECE, Permutation P-values) using strict subject isolation.
5. **Riemannian Scripts (`1_...` to `7_...`):** The geometric pipeline is similarly adapted to evaluate spatial covariance matrices utilizing Fréchet mean alignment and LOSOCV across the master cohort.
