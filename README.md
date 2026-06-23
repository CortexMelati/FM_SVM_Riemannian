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

Dus we hebben eerst een plot met het gehele hoofd meest belangrijke coherence features (Daar selecteren we de beste band van, welke het meest voorkomt)

dan selecteren we de top 10 coherence features binnen die belangrijke band (die het meest voorkomt) en in de ROI. (figuur voor maken met lijntjes, nog toevoegen). Dan creëren ze een script waarin ze trainen met die 10 features + 10 andere die ze er nog bij konden maken binnen de ROI en kijken welke settings het beste zijn.

Vervolgens gebruiken ze die settings, met de top 5/6/7/8 features (vanaf wanneer het model niet meer significante verbeteringen maakt) om het echte SVM model te trainen. In die tussenstap maken ze nog een topografie mapje met de geselecteerde top 5/6/7/8 features. En vanuit het echte SVM model krijgen we dan dus het Tsne figuur, Fig 6 A+B

* **`1_SVM_global_feature_ranking.py`**
  * **Input:** De volledige dataset met alle 855 features (alle kanalen, alle banden).
  * **Process:** Er wordt een snelle baseline SVM getraind om globale SHAP-waarden te berekenen over de gehele opzet.
  * **Output:** **Figuur 1** (Horizontale SHAP-balken van de top 15 globale features), waarmee mathematisch wordt aangetoond dat de uiteindelijke-band en de centrale kanalen de hoogste voorspellende waarde hebben. Dit rechtvaardigt de stap naar de ROI.
* **`2_SVM_roi_feature_screening.py`**
  * **Input:** Data gefilterd op de gekozen beste-band en de 9 centrale ROI-kanalen (36 mogelijke connectie-features).
  * **Process:** Een SHAP-analyse op deze sub-selectie om de top 10 connecties binnen de ROI te identificeren.
  * **Output:** Een netwerk-topografiemap met exact 10 lijnen die deze initiële screening visueel in kaart brengt.
* **`3_`SVM_`feature_selection_msffs.py`**
  * **Input:** De gescreende ROI-data uit de vorige stap.
  * **Process:** Het mSFFS-algoritme bouwt incrementele subsets op van 1 tot 20 features en berekent via *StratifiedGroupKFold* de cross-validation en training accuracies.
  * **Output:** **Figuur 3** (De mSFFS-curve met de blauwe en oranje lijnen). Aan de hand van deze curve stel je vast bij welk aantal features (in de paper 5) de accuraatheid afvlakt en de standaarddeviatie het laagst is.
* **`4_`SVM_`final_model_training.py`**
  * **Input:** De trainingsdata (80%) gereduceerd tot *strictly* de top-5(of meer) features geselecteerd uit de mSFFS-curve (Fz-Cz, Pz-P4, Fz-C3, Cz-P4, Cz-Pz).
  * **Process:** De definitieve SVM (RBF) wordt getraind via Grid Search om de optimale hyper-parameters (**$C$** en **$\gamma$**) te vinden voor exact deze selected features. De statistische significantie wordt getoetst met de 1000-shuffle permutatietest. Het model en de scaler worden hierna bevroren opgeslagen als `.pkl`.
  * **Output:** **Figuur 4** (De definitieve topografische hersenkaart met exact 5 rode lijnen + dikte naar importance die de geselecteerde biomarkers tonen).
* **`5_`SVM_`model_evaluation_and_shap.py`**
  * **Input:** De hold-out testset (20%) gefilterd naar de top selected features, en het bevroren model-artifact.
  * **Process:** Het onaangetaste model doet blinde voorspellingen op de testset. De definitieve statistieken (accuraatheid, recall, Brier-score, ECE) worden berekend. Vervolgens worden de SHAP-waarden berekend voor dit specifieke selected feature model.
  * **Output:** De Confusion Matrix (Supplementary Table S3), **Figuur 5** (De duidelijke scheiding in de t-SNE/PCA puntenwolk), en **Figuur 6A & 6B** (De definitieve bar- en beeswarm SHAP-plots van uitsluitend de gekozen biomarkers).

*(Optioneel) Bias Analyse:* Ik heb het blok voor age/sex bias even uitgeschakeld om het script niet te log te maken, maar je kunt het makkelijk later toevoegen als je dat nodig hebt voor je discussiehoofdstuk.

### Phase 2: Geometry-Based Riemannian Pipeline

* **1. Strict Preprocessing (`1_preprocess_riemann.py`):**
  Inherits the exact 80/20 subject split from the SVM pipeline. Processes raw arrays into Spatial Covariance Matrices (SCMs) for both Whole-Brain (19 channels) and Central ROI (9 channels) layouts across all 5 bands.
* **2. Tangent Space Classification (`2_train_riemann.py`):**
  Projects SCMs onto a flat tangent space originating at the Fréchet mean. Trains both TS-SVM and MDM (Minimum Distance to Mean) classifiers via Stratified Group K-Fold CV.
* **3. Riemann Validation & Topography (`4_evaluate_testset.py` & `3_plot_topo.py`):**
  Evaluates the frozen geometry models on the hold-out set, generating equivalent clinical metrics (AUPRC, ECE). Reconstructs Tangent Space weights back into anatomical electrode pairs to plot geometric network connections.
* **4. Feature Ranking (`5_riemann_feature_ranking.py`):**
  Extracts and ranks global feature weights from the Riemannian pipeline to generate a mathematical comparison against the SVM mSFFS selection.



* **`1_preprocess_riemann.py`** : (Dit was jouw script 1). Het filtert en genereert de covariantie-matrices. We moeten hier alleen zorgen dat de EC-filter 100% waterdicht is.
* **`2_SVM_Riemannian_Model_Training.py`** : (Het traint, optimaliseert (`GridSearchCV`) en kiest de winnaar (MDM of TS-SVM). Het bevriest het model (`.pkl`).
* **`3_Riemannian_Biomarker_Map.py`** : (Jouw script 3 & 5 samengevoegd). Als TS-SVM wint, extraheren we de gewichten en tekenen we de paper-ready hersenkaart (groen/bruin/roze) met 19 kanalen.
* **`4_Riemannian_Model_Evaluation.py`** : (Jouw script 4). Het evalueert de winnaar op de ongeziene testset. Het maakt het .txt-rapport (met accuraatheid, Brier, ECE, Precision, Recall) en de Confusion Matrix.
* **`5_Riemannian_Data_Distribution.py`** : We maken een t-SNE plot, maar dan op basis van de Riemannian datastructuur (Tangent Space).
* **`6_Riemannian_Training_Logbook.py`** : Het maakt het hyperparameter logboek voor Riemann.
* **`7_Riemannian_Female_Sensitivity.py`** : De female-only confounding check, maar dan op de Riemannian data.

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
