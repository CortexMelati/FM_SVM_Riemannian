"""
=============================================================================
6. CROSS-DOMAIN VALIDATION & TRADABOOST (Li et al., 2026)
=============================================================================
Overview:
    This script evaluates the generalizability of the trained models on an 
    entirely new dataset (Target Domain) using two methods:
    
    !!! at least 2 datasets need to have been processed. 
    
    1. Direct Testing: Applying the frozen Source model directly to Target data.
    2. TrAdaBoost: Using the ADAPT library to perform genuine Transfer AdaBoost,
       reweighting source domain data against a small target calibration set.

Prerequisites:
    - pip install adapt
    - A trained source model (.pkl) from train_svm.py.
    - A newly generated dataset CSV for the target domain (e.g., TDBrain).

Execution:
    python transfer_learning.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from adapt.instance_based import TrAdaBoost

# ==========================================
# 0. CONFIG IMPORT & SETTINGS
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROJECT_ROOT, RANDOM_STATE

USE_ROI = True
TARGET_BAND = 'gamma' # Focus band for transfer learning
PREFIX = "ROI_" if USE_ROI else "ALL_"

# Vul hier expliciet de namen van je twee mappen in
SOURCE_DATASET = "FM_EO_dataset" # De map met je origineel getrainde SVM
TARGET_DATASET = "cp_fm_dataset" # De map met de nieuwe data (doeldomein)

# Construeer de paden over de grenzen van de actieve dataset heen
SOURCE_DIR = PROJECT_ROOT / "results" / SOURCE_DATASET / "processed_data"
TARGET_DIR = PROJECT_ROOT / "results" / TARGET_DATASET / "processed_data"
TARGET_FIGURES = PROJECT_ROOT / "results" / TARGET_DATASET / "figures"
TARGET_FIGURES.mkdir(parents=True, exist_ok=True) # Zorg dat de map bestaat

# Koppel de juiste bestanden
MODEL_PATH = SOURCE_DIR / f"saved_model_{PREFIX}{TARGET_BAND}.pkl"
SOURCE_TRAIN_PATH = SOURCE_DIR / "final_dataset_train.csv"

# Gebruik de train-set van het nieuwe domein als target data
TARGET_DATA_PATH = TARGET_DIR / "final_dataset_train.csv"
# =============================================================================
# 1. LOAD ARTIFACTS & DATA
# =============================================================================
print(f"🚀 Starting Cross-Domain Validation (TrAdaBoost) for the {TARGET_BAND.upper()} band...")

print("\n🔍 --- PATH DIAGNOSTICS ---")
print(f"1. Zoeken naar Source Model: {MODEL_PATH}")
print(f"   -> Gevonden? {MODEL_PATH.exists()}")

print(f"2. Zoeken naar Target Data:  {TARGET_DATA_PATH}")
print(f"   -> Gevonden? {TARGET_DATA_PATH.exists()}\n")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"🚨 HET MODEL ONTBREEKT OP DIT PAD:\n{MODEL_PATH}")
if not TARGET_DATA_PATH.exists():
    raise FileNotFoundError(f"🚨 DE TARGET DATA ONTBREEKT OP DIT PAD:\n{TARGET_DATA_PATH}")

# Laad het bevroren model en de parameters
artifact = joblib.load(MODEL_PATH)
source_svm = artifact['model']
source_scaler = artifact['scaler']
roi_features = artifact['roi_features']
selected_features = artifact['selected_features']

print(f"  ✓ Bron-model geladen ({len(selected_features)} mSFFS features).")

# Laad Bron-data (Source Domain)
source_df = pd.read_csv(SOURCE_TRAIN_PATH)
X_source_raw = source_df[roi_features]
X_source_scaled = pd.DataFrame(source_scaler.transform(X_source_raw), columns=roi_features)
X_source_final = X_source_scaled[selected_features].values
y_source = source_df['Target'].values

# Laad Doel-data (Target Domain)
target_df = pd.read_csv(TARGET_DATA_PATH)
X_target_raw = target_df[roi_features]
X_target_scaled = pd.DataFrame(source_scaler.transform(X_target_raw), columns=roi_features)
X_target_final = X_target_scaled[selected_features].values
y_target = target_df['Target'].values

print(f"  ✓ Doel-dataset geladen: {X_target_final.shape[0]} segmenten.")

# =============================================================================
# 2. METHOD 1: DIRECT TESTING (Zero-Shot Baseline)
# =============================================================================
print("\n⚙️ METHOD 1: Direct Testing (Baseline)")

# Predict direct met het onaangepaste bron-model
y_pred_direct = source_svm.predict(X_target_final)
y_prob_direct = source_svm.predict_proba(X_target_final)[:, 1]

acc_direct = accuracy_score(y_target, y_pred_direct)
auc_direct = roc_auc_score(y_target, y_prob_direct)

print(f"  -> Direct Accuracy: {acc_direct:.2%}")
print(f"  -> Direct ROC-AUC:  {auc_direct:.4f}")

# =============================================================================
# 3. METHOD 2: TRADABOOST (Transfer Learning)
# =============================================================================
print("\n⚙️ METHOD 2: TrAdaBoost (Genuine Transfer Learning)")
# We reserveren een klein deel van het doeldomein voor de iteratieve kalibratie (bijv. 20%)
# De rest (80%) wordt gebruikt voor de daadwerkelijke test.

X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(
    X_target_final, y_target, test_size=0.80, random_state=RANDOM_STATE, stratify=y_target
)

print(f"  -> Kalibratiedata (Target Train): {X_tgt_train.shape[0]} segmenten.")
print(f"  -> Testdata (Target Test):        {X_tgt_test.shape[0]} segmenten.")

# Initialiseer de Base Estimator (SVM met de reeds gevonden optimale C en gamma)
base_estimator = SVC(
    C=source_svm.C, 
    gamma=source_svm.gamma, 
    kernel='rbf', 
    probability=True, 
    random_state=RANDOM_STATE
)

# Initialiseer TrAdaBoost algoritme via de adapt library
# We itereren 10 keer om de optimale weging tussen bron- en doeldata te vinden
tr_model = TrAdaBoost(
    estimator=base_estimator,
    n_estimators=10,
    random_state=RANDOM_STATE
)

# Fit TrAdaBoost: Bron-data wordt meegegeven in fit(), Doel-data via Xt en yt
print("  ⏳ TrAdaBoost is aan het trainen (Boosting iterations)...")
tr_model.fit(X_source_final, y_source, Xt=X_tgt_train, yt=y_tgt_train)

# Test het TrAdaBoost model op de resterende 80% van de doeldomein data
y_pred_adapted = tr_model.predict(X_tgt_test)
y_prob_adapted = tr_model.predict_proba(X_tgt_test)[:, 1]

acc_adapted = accuracy_score(y_tgt_test, y_pred_adapted)
auc_adapted = roc_auc_score(y_tgt_test, y_prob_adapted)

# Voor een eerlijke vergelijking, testen we ook het baseline model op DEZELFDE 80% testset
y_pred_baseline = source_svm.predict(X_tgt_test)
acc_baseline_matched = accuracy_score(y_tgt_test, y_pred_baseline)

print(f"  -> TrAdaBoost Accuracy: {acc_adapted:.2%}")
print(f"  -> (Ter vergelijking Baseline op zelfde testset: {acc_baseline_matched:.2%})")

# =============================================================================
# 4. VISUALIZATION & EXPORT
# =============================================================================
print("\n📊 Genereren van TrAdaBoost Report...")

# Staafdiagram voor prestatievergelijking
labels = ['Direct Testing (Baseline)', 'TrAdaBoost (Transfer Learning)']
accuracies = [acc_baseline_matched * 100, acc_adapted * 100]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=['#7f7f7f', '#2ca02c']) # Groen voor succesvolle transfer
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'Cross-Domain Validation Performance ({TARGET_BAND.upper()} Band)', fontsize=14)

# Voeg percentages toe op de staven
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(TARGET_FIGURES / f"transfer_learning_TrAdaBoost_comparison_{TARGET_BAND}.png", dpi=300)
plt.close()

# Confusion Matrix van het TrAdaBoost Model
cm = confusion_matrix(y_tgt_test, y_pred_adapted)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Healthy (0)', 'Fibro/CP (1)'], 
            yticklabels=['Healthy (0)', 'Fibro/CP (1)'],
            annot_kws={"size": 16})
plt.title(f'TrAdaBoost - Target Domain Confusion Matrix\n(Accuracy: {acc_adapted:.2%})', fontsize=12)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(TARGET_FIGURES / f"transfer_learning_TrAdaBoost_cm_{TARGET_BAND}.png", dpi=300)
plt.close()

print(f"✅ TrAdaBoost analyse compleet. Resultaten opgeslagen in {TARGET_FIGURES.name}/")