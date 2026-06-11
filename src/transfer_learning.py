"""
=============================================================================
6. CROSS-DOMAIN VALIDATION & TRANSFER LEARNING (Li et al., 2026)
=============================================================================
Overview:
    This script evaluates the generalizability of the trained models on an 
    entirely new dataset (Target Domain) using two methods:
    
    1. Direct Testing: Applying the frozen Source model directly to Target data.
    2. Transfer Learning: Using a subset of the Target data to recalibrate 
       the model (Instance Transfer/Domain Adaptation) before final testing.

Prerequisites:
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
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. CONFIG IMPORT & SETTINGS
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RESULTS_DIR, RANDOM_STATE

USE_ROI = True
TARGET_BAND = 'gamma' # Focus band for transfer learning
PREFIX = "ROI_" if USE_ROI else "ALL_"

# Bestanden definiëren
SOURCE_TRAIN_PATH = RESULTS_DIR / "final_dataset_train.csv" # De originele trainingsdata
MODEL_PATH = RESULTS_DIR / f"saved_model_{PREFIX}{TARGET_BAND}.pkl"

# TODO: Vul hier het pad in naar je nieuwe geëxtraheerde doel-dataset CSV (bijv. TDBrain of EO)
# Je kunt deze genereren door build_dataset.py tijdelijk te richten op je nieuwe dataset map.
TARGET_DATA_PATH = RESULTS_DIR / "target_domain_dataset.csv" 

# =============================================================================
# 1. LOAD ARTIFACTS & DATA
# =============================================================================
print(f"🚀 Starting Cross-Domain Validation for the {TARGET_BAND.upper()} band...")

if not MODEL_PATH.exists() or not TARGET_DATA_PATH.exists():
    raise FileNotFoundError("🚨 Model of Target Dataset niet gevonden. Controleer de paden.")

# Laad het bevroren model en de parameters
artifact = joblib.load(MODEL_PATH)
source_svm = artifact['model']
source_scaler = artifact['scaler']
roi_features = artifact['roi_features']
selected_features = artifact['selected_features']

print(f"  ✓ Bron-model geladen ({len(selected_features)} mSFFS features).")

# Laad Bron-data (voor de transfer learning stap)
source_df = pd.read_csv(SOURCE_TRAIN_PATH)
X_source_raw = source_df[roi_features]
X_source_scaled = pd.DataFrame(source_scaler.transform(X_source_raw), columns=roi_features)
X_source_final = X_source_scaled[selected_features].values
y_source = source_df['Target'].values

# Laad Doel-data (Nieuw domein)
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

# Predict direct met het onaangepaste model
y_pred_direct = source_svm.predict(X_target_final)
y_prob_direct = source_svm.predict_proba(X_target_final)[:, 1]

acc_direct = accuracy_score(y_target, y_pred_direct)
auc_direct = roc_auc_score(y_target, y_prob_direct)
brier_direct = brier_score_loss(y_target, y_prob_direct)

print(f"  -> Direct Accuracy: {acc_direct:.2%}")
print(f"  -> Direct ROC-AUC:  {auc_direct:.4f}")

# =============================================================================
# 3. METHOD 2: TRANSFER LEARNING (Domain Adaptation via Instance Weighting)
# =============================================================================
print("\n⚙️ METHOD 2: Transfer Learning (Instance-Weighted Adaptation)")
# We reserveren een klein deel van het doeldomein voor kalibratie (bijv. 20%)
# De rest (80%) wordt gebruikt voor de daadwerkelijke test.

# OPMERKING: Bij kleine datasets is het belangrijk om op subject-niveau te splitsen, 
# maar voor de eenvoud van de transfer learning demonstratie gebruiken we hier een stratifed split.
X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(
    X_target_final, y_target, test_size=0.80, random_state=RANDOM_STATE, stratify=y_target
)

print(f"  -> Kalibratiedata (Target): {X_tgt_train.shape[0]} segmenten.")
print(f"  -> Testdata (Target):       {X_tgt_test.shape[0]} segmenten.")

# Combineer Source en Target-Train
X_combined = np.vstack((X_source_final, X_tgt_train))
y_combined = np.concatenate((y_source, y_tgt_train))

# Instance Weighting: Target data is schaars, dus geven we deze een zwaarder gewicht in de SVM cost function
# Dit simuleert de gewichtsupdate-mechaniek van TrAdaBoost
weight_source = 1.0
weight_target = len(X_source_final) / len(X_tgt_train) if len(X_tgt_train) > 0 else 1.0
sample_weights = np.concatenate([
    np.full(len(X_source_final), weight_source),
    np.full(len(X_tgt_train), weight_target)
])

# Train het aangepaste model met de originele hyperparameters
adapted_svm = SVC(
    C=source_svm.C, 
    gamma=source_svm.gamma, 
    kernel='rbf', 
    probability=True, 
    random_state=RANDOM_STATE
)
adapted_svm.fit(X_combined, y_combined, sample_weight=sample_weights)

# Test het aangepaste model op de resterende 80% van de doeldomein data
y_pred_adapted = adapted_svm.predict(X_tgt_test)
y_prob_adapted = adapted_svm.predict_proba(X_tgt_test)[:, 1]

acc_adapted = accuracy_score(y_tgt_test, y_pred_adapted)
auc_adapted = roc_auc_score(y_tgt_test, y_prob_adapted)

# Voor een eerlijke vergelijking, testen we ook het baseline model op DEZELFDE 80% testset
y_pred_baseline = source_svm.predict(X_tgt_test)
acc_baseline_matched = accuracy_score(y_tgt_test, y_pred_baseline)

print(f"  -> Transfer Learning Accuracy: {acc_adapted:.2%}")
print(f"  -> (Ter vergelijking Baseline op zelfde testset: {acc_baseline_matched:.2%})")

# =============================================================================
# 4. VISUALIZATION & EXPORT
# =============================================================================
print("\n📊 Genereren van Transfer Learning Report...")

# Staafdiagram voor prestatievergelijking
labels = ['Direct Testing (Baseline)', 'Transfer Learning (Adapted)']
accuracies = [acc_baseline_matched * 100, acc_adapted * 100]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=['#7f7f7f', '#1f77b4'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'Cross-Domain Validation Performance ({TARGET_BAND.upper()} Band)', fontsize=14)

# Voeg percentages toe op de staven
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / f"transfer_learning_comparison_{TARGET_BAND}.png", dpi=300)
plt.close()

# Confusion Matrix van het Adapted Model
cm = confusion_matrix(y_tgt_test, y_pred_adapted)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy (0)', 'Fibro/CP (1)'], 
            yticklabels=['Healthy (0)', 'Fibro/CP (1)'],
            annot_kws={"size": 16})
plt.title(f'Transfer Learning - Target Domain Confusion Matrix\n(Accuracy: {acc_adapted:.2%})', fontsize=12)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"transfer_learning_cm_{TARGET_BAND}.png", dpi=300)
plt.close()

print(f"✅ Transfer Learning analyse compleet. Resultaten opgeslagen in {RESULTS_DIR.name}/")