"""
=============================================================================
8. CROSS-DOMAIN VALIDATION & TRADABOOST (Li et al., 2026 Replication)
=============================================================================
Overview:
    Replicates Section 2.7 and 3.4 (Table 1 & Figure 7).
    Evaluates robustness on an external Target Domain (e.g., NCCP) using the
    7 identified beta-band features.
    
    Compares two methodologies across 2 to N stratified cross-validation folds:
    1. Direct Training: Training a NEW SVM exclusively on the target training folds.
    2. Transfer Learning: Using TrAdaBoost (Source Data + Target Training folds).

Execution:
    pip install adapt (if not installed)
    python 8_SVM_cross_domain_validation.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Suppresses adapt library warnings

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# Ensure the ADAPT library is installed for TrAdaBoost
try:
    from adapt.instance_based import TrAdaBoost
except ImportError:
    print("FATAL ERROR: The 'adapt' library is not installed.")
    print("Please run: pip install adapt")
    sys.exit()

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (PROCESSED_DATA_DIR, SVM_DATA_DIR, SVM_FIGURES_DIR, 
                    FOCUS_BAND, CROSS_TARGET_DATASET, RANDOM_STATE)

print(f"Starting Cross-Domain Validation (Fig 7) for {FOCUS_BAND.upper()} band...")

# =============================================================================
# 1. LOAD DATA & ARTIFACTS
# =============================================================================
# A. Load Frozen Model Artifact (for hyperparameters & features)
model_path = SVM_DATA_DIR / f"saved_model_{FOCUS_BAND}.pkl"
if not model_path.exists():
    print(f"Error: {model_path.name} not found. Run Script 4 first.")
    sys.exit()
    
artifact = joblib.load(model_path)
frozen_svm = artifact['model']
selected_features = artifact['features']
print(f"-> Loaded architecture: {len(selected_features)} features, C={frozen_svm.C}, gamma={frozen_svm.gamma}")

# B. Load Source Data (Required for TrAdaBoost)
source_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
source_df = pd.read_csv(source_path)
if 'Condition' in source_df.columns:
    source_df = source_df[source_df['Condition'] == 'EC'].copy()

# C. Load Target Data
target_path = PROCESSED_DATA_DIR / f"target_domain_{CROSS_TARGET_DATASET.lower()}.csv"
if not target_path.exists():
    print(f"Error: Target data {target_path.name} not found. Check build_dataset.py.")
    sys.exit()

target_df = pd.read_csv(target_path)
if 'Condition' in target_df.columns:
    target_df = target_df[target_df['Condition'] == 'EC'].copy()
    
# Clean Target Domain NaNs just in case
target_df = target_df.dropna(subset=['Target'] + selected_features).copy()

print(f"-> Source Domain Data: {len(source_df)} segments.")
print(f"-> Target Domain Data: {len(target_df)} segments ({CROSS_TARGET_DATASET}).")

# =============================================================================
# 2. DATA PREPARATION (Strict Feature Isolation)
# =============================================================================
# We fit a fresh scaler on Source Data, and apply it to both
scaler = StandardScaler()
X_source = scaler.fit_transform(source_df[selected_features])
y_source = source_df['Target'].values

X_target = scaler.transform(target_df[selected_features])
y_target = target_df['Target'].values
groups_target = target_df['Subject'].values

unique_target_subjects = len(np.unique(groups_target))

# =============================================================================
# 3. FIGURE 7 ITERATIVE EXPERIMENT (2 to N Folds)
# =============================================================================
# The paper varies the number of folds to simulate increasing calibration data
# They used 2 to 24 folds. We will dynamically adapt to your dataset size.
max_folds = min(24, unique_target_subjects) 
fold_range = range(2, max_folds + 1)

results = []

print("\nRunning iterative cross-domain testing (This may take a few minutes)...")
print(f"{'Folds':<10} | {'Avg Train Subjects':<20} | {'Transfer Acc':<15} | {'Direct Acc':<15}")
print("-" * 65)

for n_splits in fold_range:
    cv_strategy = StratifiedGroupKFold(n_splits=n_splits)
    
    transfer_scores = []
    direct_scores = []
    train_subjects_count = []
    
    for train_idx, test_idx in cv_strategy.split(X_target, y_target, groups=groups_target):
        X_tgt_tr, y_tgt_tr = X_target[train_idx], y_target[train_idx]
        X_tgt_te, y_tgt_te = X_target[test_idx], y_target[test_idx]
        
        # Track number of unique subjects used for training in this fold
        num_subjects_in_fold = len(np.unique(groups_target[train_idx]))
        train_subjects_count.append(num_subjects_in_fold)
        
        # Method 1: DIRECT TRAINING (New model on Target data only)
        # Using the same hyperparameters discovered during source optimization
        direct_svm = SVC(C=frozen_svm.C, gamma=frozen_svm.gamma, kernel='rbf', random_state=RANDOM_STATE)
        direct_svm.fit(X_tgt_tr, y_tgt_tr)
        acc_direct = accuracy_score(y_tgt_te, direct_svm.predict(X_tgt_te))
        direct_scores.append(acc_direct)
        
        # Method 2: TRANSFER LEARNING (TrAdaBoost)
        # Uses both Source Data + Target Training Data to predict Target Test Data
        boost_base = SVC(C=frozen_svm.C, gamma=frozen_svm.gamma, kernel='rbf', probability=True, random_state=RANDOM_STATE)
        # Note: Using 10 estimators for speed. Paper uses standard TrAdaBoost config.
        tr_model = TrAdaBoost(estimator=boost_base, n_estimators=10, random_state=RANDOM_STATE)
        
        # X_source, y_source = Source domain. Xt, yt = Target calibration domain.
        tr_model.fit(X_source, y_source, Xt=X_tgt_tr, yt=y_tgt_tr)
        acc_transfer = accuracy_score(y_tgt_te, tr_model.predict(X_tgt_te))
        transfer_scores.append(acc_transfer)
        
    avg_train_subs = np.mean(train_subjects_count)
    mean_transfer = np.mean(transfer_scores)
    mean_direct = np.mean(direct_scores)
    
    print(f"{n_splits:<10} | {avg_train_subs:<20.1f} | {mean_transfer:<15.3f} | {mean_direct:<15.3f}")
    
    results.append({
        'Total_Folds': n_splits,
        'Avg_Train_Subjects': avg_train_subs,
        'Transfer_Learning': mean_transfer,
        'Direct_Training': mean_direct
    })

results_df = pd.DataFrame(results)
table_path = SVM_DATA_DIR / f"Table_1_cross_domain_results_{FOCUS_BAND}.csv"
results_df.to_csv(table_path, index=False, float_format='%.3f')

# =============================================================================
# 4. PLOT FIGURE 7 REPLICATION
# =============================================================================
plt.figure(figsize=(9, 6))

X = results_df['Avg_Train_Subjects'].values
y_direct = results_df['Direct_Training'].values
y_transfer = results_df['Transfer_Learning'].values

# Scatter points
plt.scatter(X, y_direct, color='#5c8cbc', label='direct training', s=60, alpha=0.9, edgecolor='white')
plt.scatter(X, y_transfer, color='#d62728', label='transfer learning', s=60, alpha=0.9, edgecolor='white')

# Linear Regression Trendlines (Order = 1)
z_dir = np.polyfit(X, y_direct, 1)
p_dir = np.poly1d(z_dir)
plt.plot(X, p_dir(X), color='gray', lw=2.5, alpha=0.8)

z_trans = np.polyfit(X, y_transfer, 1)
p_trans = np.poly1d(z_trans)
plt.plot(X, p_trans(X), color='gray', lw=2.5, alpha=0.8)

plt.title(f"Figure 7: Cross-validation scores on {CROSS_TARGET_DATASET} target set", fontsize=14, pad=15)
plt.xlabel('Mean training subjects', fontsize=12)
plt.ylabel('Mean test accuracy', fontsize=12)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(frameon=True, loc='upper left', fontsize=11)

plt.tight_layout()
fig_path = SVM_FIGURES_DIR / f"Figure_7_Cross_Domain_Validation_{FOCUS_BAND}.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"\n-> Table 1 Exported to: svm_data/{table_path.name}")
print(f"-> Figure 7 Exported to: svm_figures/{fig_path.name}")
print("PIPELINE COMPLETE.")