"""
=============================================================================
7. SVM Female-Only Sensitivity Analysis (Confounding Check)
=============================================================================
Overview:
    This script addresses potential sex-related confounding as detailed by
    Li et al. (2026). It isolates the female subjects within the training 
    dataset, extracts the frozen optimal features, and re-evaluates the 
    cross-validation performance to confirm that the model's predictive 
    capability is not driven by sex imbalance.

Execution:
    python 7_SVM_female_sensitivity_analysis.py
=============================================================================
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (PROCESSED_DATA_DIR, SVM_DATA_DIR, FOCUS_BAND, 
                    CP_FM_DIR, RANDOM_STATE)

print(f"Starting Female-Only Sensitivity Analysis ({FOCUS_BAND.upper()} Band)...")

# =============================================================================
# 1. LOAD TRAINING DATA & FILTER FOR EYES CLOSED (EC)
# =============================================================================
train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
if not train_path.exists():
    print(f"Error: Could not find {train_path.name}. Please run your previous dataset scripts first.")
    sys.exit()

train_df = pd.read_csv(train_path)

if 'Condition' in train_df.columns:
    train_df = train_df[train_df['Condition'] == 'EC'].copy()

# =============================================================================
# 2. MERGE WITH DEMOGRAPHIC METADATA & ISOLATE FEMALES
# =============================================================================
tsv_path = CP_FM_DIR / "data" / "participants.tsv"
if not tsv_path.exists():
    print(f"FATAL ERROR: Cannot find participants.tsv at path:\n{tsv_path}")
    sys.exit()

participants_df = pd.read_csv(tsv_path, sep='\t')

# Match indices exactly matching the fix from Script 5
if 'participant_id' in participants_df.columns:
    participants_df['Subject'] = participants_df['participant_id']

merged_df = pd.merge(train_df, participants_df[['Subject', 'sex']], on='Subject', how='inner')

if merged_df.empty:
    print("Error: Merge failed. Subject IDs between training data and participants.tsv do not match.")
    sys.exit()

# Filter for female participants only ('f' or 'F')
female_df = merged_df[merged_df['sex'].str.lower() == 'f'].copy()
print(f"-> Total segments in training set: {len(train_df)}")
print(f"-> Isolated female-only subset:    {len(female_df)} segments.")

y_female = female_df['Target'].values
groups_female = female_df['Subject'].values

# =============================================================================
# 3. LOAD FROZEN ARCHITECTURE (Features & Hyperparameters)
# =============================================================================
model_path = SVM_DATA_DIR / f"saved_model_{FOCUS_BAND}.pkl"
if not model_path.exists():
    print(f"Error: Frozen model {model_path.name} not found. Run Script 4 first.")
    sys.exit()

artifact = joblib.load(model_path)
selected_features = artifact['features']
frozen_svm = artifact['model']

print(f"-> Loaded {len(selected_features)} optimal mSFFS features from frozen artifact.")
print(f"-> Loaded optimized hyperparameters: C={frozen_svm.C}, gamma={frozen_svm.gamma}")

X_female = female_df[selected_features]

# =============================================================================
# 4. SCALING & STRATIFIED GROUP K-FOLD (Female-Only Space)
# =============================================================================
scaler = StandardScaler()
X_female_scaled = pd.DataFrame(scaler.fit_transform(X_female), columns=selected_features)

n_folds = 5
cv_strategy = StratifiedGroupKFold(n_splits=n_folds)
cv_splits = list(cv_strategy.split(X_female_scaled, y_female, groups=groups_female))

# =============================================================================
# 5. CROSS-VALIDATION EVALUATION
# =============================================================================
print(f"-> Running {n_folds}-fold Stratified Group CV on female subset...")

# Initialize a clean SVM model using the exact frozen parameters
sensitivity_svm = SVC(
    kernel='rbf',
    C=frozen_svm.C,
    gamma=frozen_svm.gamma,
    class_weight=frozen_svm.class_weight,
    random_state=RANDOM_STATE
)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_female_scaled.iloc[train_idx], y_female[train_idx]
    X_val, y_val = X_female_scaled.iloc[val_idx], y_female[val_idx]
    
    sensitivity_svm.fit(X_tr, y_tr)
    # Replicate balanced accuracy metric to remain consistent
    score = sensitivity_svm.score(X_val, y_val)
    fold_scores.append(score)
    print(f"   Fold {fold + 1}: Balanced Accuracy = {score:.4f}")

# Calculate Final Statistical Distributions
mean_cv = np.mean(fold_scores)
std_cv = np.std(fold_scores)
ci_margin = 1.96 * (std_cv / np.sqrt(n_folds))
ci_lower, ci_upper = mean_cv - ci_margin, mean_cv + ci_margin

print("\n" + "="*60)
print(" SENSITIVITY ANALYSIS RESULTS (FEMALE-ONLY)")
print("="*60)
print(f"-> Mean CV Balanced Accuracy: {mean_cv:.4f}")
print(f"-> Standard Deviation (SD):   {std_cv:.4f}")
print(f"-> 95% Confidence Interval:   [{ci_lower:.4f} - {ci_upper:.4f}]")
print("="*60)

# Save results for automated LaTeX text generation
output_df = pd.DataFrame([{
    'Analysis': 'Female-Only Sensitivity',
    'Mean_Accuracy': round(mean_cv, 4),
    'Std_Dev': round(std_cv, 4),
    '95_CI_Lower': round(ci_lower, 4),
    '95_CI_Upper': round(ci_upper, 4),
    'N_Segments': len(female_df)
}])
output_path = SVM_DATA_DIR / f"svm_female_sensitivity_results_{FOCUS_BAND}.csv"
output_df.to_csv(output_path, index=False)
print(f"-> Sensitivity report securely saved to: svm_data/{output_path.name}\n")