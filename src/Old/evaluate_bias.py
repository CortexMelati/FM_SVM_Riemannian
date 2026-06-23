"""
=============================================================================
7. ALGORITHMIC BIAS EVALUATION
=============================================================================
Overview:
    This script evaluates the trained models (SVM & Riemannian) for demographic 
    (Sex, Age) or hardware/cohort-specific (Study) biases by merging their 
    predictions on the hold-out test set with the participants.tsv metadata.
    
Execution:
    python 7_evaluate_bias.py
=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, recall_score

# ==========================================
# 0. CONFIG & PATHS
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, SVM_DATA_DIR, PROJECT_ROOT, RIEMANN_DATA_DIR
from config import USE_ROI, PREFIX

TARGET_BAND = 'gamma'
LAYOUT = "roi" if USE_ROI else "whole"

print("Starting Algorithmic Bias Evaluation...")

# =============================================================================
# 1. LOAD PREDICTIONS FROM BOTH MODELS
# =============================================================================
# --- Load Test Data ---
test_path = PROCESSED_DATA_DIR / "final_dataset_test.csv"
test_df = pd.read_csv(test_path)
y_true = test_df['Target'].values

# --- A. SVM Predictions ---
svm_model_path = SVM_DATA_DIR / f"saved_model_{PREFIX}{TARGET_BAND}.pkl"
svm_artifact = joblib.load(svm_model_path)
final_svm = svm_artifact['model']
scaler = svm_artifact['scaler']
roi_features = svm_artifact['roi_features']
selected_features = svm_artifact['selected_features']

X_test_scaled = pd.DataFrame(scaler.transform(test_df[roi_features]), columns=roi_features)
X_test_final = X_test_scaled[selected_features]
test_df['SVM_Pred'] = final_svm.predict(X_test_final)

# --- B. Riemann Predictions ---
riemann_model_path = RIEMANN_DATA_DIR / f"model_riemann_{TARGET_BAND.upper()}_{LAYOUT}_TSSVM.pkl"
try:
    riemann_model = joblib.load(riemann_model_path)
    covs_test = np.load(RIEMANN_DATA_DIR / f"covs_test_{TARGET_BAND.upper()}_{LAYOUT}.npy")
    test_df['Riemann_Pred'] = riemann_model.predict(covs_test)
    riemann_available = True
    print("  -> Both models successfully loaded for bias analysis.")
except Exception as e:
    print(f"  Warning: Riemann model could not be loaded. Only SVM will be evaluated. Error: {e}")
    riemann_available = False

# =============================================================================
# 2. MERGE WITH METADATA (participants.tsv)
# =============================================================================
tsv_path = PROJECT_ROOT / "data" / "CP_FM_dataset" / "data" / "participants.tsv"
participants_df = pd.read_csv(tsv_path, sep='\t')

# Match the 'participant_id' (e.g., 'sub-FMpa01') with our 'Subject' column
if 'participant_id' in participants_df.columns:
    participants_df['Subject'] = participants_df['participant_id'].str.replace('sub-', '')
    
# Merge on Subject
merged_df = pd.merge(test_df, participants_df, on='Subject', how='inner')

if merged_df.empty:
    print("Error: The merge failed. Check if the 'Subject' IDs match across datasets!")
    sys.exit()

# Handle Age formatting (convert 'n/a' to NaN and bin ages)
merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 40, 55, 100], labels=['<40', '40-55', '>55'])

# =============================================================================
# 3. CALCULATE METRICS PER SUBGROUP
# =============================================================================
bias_results = []

def evaluate_subgroup(df, category_name, category_value):
    n_samples = len(df)
    if n_samples == 0: return
    
    y_t = df['Target']
    
    # Calculate SVM
    svm_acc = accuracy_score(y_t, df['SVM_Pred'])
    svm_sens = recall_score(y_t, df['SVM_Pred'], pos_label=1, zero_division=0)
    svm_spec = recall_score(y_t, df['SVM_Pred'], pos_label=0, zero_division=0)
    
    # Calculate Riemann (if available)
    if riemann_available:
        riem_acc = accuracy_score(y_t, df['Riemann_Pred'])
        riem_sens = recall_score(y_t, df['Riemann_Pred'], pos_label=1, zero_division=0)
        riem_spec = recall_score(y_t, df['Riemann_Pred'], pos_label=0, zero_division=0)
    else:
        riem_acc, riem_sens, riem_spec = np.nan, np.nan, np.nan
        
    bias_results.append({
        'Category': category_name,
        'Group': category_value,
        'N_Segments': n_samples,
        'SVM_Accuracy': svm_acc,
        'SVM_Sensitivity': svm_sens,
        'SVM_Specificity': svm_spec,
        'Riemann_Accuracy': riem_acc,
        'Riemann_Sensitivity': riem_sens,
        'Riemann_Specificity': riem_spec
    })

# Evaluate by SEX
for sex in merged_df['sex'].dropna().unique():
    evaluate_subgroup(merged_df[merged_df['sex'] == sex], 'Sex', sex.upper())

# Evaluate by AGE GROUP
for age_grp in merged_df['age_group'].dropna().unique():
    evaluate_subgroup(merged_df[merged_df['age_group'] == age_grp], 'Age', age_grp)

# Evaluate by STUDY (Cohort)
for study in merged_df['study'].dropna().unique():
    evaluate_subgroup(merged_df[merged_df['study'] == study], 'Study/Cohort', study)

# =============================================================================
# 4. PRINT REPORT & EXPORT TO CSV
# =============================================================================
results_df = pd.DataFrame(bias_results)
results_df = results_df.sort_values(by=['Category', 'Group'])

print("\nDEMOGRAPHIC BIAS REPORT (UNSEEN TEST SET):")
print("-" * 110)
# Print a clean version for the console
display_df = results_df[['Category', 'Group', 'N_Segments', 'SVM_Accuracy', 'Riemann_Accuracy']]
print(display_df.to_string(index=False, float_format=lambda x: f"{x:.2%}"))
print("-" * 110)

# Save to CSV for LaTeX integration
output_path = PROCESSED_DATA_DIR / f"algorithmic_bias_report_{TARGET_BAND}.csv"
results_df.to_csv(output_path, index=False)
print(f"Full bias report saved to: {output_path.name}")