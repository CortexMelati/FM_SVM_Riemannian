"""
=============================================================================
1. TRAIN SVM PIPELINE (mSFFS + Grid Search + Permutation)
=============================================================================
Overview:
    This script executes the training phase of the machine learning pipeline
    across ALL 5 frequency bands (delta, theta, alpha, beta, gamma).
    
    For each band, it will:
        1. Load the 80% training dataset and filter to the Central 9 ROI channels.
        2. Apply StratifiedGroupKFold to group 30s segments by Subject, 
           strictly preventing identity/data leakage during cross-validation.
        3. Standardize the features (mean=0, variance=1).
        4. Run mSFFS to dynamically find the optimal subset (1 to 20 features).
        5. Run Grid Search to optimize SVM hyperparameters (C and gamma).
        6. Perform a 1000-iteration permutation test to validate statistical significance.
        7. Save the trained model, scaler, and feature list as a frozen '.pkl' artifact.
        
    This script strictly isolates the learning process and does not use the 
    the 20% unseen test set.

Execution:
    python train_svm.py
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, permutation_test_score, GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIG IMPORT & ABLATION SWITCH
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RESULTS_DIR, RANDOM_STATE
from config import PROCESSED_DATA_DIR, FIGURES_DIR

# ---------------------------------------------------------
# THE ABLATION SWITCH
# True  = Benchmark mode (9 Central Channels - prevents EMG artifacts)
# False = Exploratory mode (All 19 Channels - tests dimensionality & noise)
USE_ROI = True
# ---------------------------------------------------------

# =============================================================================
# 1. LOAD TRAINING DATA ONLY
# =============================================================================
print("🚀 Loading Training Dataset...")
train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
train_df = pd.read_csv(train_path)

y_train = train_df['Target'].values
groups_train = train_df['Subject'].values  # CRUCIAL: For GroupKFold

meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_train_full = train_df.drop(columns=[c for c in meta_cols if c in train_df.columns])

bands = ['delta', 'theta', 'alpha', 'beta', 'gamma'] 
roi_channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
N_PERMUTATIONS = 1000

results_summary = []

mode_name = "BENCHMARK (9-Channel ROI)" if USE_ROI else "EXPLORATORY (All 19 Channels)"
print(f"\n⚙️ Starting Training Phase in {mode_name} mode...")

for band in bands:
    print("\n" + "="*60)
    print(f"🧠 TRAINING BAND: {band.upper()}")
    print("="*60)
    
    # --- A. FILTERING (Ablation Logic) ---
    selected_band_features = []
    for col in X_train_full.columns:
        if f'({band})' in col:
            pair = col.replace(f'({band})', '').split('-')
            
            if USE_ROI:
                # Keep only pairs where BOTH channels are in the 9-channel ROI
                if pair[0] in roi_channels and pair[1] in roi_channels:
                    selected_band_features.append(col)
            else:
                # Keep ALL pairs for this band
                selected_band_features.append(col)

    X_train_roi = X_train_full[selected_band_features]
    print(f"-> Feature space size before mSFFS: {X_train_roi.shape[1]} features")
    
    # --- B. SCALING ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_roi), columns=X_train_roi.columns)

    # --- C. STRATIFIED GROUP K-FOLD (Prevents Leakage) ---
    cv_strategy = StratifiedGroupKFold(n_splits=5) 
    # cv_strategy = RepeatedStratifiedGroupKFold(n_splits=10) 
    cv_splits = list(cv_strategy.split(X_train_scaled, y_train, groups=groups_train))

    # --- D. FEATURE SELECTION (mSFFS) ---
    print("-> Running mSFFS (1 to 20 features)...")
    base_svm = SVC(kernel='rbf', gamma='scale', random_state=RANDOM_STATE)
    
    sfs = SFS(
        base_svm, 
        k_features=(1, 20),    
        forward=True,          
        floating=True,         
        scoring='accuracy', # or 'roc_auc'
        cv=cv_splits,          
        n_jobs=-1              
    )
    
    sfs = sfs.fit(X_train_scaled, y_train)
    selected_features = list(sfs.k_feature_names_)
    print(f"-> Optimal subset found ({len(selected_features)} features)")
    
    fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev', figsize=(10, 6))
    plt.title(f'mSFFS Learning Curve (Subset Accuracy) - {band.upper()} Band')
    plt.ylabel('CV Accuracy')
    plt.xlabel('Number of Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = FIGURES_DIR / f"mSFFS_learning_curve_{band}.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"-> mSFFS Learning curve saved to {curve_path.name}")
        

    X_train_final = X_train_scaled[selected_features]

    # --- E. GRID SEARCH ---
    print("-> Running Grid Search for parameters...")
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': np.logspace(-4, 1.5, 20)}
    
    grid_search = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        param_grid, cv=cv_splits, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_final, y_train)
    final_svm = grid_search.best_estimator_
    
    internal_acc = grid_search.best_score_
    print(f"-> Internal CV Accuracy: {internal_acc:.4f}")

    # --- F. PERMUTATION TEST ---
    print(f"-> Running Permutation Test ({N_PERMUTATIONS} shuffles)...")
    score, permutation_scores, pvalue = permutation_test_score(
        final_svm, X_train_final, y_train, scoring="accuracy", 
        cv=cv_splits, n_permutations=N_PERMUTATIONS, n_jobs=-1, random_state=RANDOM_STATE
    )
    print(f"-> Permutation p-value: {pvalue:.4f}")

    # --- G. SAVE ARTIFACTS TO .PKL ---
    # Add a prefix to the filename if exploratory mode is used, so you don't overwrite benchmark models
    prefix = "ROI_" if USE_ROI else "ALL_"
    model_artifact = {
        'model': final_svm,
        'scaler': scaler,
        'roi_features': selected_band_features,
        'selected_features': selected_features,
        'mode': mode_name
    }
    model_path = PROCESSED_DATA_DIR / f"saved_model_{prefix}{band}.pkl"
    joblib.dump(model_artifact, model_path)
    print(f"-> Model saved successfully to {model_path.name}")

    results_summary.append({
        'Band': band.upper(),
        'Internal_CV_Accuracy': internal_acc,
        'P_Value': pvalue,
        'C_Param': grid_search.best_params_['C'],
        'Gamma_Param': round(grid_search.best_params_['gamma'], 4),
        'Num_Features': len(selected_features),
        'Selected_Features': " | ".join(selected_features)
    })

# --- H. SUMMARY REPORT ---
print("\n" + "*"*80)
print(f"🏆 TRAINING PHASE COMPLETED - {mode_name}")
summary_df = pd.DataFrame(results_summary).sort_values(by='Internal_CV_Accuracy', ascending=False)
print(summary_df[['Band', 'Internal_CV_Accuracy', 'P_Value', 'Num_Features']].to_string(index=False))
print("*"*80)