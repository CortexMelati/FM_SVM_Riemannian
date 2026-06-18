"""
=============================================================================
9. CROSS-DOMAIN VALIDATION & TRADABOOST (Li et al., 2026 Replication)
=============================================================================
Overview:
    Evaluates the generalizability of BOTH the feature-based (SVM) and 
    geometry-based (Riemannian) models on a Target Domain.
    
    Replicates Figure 7 by iterating through different calibration sample 
    sizes and computing a linear regression trendline for:
    1. Direct Testing (Zero-Shot)
    2. TrAdaBoost (Transfer Learning via `adapt` library)

Execution:
    python 9_cross_domain_validation.py
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from adapt.instance_based import TrAdaBoost

# ==========================================
# 0. CONFIG IMPORT & SETTINGS
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROJECT_ROOT, RANDOM_STATE

# dataset nog aanpassen 

# USE_ROI = True
# TARGET_BAND = 'gamma' # Focus band for transfer learning
# PREFIX = "ROI_" if USE_ROI else "ALL_"
# LAYOUT = "roi" if USE_ROI else "whole"


# # Explicitly configure your Source and Target dataset folders here
# SOURCE_DATASET = "FM_EO_dataset" # Folder with your originally trained SVM
# TARGET_DATASET = "cp_fm_dataset" # Folder with the new data (Target Domain)

# Construct cross-dataset paths
SOURCE_DIR = PROJECT_ROOT / "results" / SOURCE_DATASET / "processed_data"
TARGET_DIR = PROJECT_ROOT / "results" / TARGET_DATASET / "processed_data"
TARGET_FIGURES = PROJECT_ROOT / "results" / TARGET_DATASET / "figures"
TARGET_FIGURES.mkdir(parents=True, exist_ok=True) 

# Link the appropriate files
MODEL_PATH = SOURCE_DIR / f"saved_model_{PREFIX}{TARGET_BAND}.pkl"
SOURCE_TRAIN_PATH = SOURCE_DIR / "final_dataset_train.csv"
TARGET_DATA_PATH = TARGET_DIR / "final_dataset_train.csv"

# =============================================================================
# 1. LOAD ARTIFACTS & DATA
# =============================================================================
print(f"Starting Cross-Domain Validation (TrAdaBoost) for the {TARGET_BAND.upper()} band...")

print("\n--- PATH DIAGNOSTICS ---")
print(f"1. Searching for Source Model: {MODEL_PATH}")
print(f"   -> Found? {MODEL_PATH.exists()}")

print(f"2. Searching for Target Data:  {TARGET_DATA_PATH}")
print(f"   -> Found? {TARGET_DATA_PATH.exists()}\n")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Error: MODEL IS MISSING AT THIS PATH:\n{MODEL_PATH}")
if not TARGET_DATA_PATH.exists():
    raise FileNotFoundError(f"Error: TARGET DATA IS MISSING AT THIS PATH:\n{TARGET_DATA_PATH}")

# Load the frozen model and parameters
artifact = joblib.load(MODEL_PATH)
source_svm = artifact['model']
source_scaler = artifact['scaler']
roi_features = artifact['roi_features']
selected_features = artifact['selected_features']

print(f"  -> Source model loaded ({len(selected_features)} mSFFS features).")

# Load Source Data
source_df = pd.read_csv(SOURCE_TRAIN_PATH)
X_source_svm = pd.DataFrame(source_scaler.transform(source_df[roi_features]), columns=roi_features)[selected_features].values
y_source_svm = source_df['Target'].values

# Load Target Data
target_df = pd.read_csv(TARGET_DATA_PATH)
X_target_svm = pd.DataFrame(source_scaler.transform(target_df[roi_features]), columns=roi_features)[selected_features].values
y_target_svm = target_df['Target'].values

print(f"  -> Target dataset loaded: {X_target_svm.shape[0]} segments.")

# =============================================================================
# 2. LOAD RIEMANN MODEL & DATA (Optional Execution)
# =============================================================================
riemann_model_path = PROJECT_ROOT / "results" / SOURCE_DATASET / "riemann_data" / f"model_riemann_{TARGET_BAND.upper()}_{LAYOUT}_TSSVM.pkl"
try:
    riemann_pipeline = joblib.load(riemann_model_path)
    # Extract components to manually enter Tangent Space for TrAdaBoost
    ts_transformer = riemann_pipeline.named_steps['ts']
    r_scaler = riemann_pipeline.named_steps['scaler']
    r_svm = riemann_pipeline.named_steps['svm']
    
    # Load Covariances
    covs_source = np.load(PROJECT_ROOT / "results" / SOURCE_DATASET / "riemann_data" / f"covs_train_{TARGET_BAND.upper()}_{LAYOUT}.npy")
    y_source_riemann = np.load(PROJECT_ROOT / "results" / SOURCE_DATASET / "riemann_data" / "y_train_riemann.npy")
    
    covs_target = np.load(PROJECT_ROOT / "results" / TARGET_DATASET / "riemann_data" / f"covs_train_{TARGET_BAND.upper()}_{LAYOUT}.npy")
    y_target_riemann = np.load(PROJECT_ROOT / "results" / TARGET_DATASET / "riemann_data" / "y_train_riemann.npy")

    # PROJECTION FIX: Project 3D matrices to flat 2D Tangent Space factors for TrAdaBoost
    X_source_riem_2d = r_scaler.transform(ts_transformer.transform(covs_source))
    X_target_riem_2d = r_scaler.transform(ts_transformer.transform(covs_target))
    riemann_available = True
    print("  -> Riemann pipeline successfully loaded and projected to Tangent Space.")
except Exception as e:
    print(f"  Warning: Riemann files could not be loaded. Only SVM will be tested. Error: {e}")
    riemann_available = False

# =============================================================================
# 3. FIGURE 7 REPLICATION EXPERIMENT (Iterative Transfer Learning)
# =============================================================================
def run_transfer_experiment(X_src, y_src, X_tgt, y_tgt, base_estimator, model_name):
    print(f"\nRunning TrAdaBoost vs Direct Testing iterations for {model_name}...")
    
    results = []
    # Test different sizes of the target calibration set (from 5% to 50%)
    calibration_fractions = np.linspace(0.05, 0.50, 10) 
    
    for frac in calibration_fractions:
        # Split target data into calibration and hold-out test
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_tgt, y_tgt, train_size=frac, random_state=RANDOM_STATE, stratify=y_tgt
        )
        calib_size = len(y_calib)
        
        # 1. DIRECT TESTING (Zero-Shot on target test set)
        base_estimator.fit(X_src, y_src)
        acc_direct = accuracy_score(y_test, base_estimator.predict(X_test))
        
        # 2. TRADABOOST
        # TrAdaBoost requires slight regularization, so we clone the SVM
        boost_estimator = SVC(C=base_estimator.C, gamma=base_estimator.gamma, kernel='rbf', probability=True, random_state=RANDOM_STATE)
        tr_model = TrAdaBoost(estimator=boost_estimator, n_estimators=10, random_state=RANDOM_STATE)
        
        # Train with source data + calibration data
        tr_model.fit(X_src, y_src, Xt=X_calib, yt=y_calib)
        acc_transfer = accuracy_score(y_test, tr_model.predict(X_test))
        
        results.append({
            'Calibration_Samples': calib_size,
            'Direct_Testing_Acc': acc_direct,
            'Transfer_Learning_Acc': acc_transfer
        })
        print(f"   -> Calib={calib_size:02d} | Direct: {acc_direct:.3f} | TrAdaBoost: {acc_transfer:.3f}")
        
    return pd.DataFrame(results)

def plot_figure_7(df_results, model_name):
    plt.figure(figsize=(9, 6))
    
    X = df_results['Calibration_Samples'].values
    y_direct = df_results['Direct_Testing_Acc'].values
    y_transfer = df_results['Transfer_Learning_Acc'].values
    
    # Scatter points
    plt.scatter(X, y_direct, color='#5c8cbc', label='direct training', s=50, alpha=0.8)
    plt.scatter(X, y_transfer, color='#d62728', label='transfer learning', s=50, alpha=0.8)
    
    # Linear Regression Trendlines (Order = 1)
    z_dir = np.polyfit(X, y_direct, 1)
    p_dir = np.poly1d(z_dir)
    plt.plot(X, p_dir(X), color='gray', lw=2)
    
    z_trans = np.polyfit(X, y_transfer, 1)
    p_trans = np.poly1d(z_trans)
    plt.plot(X, p_trans(X), color='gray', lw=2)
    
    plt.title(f"Figure 7 Replication: Cross-Validation Scores on Target Set ({model_name})", fontsize=12, pad=15)
    plt.xlabel('Mean training subjects (Calibration Size)', fontsize=11)
    plt.ylabel('Mean test accuracy', fontsize=11)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(frameon=True, loc='upper left')
    
    save_path = TARGET_FIGURES / f"Figure7_Replication_{model_name.replace(' ', '')}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {save_path.name}")

# =============================================================================
# 4. EXECUTE & PLOT
# =============================================================================
# SVM Experiment
base_svm = SVC(C=source_svm.C, gamma=source_svm.gamma, kernel='rbf', random_state=RANDOM_STATE)
svm_results = run_transfer_experiment(X_source_svm, y_source_svm, X_target_svm, y_target_svm, base_svm, "Feature-Based SVM")
plot_figure_7(svm_results, "Feature-Based SVM")
svm_results.to_csv(TARGET_FIGURES / "transfer_results_svm.csv", index=False)

# Riemann Experiment (If available)
if riemann_available:
    base_riemann = SVC(kernel='linear', class_weight='balanced', random_state=RANDOM_STATE) # TS-SVM is linear
    riem_results = run_transfer_experiment(X_source_riem_2d, y_source_riemann, X_target_riem_2d, y_target_riemann, base_riemann, "Riemannian Tangent Space")
    plot_figure_7(riem_results, "Riemannian Tangent Space")
    riem_results.to_csv(TARGET_FIGURES / "transfer_results_riemann.csv", index=False)

print("\nCross-Domain Validation (TrAdaBoost) completed successfully!")