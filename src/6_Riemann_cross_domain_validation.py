"""
=============================================================================
6. RIEMANNIAN CROSS-DOMAIN VALIDATION & TRADABOOST 
=============================================================================
Overview:
    Replicates Section 2.7 and 3.4 (Table 1 & Figure 7) for the Riemannian model.
    Evaluates robustness on an external Target Domain (e.g., NCCP).
    
    To make TrAdaBoost mathematically possible, the target domain covariance 
    matrices are projected into the 2D Tangent Space using the reference point 
    (Frechet Mean) learned from the Source Domain.

Execution:
    python 6_Riemann_cross_domain_validation.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from adapt.instance_based import TrAdaBoost
except ImportError:
    print("FATAL ERROR: The 'adapt' library is not installed. Run: pip install adapt")
    sys.exit()

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, SVM_DATA_DIR, RIEMANN_FIGURES_DIR, CROSS_TARGET_DATASET, RANDOM_STATE

def run_riemann_cross_domain():
    print("🚀 STARTING STEP 6: RIEMANNIAN CROSS-DOMAIN VALIDATION (TRADABOOST)")

    # 1. LOAD SOURCE ARTIFACT (To extract Tangent Space mapping)
    model_files = list(SVM_DATA_DIR.glob("model_riemann_*.pkl"))
    if not model_files:
        print("🚨 Geen bevroren Riemannian model gevonden.")
        sys.exit()
        
    model_path = model_files[-1]
    artifact = joblib.load(model_path)
    pipeline = artifact['model']
    band = artifact['band']
    layout = artifact['layout']
    
    print(f"-> Loaded Source Architecture: {band.upper()} Band | {layout.upper()} Layout")
    
    ts_transformer = pipeline.named_steps['ts']
    scaler = pipeline.named_steps['scaler']
    frozen_svm = pipeline.named_steps['svm'] # Contains the optimal C parameter

    # 2. LOAD SOURCE DATA & PROJECT TO TANGENT SPACE
    y_source = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    covs_source = np.load(RIEMANN_DATA_DIR / f"covs_train_{band}_{layout}.npy")
    
    X_source_ts = ts_transformer.transform(covs_source)
    X_source = scaler.transform(X_source_ts)
    print(f"-> Source Data mapped to 2D Tangent Space: {X_source.shape}")

    # 3. LOAD TARGET DATA (External Dataset)
    # Note: Ensure you have generated the covariance matrices for your target dataset!
    # For now, we assume a naming convention similar to the config target name.
    target_covs_path = RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band}_{layout}.npy"
    target_labels_path = RIEMANN_DATA_DIR / f"target_y_{CROSS_TARGET_DATASET.lower()}.npy"
    target_groups_path = RIEMANN_DATA_DIR / f"target_groups_{CROSS_TARGET_DATASET.lower()}.npy"
    
    if not target_covs_path.exists():
        print(f"\n🚨 TARGET DATA ONTBREKT! Kan {target_covs_path.name} niet vinden.")
        print("Zorg ervoor dat je doeldomein data (NCCP) eerst is omgezet naar covariantie-matrices.")
        sys.exit()

    covs_target = np.load(target_covs_path)
    y_target = np.load(target_labels_path)
    groups_target = np.load(target_groups_path)
    
    # Project Target data using the SOURCE domain's Tangent Space rules
    X_target_ts = ts_transformer.transform(covs_target)
    X_target = scaler.transform(X_target_ts)
    print(f"-> Target Data mapped to 2D Tangent Space: {X_target.shape}")

    # 4. ITERATIVE EXPERIMENT (2 to N Folds)
    unique_target_subjects = len(np.unique(groups_target))
    max_folds = min(24, unique_target_subjects) 
    fold_range = range(2, max_folds + 1)
    results = []

    print("\nRunning iterative cross-domain testing...")
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
            
            num_subjects_in_fold = len(np.unique(groups_target[train_idx]))
            train_subjects_count.append(num_subjects_in_fold)
            
            # Method 1: DIRECT TRAINING
            direct_svm = SVC(C=frozen_svm.C, kernel='linear', class_weight='balanced', random_state=RANDOM_STATE)
            direct_svm.fit(X_tgt_tr, y_tgt_tr)
            direct_scores.append(accuracy_score(y_tgt_te, direct_svm.predict(X_tgt_te)))
            
            # Method 2: TRANSFER LEARNING (TrAdaBoost)
            boost_base = SVC(C=frozen_svm.C, kernel='linear', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
            tr_model = TrAdaBoost(estimator=boost_base, n_estimators=10, random_state=RANDOM_STATE)
            tr_model.fit(X_source, y_source, Xt=X_tgt_tr, yt=y_tgt_tr)
            transfer_scores.append(accuracy_score(y_tgt_te, tr_model.predict(X_tgt_te)))
            
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

    # 5. SAVE CSV (Table 1)
    results_df = pd.DataFrame(results)
    table_path = RIEMANN_DATA_DIR / f"Table_1_Riemann_cross_domain_{band}.csv"
    results_df.to_csv(table_path, index=False, float_format='%.3f')

    # 6. PLOT FIGURE 7
    plt.figure(figsize=(9, 6))
    X_vals = results_df['Avg_Train_Subjects'].values
    
    plt.scatter(X_vals, results_df['Direct_Training'], color='#5c8cbc', label='direct training', s=60, alpha=0.9, edgecolor='white')
    plt.scatter(X_vals, results_df['Transfer_Learning'], color='#d62728', label='transfer learning', s=60, alpha=0.9, edgecolor='white')

    z_dir = np.polyfit(X_vals, results_df['Direct_Training'], 1)
    plt.plot(X_vals, np.poly1d(z_dir)(X_vals), color='gray', lw=2.5, alpha=0.8)

    z_trans = np.polyfit(X_vals, results_df['Transfer_Learning'], 1)
    plt.plot(X_vals, np.poly1d(z_trans)(X_vals), color='gray', lw=2.5, alpha=0.8)

    plt.title(f"Riemannian Fig 7: Cross-domain validation on {CROSS_TARGET_DATASET}", fontsize=14, pad=15)
    plt.xlabel('Mean training subjects', fontsize=12)
    plt.ylabel('Mean test accuracy', fontsize=12)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(frameon=True, loc='upper left', fontsize=11)

    plt.tight_layout()
    fig_path = RIEMANN_FIGURES_DIR / f"Figure_7_Riemann_Cross_Domain_{band}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\n✅ Pipeline Complete.")
    print(f"-> Table saved to: riemann_data/{table_path.name}")
    print(f"-> Figure saved to: riemann_figures/{fig_path.name}")

if __name__ == "__main__":
    run_riemann_cross_domain()