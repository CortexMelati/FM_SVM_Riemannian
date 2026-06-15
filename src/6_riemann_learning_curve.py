"""
=============================================================================
6. SUBJECT-INSULATED LEARNING CURVE DIAGNOSTICS & PLOTTING
=============================================================================
Overview:
    This unified script executes a subject-insulated learning curve cross-
    validation across incremental sample sizes (20% to 100%) and instantly 
    generates a publication-quality multi-panel line visualization.

Key Features:
    - Guarantees strict subject isolation to prevent data leakage.
    - Automates data export and multi-panel subplot line plotting.

Execution:
    python 6_riemann_learning_curve.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
import joblib

# ==========================================
# 0. CONFIG IMPORT & SYSTEM SETUP
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, FIGURES_DIR, RANDOM_STATE, BANDS

def run_learning_curve_pipeline():
    print("🚀 STARTING STEP 6: LEARNING CURVE DIAGNOSTICS AND AUTOMATED PLOTTING...")
    
    y_train = np.load(PROCESSED_DATA_DIR / "y_train_riemann.npy")
    groups_train = np.load(PROCESSED_DATA_DIR / "groups_train_riemann.npy")
    
    data_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    curve_records = []

    # --- PART 1: COMPUTE METRICS ---
    for band_name in BANDS.keys():
        X_covs = np.load(PROCESSED_DATA_DIR / f"covs_train_{band_name}.npy")
        
        model_path = PROCESSED_DATA_DIR / f"model_riemann_{band_name}_TSSVM.pkl"
        if not model_path.exists(): 
            print(f"  ⚠️ Template model for {band_name} missing. Skipping.")
            continue
        base_pipeline = joblib.load(model_path)
        
        print(f"  ⏳ Evaluating trajectory metrics for band: {band_name.upper()}")

        for fraction in data_fractions:
            fold_train_scores = []
            fold_val_scores = []
            
            for train_idx, val_idx in cv.split(X_covs, y_train, groups_train):
                fold_train_groups = groups_train[train_idx]
                fold_unique_subs = np.unique(fold_train_groups)
                
                np.random.seed(RANDOM_STATE)
                sub_sample_size = max(2, int(len(fold_unique_subs) * fraction))
                selected_subs = np.random.choice(fold_unique_subs, size=sub_sample_size, replace=False)
                
                sub_mask = np.isin(groups_train, selected_subs) & np.isin(range(len(groups_train)), train_idx)
                
                if np.sum(sub_mask) == 0 or len(np.unique(y_train[sub_mask])) < 2:
                    continue
                
                estimator = clone(base_pipeline)
                estimator.fit(X_covs[sub_mask], y_train[sub_mask])
                
                train_pred = estimator.predict(X_covs[sub_mask])
                val_pred = estimator.predict(X_covs[val_idx])
                
                fold_train_scores.append(balanced_accuracy_score(y_train[sub_mask], train_pred))
                fold_val_scores.append(balanced_accuracy_score(y_train[val_idx], val_pred))

            curve_records.append({
                'Band': band_name,
                'Subject_Fraction': f"{int(fraction*100)}%",
                'Subject_Fraction_Num': int(fraction * 100),
                'Mean_Train_Accuracy': np.mean(fold_train_scores),
                'Mean_Validation_Accuracy': np.mean(fold_val_scores)
            })

    df_curves = pd.DataFrame(curve_records)
    df_curves.to_csv(PROCESSED_DATA_DIR / "riemann_learning_curves.csv", index=False)
    print("  ✓ Metric matrices successfully serialized to disk.")

    # --- PART 2: AUTOMATED PLOTTING ---
    print("🎨 Initializing multi-panel graphic export...")
    unique_bands = df_curves['Band'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Styled color palette optimized for print and grayscale transitions
    train_color = '#1f77b4'  # Standard Dark Blue
    val_color = '#d62728'    # Standard Dark Red

    for idx, band_name in enumerate(unique_bands):
        ax = axes[idx]
        band_df = df_curves[df_curves['Band'] == band_name].sort_values(by='Subject_Fraction_Num')
        
        ax.plot(band_df['Subject_Fraction_Num'], band_df['Mean_Train_Accuracy'], 
                label='Training Set', color=train_color, linestyle='-', marker='o', lw=2.5)
        
        ax.plot(band_df['Subject_Fraction_Num'], band_df['Mean_Validation_Accuracy'], 
                label='Validation Fold', color=val_color, linestyle='--', marker='s', lw=2.5)
        
        ax.set_title(f"{band_name.upper()} Band", fontsize=14, fontweight='bold', pad=10)
        ax.axhline(0.50, color='gray', linestyle=':', alpha=0.6) # Theoretical chance line
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_ylim(0.35, 1.05)
        
        if idx >= 3:
            ax.set_xlabel('Training Subject Volume (%)', fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel('Balanced Accuracy', fontsize=11)

    # Convert the empty 6th subplot frame into a clean centralized legend space
    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=14, frameon=True, facecolor='white', edgecolor='none')
    
    plt.suptitle("TS-SVM Learning Curves: Convergence Trajectories Across Spectral Bands", 
                 fontsize=16, fontweight='bold', y=0.96)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    chart_path = FIGURES_DIR / "riemann_learning_curves.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"✅ Pipeline complete. Line diagnostics chart saved directly to: {chart_path.name}")

if __name__ == "__main__":
    run_learning_curve_pipeline()