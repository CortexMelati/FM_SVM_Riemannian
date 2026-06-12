"""
=============================================================================
5. SHAP ANALYSIS PIPELINE (Li et al., 2026 Replication)
=============================================================================
Overview:
    This script opens the frozen SVM model artifacts and the test dataset 
    to calculate SHapley Additive exPlanations (SHAP) values.
    
    It generates two figures replicating the paper:
    1. Mean Absolute SHAP values (Bar plot - Feature Importance)
    2. SHAP values summary (Bee swarm plot - Impact on model output)

Execution:
    python shap_analysis.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mne
from pathlib import Path
import sys
import joblib

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RESULTS_DIR
from config import PROCESSED_DATA_DIR, FIGURES_DIR

def run_shap_analysis(target_band='gamma', use_roi=True):
    print(f"🚀 Starting SHAP Analysis for the {target_band.upper()} band...")
    
    # 1. Load the frozen model artifact
    prefix = "ROI_" if use_roi else "ALL_"
    model_path = PROCESSED_DATA_DIR / f"saved_model_{prefix}{target_band}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"🚨 Model {model_path.name} niet gevonden. Run train_svm.py eerst.")
        
    artifact = joblib.load(model_path)
    final_svm = artifact['model']
    scaler = artifact['scaler']
    roi_features = artifact['roi_features']
    selected_features = artifact['selected_features']
    
    print(f"  ✓ Model geladen met {len(selected_features)} mSFFS geselecteerde features.")

    # 2. Load the test dataset (SHAP is evaluated on the unseen data)
    test_path = PROCESSED_DATA_DIR / "final_dataset_test.csv"
    test_df = pd.read_csv(test_path)
    
    # Filter and scale exactly like in the test script
    X_test_roi = test_df[roi_features]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_roi), columns=roi_features)
    X_test_final = X_test_scaled[selected_features]

    # 3. Initialize SHAP KernelExplainer for SVM (RBF)
    # We use the test set itself as the background distribution to save computation time,
    # optionally you could use shap.kmeans(X_train, 10) as background.
    print("  ⏳ Berekenen van SHAP values (Dit kan even duren bij RBF kernels)...")
    
    # Gebruik predict_proba om de impact op de kans voor Fibromyalgie (klasse 1) te meten
    explainer = shap.KernelExplainer(final_svm.predict_proba, X_test_final)
    shap_values = explainer.shap_values(X_test_final)
    
    # Omdat het binaire classificatie is (HC=0, FM=1), geeft predict_proba 2 arrays terug.
    # We zijn geïnteresseerd in de SHAP values voor de voorspelling van FM (index 1).
    shap_values_fm = shap_values[1] if isinstance(shap_values, list) else shap_values

    # 4. Generate & Save Plots
    print("  📊 Genereren van visualisaties...")
    
    # A. Bar Plot (Mean Absolute SHAP) - Replicates Fig 6A
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_fm, X_test_final, plot_type="bar", show=False)
    plt.title(f"{target_band.upper()} Band - Mean Absolute SHAP Values (Feature Importance)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"SHAP_bar_plot_{target_band}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # B. Summary Bee Swarm Plot - Replicates Fig 6B
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_fm, X_test_final, show=False)
    plt.title(f"{target_band.upper()} Band - SHAP Values Summary")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"SHAP_summary_plot_{target_band}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ SHAP-analyse compleet. Plots opgeslagen in {RESULTS_DIR.name}/")


if __name__ == "__main__":
    # Optioneel: Loop over alle banden, maar de focus van de paper ligt op Gamma
    # run_shap_analysis(target_band='theta')
    run_shap_analysis(target_band='gamma', use_roi=True)