"""
=============================================================================
4. EVALUATE UNSEEN TEST SET (RIEMANNIAN MULTI-BAND & DUAL LAYOUT)
=============================================================================
Overview:
    This script evaluates the frozen Riemannian models (trained in Step 2) 
    exclusively on the 20% hold-out unseen test set generated in Step 1.
    It computes the final clinical performance metrics (Unseen Accuracy, 
    ROC-AUC, Sensitivity, Specificity, and now AUPRC, Precision, and ECE) 
    across both Whole-Brain and ROI layouts.

Execution:
    python 4_evaluate_testset.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
from sklearn.metrics import (recall_score, roc_auc_score, brier_score_loss, 
                             balanced_accuracy_score, average_precision_score, 
                             precision_score)

# ==========================================
# 0. CONFIG IMPORT & SYSTEM SETUP
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, BANDS, RIEMANN_DATA_DIR

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) across 10 probability bins.
    Essential for quantifying the clinical reliability of the model's confidence.
    """
    bin_edges = np.linspace(0., 1., n_bins + 1)
    binned_true = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = binned_true == i
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            bin_weight = np.sum(bin_mask) / len(y_true)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

def evaluate_testset():
    print("🚀 STARTING STEP 4: RIGOROUS EVALUATION ON UNSEEN HOLD-OUT TEST SET")
    
    # Load the actual ground truth targets for the test population
    y_test_path = RIEMANN_DATA_DIR / "y_test_riemann.npy"
    if not y_test_path.exists():
        raise FileNotFoundError(f"🚨 Test labels missing at: {y_test_path.name}. Execute Step 1 first.")
        
    y_test = np.load(y_test_path)
    test_results = []

    # Iterate systematically through each canonical frequency band
    for band_name in BANDS.keys():
        # Iterate over both spatial layouts
        for layout in ['whole', 'roi']:
            covs_test_path = RIEMANN_DATA_DIR / f"covs_test_{band_name}_{layout}.npy"
            
            if not covs_test_path.exists():
                print(f"  ⚠️ Warning: Test covariance matrices for {band_name} ({layout}) not found. Skipping.")
                continue
                
            covs_test = np.load(covs_test_path)
            
            # Test both saved model architectures per frequency band and layout
            for model_type in ['TSSVM', 'MDM']:
                model_path = RIEMANN_DATA_DIR / f"model_riemann_{band_name}_{layout}_{model_type}.pkl"
                
                if not model_path.exists():
                    print(f"  ⚠️ Warning: Model architecture artifact '{model_path.name}' missing. Skipping.")
                    continue
                    
                # Load the frozen pipeline state
                model = joblib.load(model_path)
                
                # Predict discrete classes and continuous class probabilities
                y_pred = model.predict(covs_test)
                y_prob = model.predict_proba(covs_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(covs_test)
                
                # Compute standardized generalization metrics
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                sens = recall_score(y_test, y_pred, pos_label=1)
                spec = recall_score(y_test, y_pred, pos_label=0)
                roc_auc = roc_auc_score(y_test, y_prob)
                brier = brier_score_loss(y_test, y_prob)
                
                # Compute the newly added clinical performance metrics
                auprc = average_precision_score(y_test, y_prob)
                prec = precision_score(y_test, y_pred, zero_division=0)
                ece = expected_calibration_error(y_test, y_prob)
                
                # Append scores to the master results list
                test_results.append({
                    'Band': band_name,
                    'Layout': layout,
                    'Model': model_type,
                    'Bal_Acc': bal_acc,
                    'AUPRC': auprc,
                    'ROC_AUC': roc_auc,
                    'Sensitivity': sens,
                    'Specificity': spec,
                    'Precision': prec,
                    'Brier_Score': brier,
                    'ECE': ece
                })

    if not test_results:
        print("❌ Evaluation failed: No models or test datasets were successfully evaluated.")
        return

    # ==========================================
    # 1. EXPORT & SUMMARY PRINTING
    # ==========================================
    df_results = pd.DataFrame(test_results)
    
    # Sort the performance matrix hierarchically for academic presentation
    df_results = df_results.sort_values(by=['Layout', 'Model', 'Bal_Acc'], ascending=[True, True, False])
    
    print("\n🏆 FINAL METRIC REPORT (UNSEEN TARGET POPULATION):")
    print("-" * 120)
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 120)
    
    # Export to CSV for explicit integration within latex table environments
    output_path = RIEMANN_DATA_DIR / "riemann_testset_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Unseen verification complete. Performance data saved to: {output_path.name}")

if __name__ == "__main__":
    evaluate_testset()