"""
=============================================================================
4. EVALUATE UNSEEN TEST SET (RIEMANNIAN MULTI-BAND)
=============================================================================
Overview:
    This script evaluates the frozen Riemannian models (trained in Step 2) 
    exclusively on the 20% hold-out unseen test set generated in Step 1.
    It computes the final clinical performance metrics (Unseen Accuracy, 
    ROC-AUC, Sensitivity, Specificity) for direct thesis baseline comparison.

Execution:
    python 4_evaluate_testset.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
from sklearn.metrics import recall_score, roc_auc_score, brier_score_loss, balanced_accuracy_score

# ==========================================
# 0. CONFIG IMPORT & SYSTEM SETUP
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, BANDS

def evaluate_testset():
    print("🚀 STARTING STEP 4: RIGOROUS EVALUATION ON UNSEEN HOLD-OUT TEST SET")
    
    # Load the actual ground truth targets for the test population
    y_test_path = PROCESSED_DATA_DIR / "y_test_riemann.npy"
    if not y_test_path.exists():
        raise FileNotFoundError(f"🚨 Test labels missing at: {y_test_path.name}. Execute Step 1 first.")
        
    y_test = np.load(y_test_path)
    test_results = []

    # Iterate systematically through each canonical frequency band
    for band_name in BANDS.keys():
        covs_test_path = PROCESSED_DATA_DIR / f"covs_test_{band_name}.npy"
        
        if not covs_test_path.exists():
            print(f"  ⚠️ Warning: Test covariance matrices for {band_name} not found. Skipping.")
            continue
            
        covs_test = np.load(covs_test_path)
        
        # Test both saved model architectures per frequency band
        for model_type in ['TSSVM', 'MDM']:
            model_path = PROCESSED_DATA_DIR / f"model_riemann_{band_name}_{model_type}.pkl"
            
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
            
            # Append scores to the master results list
            test_results.append({
                'Band': band_name,
                'Model': model_type,
                'Unseen_Accuracy': bal_acc,
                'ROC_AUC': roc_auc,
                'Brier_Score': brier,
                'Sensitivity': sens,
                'Specificity': spec
            })

    if not test_results:
        print("❌ Evaluation failed: No models or test datasets were successfully evaluated.")
        return

    # ==========================================
    # 1. EXPORT & SUMMARY PRINTING
    # ==========================================
    df_results = pd.DataFrame(test_results)
    
    # Sort the performance matrix hierarchically for academic presentation
    df_results = df_results.sort_values(by=['Model', 'Unseen_Accuracy'], ascending=[True, False])
    
    print("\n🏆 FINAL METRIC REPORT (UNSEEN TARGET POPULATION):")
    print("-" * 100)
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 100)
    
    # Export to CSV for explicit integration within latext table environments
    output_path = PROCESSED_DATA_DIR / "riemann_testset_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Unseen verification complete. Performance data saved to: {output_path.name}")

if __name__ == "__main__":
    evaluate_testset()