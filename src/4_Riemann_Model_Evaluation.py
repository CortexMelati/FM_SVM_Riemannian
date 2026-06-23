"""
=============================================================================
4. RIEMANNIAN MODEL EVALUATION (Test Set)
=============================================================================
Overview:
    This script evaluates the frozen Riemannian winning model (e.g., Delta TSSVM)
    on the strictly isolated 20% hold-out test set. 
    It computes the final clinical metrics and generates a confusion matrix,
    matching the exact evaluation protocol of the SVM pipeline.

Execution:
    python 4_Riemann_Model_Evaluation.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score)

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, SVM_DATA_DIR, RIEMANN_FIGURES_DIR

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_weight = np.sum(in_bin) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

def evaluate_riemann_testset():
    print("🚀 STARTING STEP 4: RIEMANNIAN EVALUATION ON UNSEEN TEST SET")

    # 1. ZOEK HET BEVROREN MODEL
    model_files = list(SVM_DATA_DIR.glob("model_riemann_*.pkl"))
    if not model_files:
        print("🚨 Geen bevroren Riemannian model gevonden in svm_data/.")
        sys.exit()
        
    model_path = model_files[-1]
    artifact = joblib.load(model_path)
    
    pipeline = artifact['model']
    band = artifact['band']
    layout = artifact['layout']
    model_type = artifact['model_type']
    
    print(f"-> Loaded Frozen Model: {band.upper()} Band | {layout.upper()} Layout | {model_type}")

    # 2. LAAD DE TEST DATA
    y_test_path = RIEMANN_DATA_DIR / "y_test_riemann.npy"
    covs_test_path = RIEMANN_DATA_DIR / f"covs_test_{band}_{layout}.npy"
    
    if not y_test_path.exists() or not covs_test_path.exists():
        print("🚨 Test data (y_test of covs_test) ontbreekt. Zorg dat Script 1 is gerund.")
        sys.exit()

    y_test = np.load(y_test_path)
    X_covs_test = np.load(covs_test_path)
    print(f"-> Test data loaded: {X_covs_test.shape[0]} segments.")

    # 3. VOORSPELLEN EN METRICS BEREKENEN
    print("\n-> Predicting on Unseen Data...")
    y_pred = pipeline.predict(X_covs_test)
    y_prob = pipeline.predict_proba(X_covs_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X_covs_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob) 
    brier = brier_score_loss(y_test, y_prob)
    ece = expected_calibration_error(y_test, y_prob)

    # 4. TEKST RAPPORT GENEREREN
    report_text = (
        f"====================================================\n"
        f" FINAL RIEMANNIAN TEST SET METRICS - {band.upper()} BAND \n"
        f"====================================================\n"
        f"Accuracy:        {acc:.4f}\n"
        f"Precision:       {prec:.4f}\n"
        f"Recall:          {rec:.4f}\n"
        f"ROC-AUC:         {auc:.4f}\n"
        f"AUPRC:           {auprc:.4f}\n"
        f"Brier Score:     {brier:.4f}\n"
        f"ECE:             {ece:.4f}\n"
        f"====================================================\n"
    )

    print(f"\n{report_text}")
    report_path = SVM_DATA_DIR / f"final_test_metrics_riemann_{band}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"-> Metrics report permanently saved to: svm_data/{report_path.name}")

    # 5. CONFUSION MATRIX PLOTTEN
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', # Oranje om hem te onderscheiden van de blauwe SVM plot
                xticklabels=['Healthy (0)', 'Fibro (1)'], 
                yticklabels=['Healthy (0)', 'Fibro (1)'],
                annot_kws={"size": 16})
    plt.title(f'Riemannian FINAL Validation ({band.upper()})\n(Accuracy: {acc:.2%})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RIEMANN_FIGURES_DIR / f"final_confusion_matrix_riemann_{band}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Confusion Matrix saved to: riemann_figures/{plot_path.name}")

if __name__ == "__main__":
    evaluate_riemann_testset()