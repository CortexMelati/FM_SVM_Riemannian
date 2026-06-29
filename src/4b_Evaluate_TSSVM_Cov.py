"""
=============================================================================
4b. QUICK EVALUATE SECOND-BEST MODEL (TSSVM_Cov)
=============================================================================
Overview:
    Trains the specific TSSVM_Cov configuration (Theta, C=1, RBF) directly
    on the train set and evaluates it on the unseen hold-out test set.
    python 4b_Evaluate_TSSVM_Cov.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import mne

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score)
from sklearn.base import BaseEstimator, TransformerMixin

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, SVM_DATA_DIR, RIEMANN_FIGURES_DIR, BEST_CHANNELS_EVALUATE, CHANNELS_1020, BANDS, RANDOM_STATE

# --- Custom Transformers ---
class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X):
        iir_params = dict(order=4, ftype='butter', output='sos')
        return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=iir_params, verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]

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

def train_and_evaluate_second_model():
    print("🚀 STARTING QUICK EVALUATION: TSSVM_Cov (Theta Band)")

    # 1. Instellingen ophalen
    band_name = 'Theta'
    l_freq, h_freq = BANDS[band_name]
    ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]

    # 2. Data inladen
    X_train_raw = np.load(RIEMANN_DATA_DIR / "X_train_raw.npy")
    y_train = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    X_test_raw = np.load(RIEMANN_DATA_DIR / "X_test_raw.npy")
    y_test = np.load(RIEMANN_DATA_DIR / "y_test_riemann.npy")

    # 3. Bouw de specifieke pipeline (Met C=1 en RBF kernel)
    print("🧠 Training specific TSSVM_Cov model on Training Data...")
    pipeline = Pipeline([
        ('filter', MNEBandPass(l_freq, h_freq, 500)),
        ('roi', ROIExtractor(ROI_INDICES)),
        ('cov', Covariances(estimator='oas')),
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1, kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
    ])
    
    pipeline.fit(X_train_raw, y_train)

    # 4. Evalueer op Test Data
    print("\n-> Predicting on Unseen Data...")
    y_pred = pipeline.predict(X_test_raw)
    y_prob = pipeline.predict_proba(X_test_raw)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob) 
    brier = brier_score_loss(y_test, y_prob)
    ece = expected_calibration_error(y_test, y_prob)

    # 5. Rapport Genereren
    report_text = (
        f"====================================================\n"
        f" FINAL RIEMANNIAN TEST SET METRICS - THETA BAND \n"
        f" Architecture: TSSVM_Cov (Second-Best)\n"
        f"====================================================\n"
        f"Balanced Accuracy: {acc:.4f}\n"
        f"Precision:       {prec:.4f}\n"
        f"Sensitivity:     {rec:.4f}\n"
        f"ROC-AUC:         {auc:.4f}\n"
        f"AUPRC:           {auprc:.4f}\n"
        f"Brier Score:     {brier:.4f}\n"
        f"ECE:             {ece:.4f}\n"
        f"====================================================\n"
    )

    print(f"\n{report_text}")
    report_path = SVM_DATA_DIR / f"final_test_metrics_riemann_Theta_Cov.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    # 6. Confusion Matrix Plotten
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Healthy (0)', 'Fibro (1)'], 
                yticklabels=['Healthy (0)', 'Fibro (1)'],
                annot_kws={"size": 16})
    plt.title(f'Riemannian FINAL Validation (TSSVM Cov - THETA)\n(Accuracy: {acc:.2%})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plot_path = RIEMANN_FIGURES_DIR / f"final_confusion_matrix_riemann_Theta_Cov.png"
    plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ Opgeslagen als: {plot_path.name}")

if __name__ == "__main__":
    train_and_evaluate_second_model()