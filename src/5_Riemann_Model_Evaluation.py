"""
=============================================================================
5. RIEMANNIAN MODEL EVALUATION (Test Set - Subject Level)
=============================================================================
Overview:
    This script evaluates the frozen Riemannian winning models 
    on the strictly isolated 20% hold-out test set across all frequency bands.
    Crucially, it applies Majority Voting to group the 1-second epochs back 
    into clinical predictions per subject, matching the SVM evaluation protocol.

Execution:
    python 5_Riemann_Model_Evaluation.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import ast
import mne

from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score)

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, BANDS, BEST_CHANNELS_EVALUATE, CHANNELS_1020, RANDOM_STATE

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

def evaluate_riemann_testset():
    print("🚀 STARTING STEP 5: RIEMANNIAN EVALUATION ON UNSEEN TEST SET (SUBJECT-LEVEL)")

    X_train_raw = np.load(RIEMANN_DATA_DIR / "X_train_raw.npy")
    y_train = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    X_test_raw = np.load(RIEMANN_DATA_DIR / "X_test_raw.npy")
    y_test = np.load(RIEMANN_DATA_DIR / "y_test_riemann.npy")
    # LOAD THE GROUPS FOR THE TEST SET TO ENABLE MAJORITY VOTING
    groups_test = np.load(RIEMANN_DATA_DIR / "groups_test_riemann.npy") 
    
    scoreboard_path = RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv"
    if not scoreboard_path.exists():
        sys.exit("🚨 Scoreboard not found! Please run Script 2/3 first.")
    scoreboard = pd.read_csv(scoreboard_path)
    
    ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]
    valid_architectures = ['TSSVM_Cov', 'TSSVM_Xdawn']

    final_results = []

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"\n{'='*50}\n📡 ANALYZING BAND: {band_name.upper()}\n{'='*50}")
        
        band_scores = scoreboard[(scoreboard['Band'] == band_name.upper()) & (scoreboard['Architecture'].isin(valid_architectures))]
        if band_scores.empty:
            print(f"⚠️ No valid results found for {band_name}. Skipping...")
            continue
            
        best_row = band_scores.loc[band_scores['CV_Balanced_Accuracy'].idxmax()]
        arch = best_row['Architecture']
        params = ast.literal_eval(best_row['Optimal_Params'])
        
        print(f"-> Optimal Architecture: {arch}")
        # FIX 1: Verwijder svm__ prefix
        print(f"-> Optimal Params: C={params['C']}, Kernel={params['kernel']}")
        
        steps = [
            ('filter', MNEBandPass(l_freq, h_freq, 500)),
            ('roi', ROIExtractor(ROI_INDICES))
        ]
        
        if arch == 'TSSVM_Cov':
            steps.extend([('cov', Covariances(estimator='oas')), ('ts', TangentSpace(metric='riemann'))])
        elif arch == 'TSSVM_Xdawn':
            steps.extend([('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), ('ts', TangentSpace(metric='riemann'))])
            
        # FIX 2: Verwijder svm__ prefix in de model setup
        steps.extend([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=params['C'], kernel=params['kernel'], class_weight='balanced', probability=True, random_state=RANDOM_STATE))
        ])
        
        pipeline = Pipeline(steps)
        print("-> Training optimal pipeline on full training data...")
        pipeline.fit(X_train_raw, y_train)

        print("-> Predicting on Unseen Test Data (1-second epochs)...")
        y_pred_epochs = pipeline.predict(X_test_raw)
        y_prob_epochs = pipeline.predict_proba(X_test_raw)[:, 1]

        # --- APPLY MAJORITY VOTING FOR SUBJECT-LEVEL EVALUATION ---
        print("-> Aggregating predictions to Subject-Level...")
        df_preds = pd.DataFrame({
            'Subject': groups_test,
            'True_Label': y_test,
            'Pred_Class': y_pred_epochs,
            'Pred_Prob': y_prob_epochs
        })

        # Calculate the majority vote and average probability per subject
        df_subject = df_preds.groupby('Subject').agg(
            True_Label=('True_Label', 'first'), 
            Pred_Class=('Pred_Class', lambda x: x.mode()[0]), # Majority Vote
            Pred_Prob=('Pred_Prob', 'mean') # Average confidence
        ).reset_index()

        y_test_sub = df_subject['True_Label'].values
        y_pred_sub = df_subject['Pred_Class'].values
        y_prob_sub = df_subject['Pred_Prob'].values

        print(f"-> Evaluation compressed from {len(y_test)} epochs to {len(y_test_sub)} unique subjects/macro-segments.")

        # --- CALCULATE METRICS ON SUBJECT LEVEL ---
        acc = accuracy_score(y_test_sub, y_pred_sub)
        prec = precision_score(y_test_sub, y_pred_sub, zero_division=0)
        rec = recall_score(y_test_sub, y_pred_sub, zero_division=0)
        auc = roc_auc_score(y_test_sub, y_prob_sub)
        auprc = average_precision_score(y_test_sub, y_prob_sub) 
        brier = brier_score_loss(y_test_sub, y_prob_sub)
        ece = expected_calibration_error(y_test_sub, y_prob_sub)

        # FIX 3: Verwijder svm__ prefix in de export tabel
        final_results.append({
            'Band': band_name.upper(),
            'Optimal_Architecture': arch,
            'Optimal_Params': f"C={params['C']}, {params['kernel']}",
            'Bal_Accuracy': f"{acc:.2%}",
            'Sensitivity': f"{rec:.2%}",
            'Precision': f"{prec:.2%}",
            'AUPRC': f"{auprc:.4f}",
            'AUROC': f"{auc:.4f}",
            'Brier': f"{brier:.4f}",
            'ECE': f"{ece:.4f}"
        })

        # --- GENERATE PLOTS ---
        cm = confusion_matrix(y_test_sub, y_pred_sub)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Healthy (0)', 'Fibro (1)'], 
                    yticklabels=['Healthy (0)', 'Fibro (1)'],
                    annot_kws={"size": 16})
        plt.title(f'Riemannian FINAL Validation ({arch} - {band_name.upper()})\nSubject-Level (Accuracy: {acc:.2%})', fontsize=14)
        plt.ylabel('True Clinical Diagnosis', fontsize=12)
        plt.xlabel('Predicted Diagnosis (Majority Vote)', fontsize=12)
        plt.tight_layout()
        
        RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = RIEMANN_FIGURES_DIR / f"final_confusion_matrix_riemann_{band_name}_{arch}.png"
        plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()

    # 6. EXPORT MASTER TABLE FOR LATEX
    results_df = pd.DataFrame(final_results)
    csv_path = RIEMANN_DATA_DIR / "final_riemannian_test_table.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}\n🏆 ALL BANDS EVALUATED SUCCESSFULLY (SUBJECT-LEVEL)!\n{'='*70}")
    print("Here is your final data for the LaTeX Table 2:\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_riemann_testset()