"""
=============================================================================
ABLATION STUDY: Nested CV vs Single CV (RIEMANNIAN)
=============================================================================
Dit script toetst of de hyperparameter overschatting (Single CV bias)
ook optreedt in het Riemannian (TSSVM) framework op de trainingsset.

Output:
    - Een tekstueel rapport (.txt) met de exacte scores per band.
    - Een staafdiagram (.png) vergelijkbaar met de SVM ablation.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import mne

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, SVM_FIGURES_DIR, RANDOM_STATE, BANDS, BEST_BANDS, SFREQ_MAP, ACTIVE_DATASET_NAME

SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)

class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): 
        return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

def run_riemann_ablation():
    print("🚀 STARTING ABLATION STUDY: NESTED VS SINGLE CV (RIEMANNIAN)\n" + "="*70)

    X_raw = np.load(RIEMANN_DATA_DIR / "X_train_raw.npy")
    y = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_train_riemann.npy")

    svm_param_grid = [
        {'svm__C': [0.001, 0.01, 0.1, 1, 10], 'svm__kernel': ['linear']},
        {'svm__C': [0.001, 0.01, 0.1, 1, 10], 'svm__kernel': ['rbf'], 'svm__gamma': ['scale', 'auto']}
    ]

    cv_inner = StratifiedGroupKFold(n_splits=5)
    cv_outer = StratifiedGroupKFold(n_splits=5)

    results = []

    for band_name in BEST_BANDS:
        l_freq, h_freq = BANDS[band_name]
        print(f"📡 PROCESSING BAND: {band_name.upper()} (TSSVM_Xdawn)")

        # Bouw de volledige pipeline op
        steps = [
            ('filter', MNEBandPass(l_freq, h_freq, SFREQ)),
            ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), 
            ('ts', TangentSpace(metric='riemann')), 
            ('scaler', StandardScaler()),
            ('svm', SVC(class_weight='balanced', random_state=RANDOM_STATE))
        ]
        fe_pipeline = Pipeline(steps)

        # --- 1. Single CV (Biased) ---
        print("   -> Calculating Single CV (GridSearchCV on full train set)...")
        clf_single = GridSearchCV(
            estimator=fe_pipeline, param_grid=svm_param_grid, cv=cv_inner, 
            scoring='balanced_accuracy', n_jobs=-1
        )
        clf_single.fit(X_raw, y, groups=groups)
        single_cv_score = clf_single.best_score_
        
        # --- 2. Nested CV (Unbiased - Handmatige loop) ---
        print("   -> Calculating Nested CV (Iterative evaluation)...")
        nested_scores = []
        for train_idx, test_idx in cv_outer.split(X_raw, y, groups=groups):
            X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            g_tr = groups[train_idx]

            inner_gs = GridSearchCV(
                estimator=fe_pipeline, param_grid=svm_param_grid, cv=cv_inner, 
                scoring='balanced_accuracy', n_jobs=-1
            )
            inner_gs.fit(X_tr, y_tr, groups=g_tr)
            
            best_model = inner_gs.best_estimator_
            y_pred = best_model.predict(X_te)
            
            from sklearn.metrics import balanced_accuracy_score
            fold_acc = balanced_accuracy_score(y_te, y_pred)
            nested_scores.append(fold_acc)

        nested_cv_score_mean = np.mean(nested_scores)
        bias = single_cv_score - nested_cv_score_mean
        
        print(f"   Single CV: {single_cv_score:.4f} | Nested CV: {nested_cv_score_mean:.4f} | Overestimation: {bias:.4f}\n")
        
        results.append({
            'Band': band_name.upper(),
            'Single CV (Biased)': single_cv_score,
            'Nested CV (Unbiased)': nested_cv_score_mean,
            'Bias (Overestimation)': bias
        })

    df_results = pd.DataFrame(results)

    # =============================================================================
    # OUTPUT 1: TEXT REPORT (.txt)
    # =============================================================================
    report_path = RIEMANN_DATA_DIR / "ablation_study_cv_report_riemann.txt"
    with open(report_path, "w") as f:
        f.write("=====================================================================\n")
        f.write(" ABLATION STUDY: NESTED VS SINGLE CV (RIEMANNIAN TSSVM_Xdawn)\n")
        f.write("=====================================================================\n")
        f.write(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n=====================================================================\n")
    print(f"✅ Tekstrapport opgeslagen in: {report_path}")

    # =============================================================================
    # OUTPUT 2: VISUALIZATION (.png)
    # =============================================================================
    df_melted = df_results.melt(id_vars='Band', value_vars=['Single CV (Biased)', 'Nested CV (Unbiased)'], 
                                var_name='Method', value_name='Balanced Accuracy')

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(x='Band', y='Balanced Accuracy', hue='Method', data=df_melted, 
                     palette=['#2ca02c', '#ff7f0e'], edgecolor='black', alpha=0.8)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), textcoords = 'offset points', fontsize=10)

    plt.title('Riemannian Ablation: Single vs. Nested CV Hyperparameter Tuning', fontsize=14, pad=20)
    plt.ylabel('Training Balanced Accuracy', fontsize=12)
    plt.xlabel('EEG Frequency Band', fontsize=12)
    plt.ylim(0.4, max(df_melted['Balanced Accuracy']) + 0.1)
    plt.legend(title='Validation Methodology', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    SVM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = SVM_FIGURES_DIR / "Figure_Ablation_CV_Comparison_Riemann.png"
    plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figuur opgeslagen in: {plot_path}")
    print("="*70)

if __name__ == "__main__":
    run_riemann_ablation()