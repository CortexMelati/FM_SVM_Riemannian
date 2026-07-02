"""
=============================================================================
2. RIEMANNIAN BAND SELECTION & GRID SEARCH (Unified Pipeline)
=============================================================================
Overview:
    This script evaluates the most robust Riemannian configuration. It tests 
    all 5 canonical frequency bands across three primary architectures:
    1. MDM (Covariance)
    2. TS-SVM (Covariance)
    3. TS-SVM (Xdawn Spatial Filtering)
    4. TS-SVM (Coherence - using lagged coherence)

    For the SVM architectures, it utilizes GridSearchCV to empirically select 
    the optimal Kernel ('linear' vs 'rbf') and Regularization parameter (C).
    
    !! training takes a long time. Best version will be kept.

Execution:
    python 2_train_riemann.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
import mne
from tqdm import tqdm # Toegevoegd voor de voortgangsbalk

from pyriemann.estimation import Covariances, Coherences, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score

try:
    from pyriemann.preprocessing import NearestSPD
except ImportError:
    try:
        from pyriemann.estimation import NearestSPD
    except ImportError:
        from sklearn.base import BaseEstimator, TransformerMixin
        from pyriemann.utils.base import nearest_sym_pos_def
        class NearestSPD(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return np.array([nearest_sym_pos_def(x) for x in X])



import warnings
# Voeg dit toe bovenaan je script:
warnings.filterwarnings("ignore", message="DC and Nyquist bins are not defined*")

# Ensure central config logic is imported
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RANDOM_STATE, BANDS, RIEMANN_DATA_DIR, SVM_DATA_DIR, SFREQ_MAP, ACTIVE_DATASET_NAME, CHANNELS_1020, BEST_CHANNELS_EVALUATE

SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]

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

class AverageFrequencies(BaseEstimator, TransformerMixin):
    """Slaat de 3D Coherence output (kanalen x kanalen x frequenties) plat naar een 2D matrix"""
    def fit(self, X, y=None): return self
    def transform(self, X):
        # Als de output 4D is (epochs, channels, channels, frequencies), neem dan het gemiddelde over de frequenties
        return np.mean(X, axis=-1) if X.ndim == 4 else X


def run_comprehensive_band_selection():
    print("🚀 STARTING STEP 2: UNIFIED RIEMANNIAN TRAINING & GRID SEARCH")
    
    raw_data_path = RIEMANN_DATA_DIR / "X_train_raw.npy"
    if not raw_data_path.exists():
        sys.exit(f"🚨 Missing raw data: {raw_data_path.name}. Run Script 1.")

    X_raw = np.load(raw_data_path)
    y = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_train_riemann.npy")

    param_grid_svm = [
        {
            'svm__kernel': ['linear'], 
            'svm__C': [0.001, 0.01, 0.1, 1, 10]
        },
        {
            'svm__kernel': ['rbf'], 
            'svm__C': [0.001, 0.01, 0.1, 1, 10], 
            'svm__gamma': ['scale', 'auto']
        }
    ]
    
    # K-Folds instellen (5 outer folds)
    n_repeats = 10
    n_splits = 5
    results = []
    best_score = 0
    best_pipeline = None

    results = []
    best_score = 0
    best_pipeline = None
    best_model_name = ""

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"\n{'='*50}\n📡 FREQUENCY BAND: {band_name.upper()}\n{'='*50}")
        
        pipelines = {
            # 'MDM_Cov': Pipeline([
            #     ('filter', MNEBandPass(l_freq, h_freq, SFREQ)),
            #     ('roi', ROIExtractor(ROI_INDICES)),
            #     ('cov', Covariances(estimator='oas')),
            #     ('mdm', MDM(metric=dict(mean='riemann', distance='riemann')))
            # ]),
            'TSSVM_Cov': Pipeline([
                ('filter', MNEBandPass(l_freq, h_freq, SFREQ)),
                ('roi', ROIExtractor(ROI_INDICES)),
                ('cov', Covariances(estimator='oas')),
                ('ts', TangentSpace(metric='riemann')),
                ('scaler', StandardScaler()),
                ('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE))
            ]),
            'TSSVM_Xdawn': Pipeline([
                ('filter', MNEBandPass(l_freq, h_freq, SFREQ)),
                ('roi', ROIExtractor(ROI_INDICES)),
                ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), 
                ('ts', TangentSpace(metric='riemann')),
                ('scaler', StandardScaler()),
                ('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE))
            ]),
            # 'TSSVM_Coh': Pipeline([
            #     ('filter', MNEBandPass(l_freq, h_freq, SFREQ)),
            #     ('roi', ROIExtractor(ROI_INDICES)),
            #     ('coh', Coherences(coh='lagged')),
            #     ('avg_freq', AverageFrequencies()), # <--- DEZE NIEUWE STAP PERST HEM PLAT!
            #     ('spd', NearestSPD()),
            #     ('ts', TangentSpace(metric='riemann')),
            #     ('scaler', StandardScaler()),
            #     ('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE))
            # ])
        }

        for p_name, pipe in pipelines.items():
            print(f" ⚙️ Evaluating Architecture: {p_name}")
            fold_scores = []
            
            # De handmatige 10x10 herhalingsloop
            for r in range(n_repeats):
                # Genereer elke repeat een unieke split op basis van een nieuwe seed
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + r)
                
                for train_idx, val_idx in cv.split(X_raw, y, groups):
                    if 'SVM' in p_name:
                        # Binnen de fold zoeken we naar de beste C via een interne 3-fold split
                        cv_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + r)
                        search = GridSearchCV(pipe, param_grid_svm, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1)
                        search.fit(X_raw[train_idx], y[train_idx], groups=groups[train_idx])
                        score = search.score(X_raw[val_idx], y[val_idx])
                    else:
                        pipe.fit(X_raw[train_idx], y[train_idx])
                        score = balanced_accuracy_score(y[val_idx], pipe.predict(X_raw[val_idx]))
                    
                    fold_scores.append(score)
            
            mean_acc = np.mean(fold_scores)
            print(f"    ✅ Result 10x10 CV -> Mean Bal. Acc: {mean_acc:.4f}")
            param_log = str(search.best_params_) if 'SVM' in p_name else "N/A"
            
            results.append({'Band': band_name.upper(), 'Architecture': p_name, 'CV_Balanced_Accuracy': mean_acc, 'Optimal_Params': param_log})
            
            if mean_acc > best_score:
                best_score, best_model_name = mean_acc, f"model_riemann_{band_name}_{p_name}.pkl"
                best_pipeline = search.best_estimator_ if 'SVM' in p_name else search
                
                # Fit final model silently without progress bar
                best_pipeline.fit(X_raw, y)

    # Export & Freeze
    pd.DataFrame(results).sort_values(by='CV_Balanced_Accuracy', ascending=False).to_csv(RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv", index=False)
    
    artifact_path = SVM_DATA_DIR / best_model_name
    joblib.dump({'model': best_pipeline, 'band': best_model_name.split('_')[2], 'layout': 'roi', 'training_balanced_accuracy': best_score}, artifact_path)
    print(f"\n{'='*70}\n🏆 OVERALL WINNER: {best_model_name} (Accuracy: {best_score:.4f})\nFrozen and saved to: svm_data/{artifact_path.name}\n{'='*70}")

    log_text = (
        f"====================================================\n"
        f" RIEMANNIAN TRAINING LOG (BEST MODEL) \n"
        f"====================================================\n"
        f"Winning Architecture:  {best_model_name.replace('.pkl', '')}\n"
        f"Frequency Band:        {best_model_name.split('_')[2]}\n"
        f"Balanced Accuracy:     {best_score:.4f}\n"
        f"Optimal Parameters:    {search.best_params_ if hasattr(search, 'best_params_') else 'N/A'}\n"
        f"Feature Extraction:    Covariance / Coherence -> Tangent Space Projection\n"
        f"Cross-Validation:      5-Fold Stratified Group CV\n"
        f"====================================================\n"
    )
    
    log_path = SVM_DATA_DIR / f"riemann_training_report_{best_model_name.split('_')[2]}.txt"
    with open(log_path, "w") as f:
        f.write(log_text)
    print(f"-> Logboek succesvol opgeslagen in: svm_data/{log_path.name}")
    
    
if __name__ == "__main__":
    run_comprehensive_band_selection()