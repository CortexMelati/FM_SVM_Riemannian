"""
=============================================================================
2. RIEMANNIAN EXPLORATION (19 CHANNELS - WHOLE BRAIN)
=============================================================================
Overview:
    Evaluates all 5 frequency bands using the Whole Brain layout.
    Saves the best model (.pkl) for each of the 5 bands.
    Optimized via Fold-Level Pre-Transformation to prevent redundant 
    Riemannian Mean computations during GridSearch. Includes performance timers.

    for r in tqdm(range(2) <- should be 10

Execution:
    python 2_riemann_whole_brain.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
import mne
import warnings
import time 
from tqdm import tqdm

from pyriemann.estimation import Covariances, Coherences, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score

# --- PyRiemann Compatibility ---
try:
    from pyriemann.preprocessing import NearestSPD
except ImportError:
    try:
        from pyriemann.estimation import NearestSPD
    except ImportError:
        from pyriemann.utils.base import nearest_sym_pos_def
        class NearestSPD(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return np.array([nearest_sym_pos_def(x) for x in X])

warnings.filterwarnings("ignore", message="DC and Nyquist bins are not defined*")

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RANDOM_STATE, BANDS, RIEMANN_DATA_DIR, SVM_DATA_DIR, SFREQ_MAP, ACTIVE_DATASET_NAME

SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)

class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X):
        return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class AverageFrequencies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.mean(X, axis=-1) if X.ndim == 4 else X

def run_whole_brain_exploration():
    total_start_time = time.time()
    print("🚀 STARTING SCRIPT 2: FULL BRAIN EXPLORATION (19 CHANNELS)")
    
    X_raw = np.load(RIEMANN_DATA_DIR / "X_train_raw.npy")
    y = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_train_riemann.npy")

    svm_param_grid = [
        {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
        {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}
    ]

    results = []

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"\n{'='*60}\n📡 ANALYZING: {band_name.upper()} BAND (WHOLE BRAIN)\n{'='*60}")
        band_start_time = time.time()
        
        X_covs = np.load(RIEMANN_DATA_DIR / f"covs_train_{band_name}_whole.npy")
        architectures = ['MDM_Cov', 'TSSVM_Cov', 'TSSVM_Xdawn', 'TSSVM_Coh']

        for p_name in architectures:
            arch_start_time = time.time()
            print(f" ⚙️ Evaluating Architecture: {p_name} ...")
            
            X_input = X_covs if 'Cov' in p_name else X_raw
            fold_scores = []
            
            if p_name == 'MDM_Cov':
                pipe_mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
                # Voeg TQDM toe aan de 10 repeats
                for r in tqdm(range(2), desc=f"   🔄 {p_name}", leave=False, colour='cyan'):
                    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r)
                    for train_idx, val_idx in cv.split(X_input, y, groups):
                        pipe_mdm.fit(X_input[train_idx], y[train_idx])
                        score = balanced_accuracy_score(y[val_idx], pipe_mdm.predict(X_input[val_idx]))
                        fold_scores.append(score)
                best_params_log = "N/A"
            
            else:
                # Voeg TQDM toe aan de 10 repeats # select range(2 to 5) to find the best model, then put it on 10 rep if you want to do it on the best model
                for r in tqdm(range(2), desc=f"   🔄 {p_name}", leave=False, colour='cyan'):
                    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r)
                    for train_idx, val_idx in cv.split(X_input, y, groups):
                        
                        if p_name == 'TSSVM_Cov':
                            fe_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                        elif p_name == 'TSSVM_Xdawn':
                            fe_steps = [('filter', MNEBandPass(l_freq, h_freq, SFREQ)), ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                        elif p_name == 'TSSVM_Coh':
                            fe_steps = [('filter', MNEBandPass(l_freq, h_freq, SFREQ)), ('coh', Coherences(coh='lagged')), ('avg_freq', AverageFrequencies()), ('spd', NearestSPD()), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                        
                        fe_pipeline = Pipeline(fe_steps)
                        
                        X_train_trans = fe_pipeline.fit_transform(X_input[train_idx], y[train_idx])
                        X_val_trans = fe_pipeline.transform(X_input[val_idx])
                        
                        cv_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + r)
                        model_svm = SVC(class_weight='balanced', random_state=RANDOM_STATE)
                        
                        search = GridSearchCV(model_svm, svm_param_grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
                        
                        inner_groups = groups[train_idx]
                        search.fit(X_train_trans, y[train_idx], groups=inner_groups)
                        
                        score = search.score(X_val_trans, y[val_idx])
                        fold_scores.append(score)
                
                best_params_log = str(search.best_params_)

            mean_acc = np.mean(fold_scores)
            arch_time = time.time() - arch_start_time
            # Gebruik \r (carriage return) om de tekst over de oude voortgangsbalk te printen voor een strakke output
            print(f"\r    ✅ Mean Bal. Acc: {mean_acc:.4f} (Completed in {arch_time:.1f}s)")
            
            results.append({
                'Band': band_name.upper(), 
                'Layout': 'WHOLE', 
                'Architecture': p_name, 
                'CV_Balanced_Accuracy': mean_acc, 
                'Optimal_Params': best_params_log
            })
            
        band_time = (time.time() - band_start_time) / 60
        print(f"⏱️ Band {band_name.upper()} finished in {band_time:.2f} minutes.")

    # =========================================================================
    # BEPAAL DE WINNAAR EN SLA OP ALS VOLLEDIGE COMPATIBELE PIPELINE
    # =========================================================================
    df_results = pd.DataFrame(results).sort_values(by='CV_Balanced_Accuracy', ascending=False)
    df_results.to_csv(RIEMANN_DATA_DIR / "scoreboard_whole_brain.csv", index=False)
    
    for band_name in BANDS.keys():
        band_rows = df_results[df_results['Band'] == band_name.upper()]
        if band_rows.empty: continue
        
        best_row = band_rows.iloc[0]
        best_arch = best_row['Architecture']
        
        print(f"-> Freezing final model for {band_name.upper()} ({best_arch})...")
        
        if best_arch == 'MDM_Cov':
            final_pipe = MDM(metric=dict(mean='riemann', distance='riemann'))
            X_final_input = np.load(RIEMANN_DATA_DIR / f"covs_train_{band_name}_whole.npy")
        else:
            import ast
            p_dict = ast.literal_eval(best_row['Optimal_Params'])
            
            if best_arch == 'TSSVM_Cov':
                final_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                X_final_input = np.load(RIEMANN_DATA_DIR / f"covs_train_{band_name}_whole.npy")
            elif best_arch == 'TSSVM_Xdawn':
                final_steps = [('filter', MNEBandPass(BANDS[band_name][0], BANDS[band_name][1], SFREQ)), ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                X_final_input = X_raw
            elif best_arch == 'TSSVM_Coh':
                final_steps = [('filter', MNEBandPass(BANDS[band_name][0], BANDS[band_name][1], SFREQ)), ('coh', Coherences(coh='lagged')), ('avg_freq', AverageFrequencies()), ('spd', NearestSPD()), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                X_final_input = X_raw
                
            final_steps.append(('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE, **p_dict)))
            final_pipe = Pipeline(final_steps)
            
        final_pipe.fit(X_final_input, y)
        best_name = f"model_riemann_{band_name}_whole_{best_arch}.pkl"
        joblib.dump({'model': final_pipe, 'band': band_name, 'layout': 'whole', 'training_balanced_accuracy': best_row['CV_Balanced_Accuracy']}, SVM_DATA_DIR / best_name)

    total_time = (time.time() - total_start_time) / 60
    print(f"\n✅ Script 2 Complete! Total Execution Time: {total_time:.2f} minutes.")
    print("Whole Brain scoreboard and optimized models saved.")

if __name__ == "__main__":
    run_whole_brain_exploration()