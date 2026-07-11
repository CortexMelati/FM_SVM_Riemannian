"""
=============================================================================
3. RIEMANNIAN ABLATION (ROI / WHOLE BRAIN SWITCH)
=============================================================================
Overview:
    Automatically reads the scoreboard from Script 2, identifies the Top 2 
    performing frequency bands, and tests them using either the 9-channel ROI 
    layout or the 19-channel WHOLE brain layout based on the toggle.
    Crucially: Freezes and saves ALL evaluated models (not just the winner) 
    so they can be compared head-to-head in Script 5.

Execution:
    python 3_riemann_roi_ablation.py
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

from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", message="DC and Nyquist bins are not defined*")

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RANDOM_STATE, BANDS, BEST_BANDS, RIEMANN_DATA_DIR, SFREQ_MAP, ACTIVE_DATASET_NAME, CHANNELS_1020, BEST_CHANNELS_EVALUATE

# =============================================================================
# TOGGLE: TRUE = 19 Channels (WHOLE) | FALSE = 9 Channels (ROI)
# =============================================================================
RUN_AS_WHOLE_BRAIN = True
# =============================================================================

SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]
LAYOUT_NAME = 'WHOLE' if RUN_AS_WHOLE_BRAIN else 'ROI'

class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]

def run_ablation():
    total_start_time = time.time()
    
    channel_count = 19 if RUN_AS_WHOLE_BRAIN else 9
    print(f"STARTING SCRIPT 3: ABLATION ({channel_count} CHANNELS - {LAYOUT_NAME})")
    
    comprehensive_path = RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv"
    if comprehensive_path.exists():
        df_existing = pd.read_csv(comprehensive_path)
    else:
        df_existing = pd.DataFrame()
    
    best_bands = BEST_BANDS 
    bands_str = ", ".join([b.upper() for b in best_bands])
    print(f"Running {LAYOUT_NAME} Ablation on the following bands: {bands_str}.")

    X_raw = np.load(RIEMANN_DATA_DIR / "X_train_raw.npy")
    y = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_train_riemann.npy")
    
    svm_param_grid = [
        {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
        {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}
    ]

    results = []
    architectures = ['TSSVM_Cov', 'TSSVM_Xdawn']

    for band_name in best_bands:
        l_freq, h_freq = BANDS[band_name]
        print(f"\n{'='*60}\n ANALYZING: {band_name.upper()} BAND ({LAYOUT_NAME})\n{'='*60}")
        
        # DE FIX: Dit zoekt nu netjes naar _whole.npy of _roi.npy
        cov_file = f"covs_train_{band_name}_whole.npy" if RUN_AS_WHOLE_BRAIN else f"covs_train_{band_name}_roi.npy"
        X_covs = np.load(RIEMANN_DATA_DIR / cov_file)

        for p_name in architectures:
            arch_start_time = time.time()
            print(f"Evaluating Architecture: {p_name} ...")
            
            X_input = X_covs if p_name == 'TSSVM_Cov' else X_raw
            fold_scores = []
            
            for r in tqdm(range(10), desc=f"    {p_name} (10 Repeats)", leave=False, colour='cyan'):
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r)
                for train_idx, val_idx in cv.split(X_input, y, groups):
                    
                    if p_name == 'TSSVM_Cov':
                        fe_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                    elif p_name == 'TSSVM_Xdawn':
                        fe_steps = [('filter', MNEBandPass(l_freq, h_freq, SFREQ))]
                        # Only add ROIExtractor when in ROI mode
                        if not RUN_AS_WHOLE_BRAIN:
                            fe_steps.append(('roi', ROIExtractor(ROI_INDICES)))
                        
                        fe_steps.extend([
                            ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), 
                            ('ts', TangentSpace(metric='riemann')), 
                            ('scaler', StandardScaler())
                        ])
                    
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
            print(f"\r    Mean Bal. Acc: {mean_acc:.4f} (Completed in {arch_time:.1f}s)")
            
            results.append({
                'Band': band_name.upper(), 
                'Layout': LAYOUT_NAME, 
                'Architecture': p_name, 
                'CV_Balanced_Accuracy': mean_acc, 
                'Optimal_Params': best_params_log
            })

    # =========================================================================
    # Save and make report
    # =========================================================================
    df_new_run = pd.DataFrame(results)
    
    report_text = "====================================================\n"
    report_text += f" FINAL {LAYOUT_NAME} ABLATION RESULTS \n"
    report_text += "====================================================\n\n"
    
    for band_name in best_bands:
        band_rows = df_new_run[df_new_run['Band'] == band_name.upper()].sort_values(by='CV_Balanced_Accuracy', ascending=False)
        
        best_row = band_rows.iloc[0]
        report_text += f"🏆 WINNER: {band_name.upper()} BAND ({LAYOUT_NAME})\n"
        report_text += f"Architecture:      {best_row['Architecture']}\n"
        report_text += f"Balanced Accuracy: {best_row['CV_Balanced_Accuracy']:.4f}\n"
        report_text += f"Optimal Params:    {best_row['Optimal_Params']}\n"
        report_text += "-"*52 + "\n"
        
        # Freezing the models
        for _, row in band_rows.iterrows():
            arch = row['Architecture']
            print(f"-> Freezing {LAYOUT_NAME} model for {band_name.upper()} ({arch})...")
            
            import ast
            p_dict = ast.literal_eval(row['Optimal_Params'])
            
            if arch == 'TSSVM_Cov':
                final_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                X_final_input = np.load(RIEMANN_DATA_DIR / cov_file)
            elif arch == 'TSSVM_Xdawn':
                final_steps = [('filter', MNEBandPass(BANDS[band_name][0], BANDS[band_name][1], SFREQ))]
                if not RUN_AS_WHOLE_BRAIN:
                    final_steps.append(('roi', ROIExtractor(ROI_INDICES)))
                    
                final_steps.extend([
                    ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), 
                    ('ts', TangentSpace(metric='riemann')), 
                    ('scaler', StandardScaler())
                ])
                X_final_input = X_raw
                
            final_steps.append(('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE, **p_dict)))
            final_pipe = Pipeline(final_steps)
            
            final_pipe.fit(X_final_input, y)
            
            layout_str = 'whole' if RUN_AS_WHOLE_BRAIN else 'roi'
            best_name = f"model_riemann_{band_name}_{layout_str}_{arch}.pkl"
            
            joblib.dump({'model': final_pipe, 'band': band_name, 'layout': layout_str, 'training_balanced_accuracy': row['CV_Balanced_Accuracy']}, RIEMANN_DATA_DIR / best_name)

    # add to earlier runs
    if not df_existing.empty:
        df_existing = df_existing[~((df_existing['Layout'] == LAYOUT_NAME) & (df_existing['Band'].isin([b.upper() for b in best_bands])))]
        df_final = pd.concat([df_existing, df_new_run]).sort_values(by=['Band', 'CV_Balanced_Accuracy'], ascending=[True, False])
    else:
        df_final = df_new_run.sort_values(by=['Band', 'CV_Balanced_Accuracy'], ascending=[True, False])

    df_final.to_csv(comprehensive_path, index=False)
    
    total_time = (time.time() - total_start_time) / 60
    print(f"\n{report_text}")
    print(f"Script 3 Complete! Total Execution Time: {total_time:.2f} minutes.")
    print("-> Final Scoreboard updated and ALL evaluated models frozen.")

if __name__ == "__main__":
    run_ablation()