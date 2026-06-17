"""
=============================================================================
2. TRAIN RIEMANN (MULTI-BAND & DUAL-LAYOUT EXPERIMENT)
=============================================================================
Overview:
    Trains TS-SVM and MDM on the covariance matrices of ALL frequency bands
    (Delta, Theta, Alpha, Beta, Gamma) and BOTH spatial layouts (Whole vs ROI)
    via Stratified Group K-Fold CV.
    
Execution:
    python 2_train_riemann.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, RANDOM_STATE, BANDS, RIEMANN_DATA_DIR

from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_curve, auc, confusion_matrix

def train_models():
    print("🚀 STARTING STEP 2: MULTI-BAND & DUAL-LAYOUT MODEL TRAINING (TS-SVM vs MDM)")
    
    y = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_train_riemann.npy")

    pipelines = {
        'TS-SVM': Pipeline([
            ('ts', TangentSpace(metric='riemann')),
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', 
                        class_weight='balanced', 
                        probability=True, 
                        random_state=RANDOM_STATE))
        ]),
        'MDM': Pipeline([
            ('mdm', MDM(metric=dict(mean='riemann', 
                                    distance='riemann')))
        ])
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    plot_data = {'roc': {}, 'cm': {}}

    # LOOP OVER SAVED BANDS
    for band_name in BANDS.keys():
        # NEW: LOOP OVER BOTH SPATIAL LAYOUTS
        for layout in ['whole', 'roi']:
            print(f"\n📡 FREQUENCY BAND: {band_name.upper()} | LAYOUT: {layout.upper()}")
            
            # Load specific covariances for this band and layout
            X_covs = np.load(RIEMANN_DATA_DIR / f"covs_train_{band_name}_{layout}.npy")
            
            for p_name, pipe in pipelines.items():
                run_name = f"{p_name} | {band_name} | {layout}"
                print(f"  ⚙️ Training: {p_name}...")
                
                y_true, y_pred, y_prob = [], [], []
                
                for train_idx, val_idx in cv.split(X_covs, y, groups):
                    pipe.fit(X_covs[train_idx], y[train_idx])
                    y_pred.extend(pipe.predict(X_covs[val_idx]))
                    
                    prob = pipe.predict_proba(X_covs[val_idx])[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_covs[val_idx])
                    y_prob.extend(prob)
                    y_true.extend(y[val_idx])

                y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

                sens = recall_score(y_true, y_pred, pos_label=1)
                spec = recall_score(y_true, y_pred, pos_label=0)
                bal_acc = (sens + spec) / 2
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                
                results.append({
                    'Band': band_name, 'Layout': layout, 'Model': p_name, 'Bal_Acc': bal_acc, 
                    'Sens (Pain)': sens, 'Spec (HC)': spec, 'ROC_AUC': roc_auc
                })
                
                plot_data['roc'][run_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
                plot_data['cm'][run_name] = confusion_matrix(y_true, y_pred)

                print(f"     -> Bal. Acc: {bal_acc:.2%} | AUC: {roc_auc:.3f}")

                # Save definitive model (THIS CONTAINS THE SVM WEIGHTS!)
                pipe.fit(X_covs, y)
                joblib.dump(pipe, PROCESSED_DATA_DIR / f"model_riemann_{band_name}_{layout}_{p_name.replace('-','')}.pkl")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RIEMANN_DATA_DIR / "riemann_multiband_layout_results.csv", index=False)
    joblib.dump(plot_data, RIEMANN_DATA_DIR / "riemann_plot_data.pkl")
    
    print("\n✅ Training complete! All bands and layouts processed.")

if __name__ == "__main__":
    train_models()