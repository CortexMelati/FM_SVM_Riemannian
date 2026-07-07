"""
=============================================================================
7. RIEMANNIAN CROSS-DOMAIN VALIDATION & TRADABOOST (UNIFIED)
=============================================================================
Overview:
    Replicates Section 2.7 and 3.4 (Table 1 & Figure 7) for the Riemannian model.
    Evaluates robustness on an external Target Domain (NCCP).
    
    Uses Smart Routing (If/Else):
    - If model is TSSVM_Cov: Loads precomputed Covariance Matrices.
    - If model is TSSVM_Xdawn: Loads Raw epochs and applies spatial filtering.
    
    Automatically generates comparative tables and figures for both architectures.

Execution:
    python 7_Riemann_cross_domain_validation.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import joblib
import mne
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from adapt.instance_based import TrAdaBoost
except ImportError:
    sys.exit("🚨 FATAL ERROR: Run 'pip install adapt' first.")

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# Panic-Proof settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import adapt only if necessary, use try-except to avoid crash
try:
    from adapt.instance_based import TrAdaBoost
    HAS_ADAPT = True
except ImportError:
    HAS_ADAPT = False

from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, CROSS_TARGET_DATASET, RANDOM_STATE

# SVM_DATA_DIR is bewust verwijderd om kruisbesmetting te voorkomen!
from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, CROSS_TARGET_DATASET, RANDOM_STATE

# =========================================================================
# BLAUWDRUKKEN: Nodig om de Xdawn pipeline uit te pakken
# =========================================================================
class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]
# =========================================================================

def run_unified_cross_domain():
    print("🚀 STARTING SCRIPT 7: UNIFIED RIEMANNIAN CROSS-DOMAIN VALIDATION")

    #amend to the best models and the best channels where needed. (in this case both models performed the same in the theta band)
    target_models = [
        # "model_riemann_Theta_roi_TSSVM_Cov.pkl",
        "model_riemann_Theta_roi_TSSVM_Xdawn.pkl"
    ]
    
    for model_name in target_models:
        model_path = RIEMANN_DATA_DIR / model_name
        if not model_path.exists():
            print(f"\n🚨 Overslaan: Model {model_name} niet gevonden in de riemann_data map!")
            print("Zorg dat je de .pkl file vanuit je oude map naar riemann_data hebt verplaatst.")
            continue
            
        arch_name = model_path.stem.split("roi_")[-1]
        print(f"\n{'='*70}\n🧠 Evaluating Cross-Domain Robustness for: {arch_name}\n{'='*70}")
            
        artifact = joblib.load(model_path)
        full_pipeline = artifact['model']
        band = artifact['band']
        layout = artifact['layout']
        
        # Knip de feature extractor (FE) en SVM los
        fe_pipeline = Pipeline(full_pipeline.steps[:-1])
        frozen_svm = full_pipeline.named_steps['svm'] 
        
        # Labels en groepen ophalen (voor Source en Target)
        y_source = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")
        y_target = np.load(RIEMANN_DATA_DIR / f"target_y_{CROSS_TARGET_DATASET.lower()}.npy")
        groups_target = np.load(RIEMANN_DATA_DIR / f"target_groups_{CROSS_TARGET_DATASET.lower()}.npy")

        # --- SMART ROUTING ---
        if 'Cov' in arch_name:
            print("-> SMART ROUTING: Loading precomputed Covariance matrices...")
            X_source_input = np.load(RIEMANN_DATA_DIR / f"covs_train_{band.lower()}_{layout.lower()}.npy")
            target_path = RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band.capitalize()}_{layout.lower()}.npy"
            if not target_path.exists():
                sys.exit(f"🚨 Target data mist: {target_path.name}")
            X_target_input = np.load(target_path)
            
        else:
            print("-> SMART ROUTING: Loading RAW epochs for xDAWN spatial filtering...")
            X_source_input = np.load(RIEMANN_DATA_DIR / "X_train_raw.npy")
            target_path = RIEMANN_DATA_DIR / f"target_X_{CROSS_TARGET_DATASET.lower()}_raw.npy"
            if not target_path.exists():
                sys.exit(f"🚨 Target data mist: {target_path.name}. Voeg de save-regel toe in script 1!")
            X_target_input = np.load(target_path)

        # 3. Projecteer de data met de bevroren pijplijn
        print("-> Projecting Source & Target to the unified Tangent Space...")
        X_source = fe_pipeline.transform(X_source_input)
        X_target = fe_pipeline.transform(X_target_input)

        # 4. TRADABOOST ITERATION (Met Dynamische Folds op basis van minority class)
        df_target_subs = pd.DataFrame({'Subject': groups_target, 'Label': y_target}).drop_duplicates()
        class_counts = df_target_subs['Label'].value_counts()
        min_class_count = class_counts.min()
        
        max_folds = 10 # min_class_count (if you have the time to calculate)
        results = []

        print(f"\n📊 Target Dataset Demographics: {class_counts.to_dict()}")
        print(f"-> Set Maximum Stratified Folds to: {max_folds}")
        print(f"\nRunning iterative testing (2 to {max_folds} Folds)...")
        print(f"{'Folds':<10} | {'Avg Train Subs':<15} | {'Transfer Acc':<15} | {'Direct Acc':<15}")
        print("-" * 65)

        for n_splits in tqdm(range(2, max_folds + 1), desc="Folds Progress", colour='green'):
            cv = StratifiedGroupKFold(n_splits=n_splits)
            transfer_scores, direct_scores, train_subs = [], [], []
            
            for train_idx, test_idx in cv.split(X_target, y_target, groups=groups_target):
                X_tgt_tr, y_tgt_tr = X_target[train_idx], y_target[train_idx]
                X_tgt_te, y_tgt_te = X_target[test_idx], y_target[test_idx]
                
                train_subs.append(len(np.unique(groups_target[train_idx])))
                
                # Method 1: DIRECT
                d_svm = SVC(C=frozen_svm.C, kernel=frozen_svm.kernel, class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000, cache_size=1000)
                d_svm.fit(X_tgt_tr, y_tgt_tr)
                pred_d = d_svm.predict(X_tgt_te)
                
                # # Method 2: TRANSFER (verbose=0 om spam te voorkomen)
                # boost = SVC(C=frozen_svm.C, kernel=frozen_svm.kernel, class_weight='balanced', probability=True, random_state=RANDOM_STATE, max_iter=2000, cache_size=1000)
                # tr = TrAdaBoost(estimator=boost, n_estimators=10, random_state=RANDOM_STATE, verbose=0)
                
                pred_tr = pred_d.copy() # Default naar Direct als Transfer faalt
                if HAS_ADAPT:
                    try:
                        boost = SVC(C=frozen_svm.C, kernel=frozen_svm.kernel, class_weight='balanced', probability=True, random_state=RANDOM_STATE, max_iter=1000, cache_size=1000)
                        tr = TrAdaBoost(estimator=boost, n_estimators=5, random_state=RANDOM_STATE, verbose=0)
                        tr.fit(X_source, y_source, Xt=X_tgt_tr, yt=y_tgt_tr)
                        pred_tr = tr.predict(X_tgt_te)
                    except:
                        pass
            

                # Ignore libsvm warning 
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tr.fit(X_source, y_source, Xt=X_tgt_tr, yt=y_tgt_tr)
                
                pred_tr = tr.predict(X_tgt_te)
                
                # Subject-Level Voting
                df_fold = pd.DataFrame({'Subject': groups_target[test_idx], 'True': y_tgt_te, 'Dir': pred_d, 'Trans': pred_tr})
                df_sub = df_fold.groupby('Subject').agg(True_L=('True', 'first'), V_Dir=('Dir', lambda x: x.mode()[0]), V_Tr=('Trans', lambda x: x.mode()[0])).reset_index()
                
                direct_scores.append(accuracy_score(df_sub['True_L'], df_sub['V_Dir']))
                transfer_scores.append(accuracy_score(df_sub['True_L'], df_sub['V_Tr']))
                
            # Schrijf de resultaten per fold strak naar de console
            tqdm.write(f"{n_splits:<10} | {np.mean(train_subs):<15.1f} | {np.mean(transfer_scores):<15.3f} | {np.mean(direct_scores):<15.3f}")
            results.append({'Total_Folds': n_splits, 'Avg_Train_Subjects': np.mean(train_subs), 'Transfer_Learning': np.mean(transfer_scores), 'Direct_Training': np.mean(direct_scores)})
        # 5. OPSLAAN & PLOTTEN
        results_df = pd.DataFrame(results)
        results_df.to_csv(RIEMANN_DATA_DIR / f"Table_1_Riemann_cross_domain_{band}_{arch_name}.csv", index=False)
        
        plt.figure(figsize=(9, 6))
        plt.scatter(results_df['Avg_Train_Subjects'], results_df['Direct_Training'], color='#5c8cbc', label='Direct Training', s=60, alpha=0.9, edgecolor='white')
        plt.scatter(results_df['Avg_Train_Subjects'], results_df['Transfer_Learning'], color='#d62728', label='Transfer Learning', s=60, alpha=0.9, edgecolor='white')

        z_dir = np.polyfit(results_df['Avg_Train_Subjects'], results_df['Direct_Training'], 1)
        plt.plot(results_df['Avg_Train_Subjects'], np.poly1d(z_dir)(results_df['Avg_Train_Subjects']), color='gray', lw=2.5, alpha=0.8)

        z_trans = np.polyfit(results_df['Avg_Train_Subjects'], results_df['Transfer_Learning'], 1)
        plt.plot(results_df['Avg_Train_Subjects'], np.poly1d(z_trans)(results_df['Avg_Train_Subjects']), color='gray', lw=2.5, alpha=0.8)

        plt.title(f"Riemannian Fig 7: Cross-domain validation on {CROSS_TARGET_DATASET}\nArchitecture: {arch_name}", fontsize=13, pad=15)
        plt.xlabel('Mean training subjects', fontsize=12)
        plt.ylabel('Mean test accuracy', fontsize=12)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(frameon=True, loc='upper left', fontsize=11)
        plt.tight_layout()
        
        fig_name = f"Figure_7_Riemann_Cross_Domain_{band}_{arch_name}.png"
        plt.savefig(RIEMANN_FIGURES_DIR / fig_name, dpi=300)
        plt.close()
        
        print(f"✅ Opgeslagen: Tabel 1 en Figuur 7 voor {arch_name}!")

    print("\n✅ SCRIPT 7 VOLLEDIG AFGEROND VOOR BEIDE MODELLEN.")

if __name__ == "__main__":
    run_unified_cross_domain()