"""
=============================================================================
4. Final SVM Training & Hyperparameter Tuning (ALL BANDS AUTOMATED)
=============================================================================
Overview:
    This script trains the definitive SVM models for ALL available frequency
    bands using ONLY the optimal features identified by mSFFS (Script 3). 
    
    CRITICAL FIX: It strictly uses 'balanced_accuracy' and 'class_weight' 
    to prevent the model from lazily predicting the majority class in 
    imbalanced datasets. 
    
    It outputs the frozen model artifacts (.pkl) and the permutation 
    test results required for statistical significance reporting.

Execution:
    python 4_SVM_final_model_training.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib
import mne

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, permutation_test_score

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, 
                    SVM_DATA_DIR, SVM_FIGURES_DIR, BANDS)

def train_all_svm_models():
    print("🚀 STARTING STEP 4: AUTOMATED SVM TRAINING FOR ALL BANDS")

    # =============================================================================
    # 1. LOAD TRAINING DATA (Load once for efficiency)
    # =============================================================================
    print("-> Loading Training Dataset...")
    train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
    if not train_path.exists():
        sys.exit("🚨 Training dataset not found. Please run preprocessing first.")
        
    train_df = pd.read_csv(train_path)
    y_train = train_df['Target'].values
    groups_train = train_df['Subject'].values

    # =============================================================================
    # 2. LOOP OVER ALL BANDS
    # =============================================================================
    for band_name in BANDS.keys():
        band_name_lower = band_name.lower()
        print(f"\n{'='*60}\n📡 PROCESSING BAND: {band_name.upper()}\n{'='*60}")
        
        # Check if mSFFS features exist for this band
        features_path = SVM_DATA_DIR / f"final_msffs_selected_features_{band_name_lower}.csv"
        if not features_path.exists():
            print(f"⚠️ Skipping {band_name.upper()}: No mSFFS features found (Run Script 3 for this band first).")
            continue

        selected_features = pd.read_csv(features_path)['Selected_Features'].tolist()
        print(f"-> Loaded {len(selected_features)} optimal features from mSFFS.")

        X_train_final = train_df[selected_features]

        # =============================================================================
        # 3. SCALING & STRATIFIED GROUP K-FOLD
        # =============================================================================
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=selected_features)

        cv_strategy = StratifiedGroupKFold(n_splits=5)
        cv_splits = list(cv_strategy.split(X_train_scaled, y_train, groups=groups_train))

        # =============================================================================
        # 4. GRID SEARCH CV (Balanced Optimization)
        # =============================================================================
        print("-> Commencing GridSearchCV for C and gamma optimization...")

        param_grid = {
            # C can also be logspace for better tuning: np.logspace(-2, 2, 5)
            'C': [0.01, 0.1, 1, 10],
            'gamma': np.logspace(-4, 1.5, 20), 
            'class_weight': ['balanced'] 
        }

        base_svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)

        grid_search = GridSearchCV(
            estimator=base_svm,
            param_grid=param_grid,
            cv=cv_splits,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)
        best_svm = grid_search.best_estimator_

        print("\nGRID SEARCH RESULTS:")
        print(f"-> Best Parameters: {grid_search.best_params_}")
        print(f"-> Final Internal CV Balanced Accuracy: {grid_search.best_score_:.4f}")


        # cv_results_ (instead of best_params_) shows results of each split in an ndarray for a pandas dataframe
        # The key 'params' is used to store a list of parameter settings dicts for all the parameter candidates.
        # The mean_fit_time, std_fit_time, mean_score_time and std_score_time are all in seconds.
        # For multi-metric evaluation, the scores for all the scorers are available in the cv_results_ dict 
        # at the keys ending with that scorer’s name ('_<scorer_name>') instead of '_score' shown above. 
        # (‘split0_test_precision’, ‘mean_train_precision’ etc.)
        
        
        # =============================================================================
        # 5. PERMUTATION TEST
        # =============================================================================
        N_PERMUTATIONS = 1000
        print(f"\n-> Running {N_PERMUTATIONS}-iteration Permutation Test (This may take a minute)...")

        score, permutation_scores, pvalue = permutation_test_score(
            best_svm, X_train_scaled, y_train, 
            groups=groups_train, cv=cv_splits, 
            n_permutations=N_PERMUTATIONS, n_jobs=-1, random_state=RANDOM_STATE, 
            scoring='balanced_accuracy'
        )

        print("\nPERMUTATION TEST RESULTS:")
        print(f"-> True Model Score (Balanced): {score:.4f}")
        print(f"-> Mean Permuted Score: {permutation_scores.mean():.4f}")
        print(f"-> P-value: {pvalue:.4f}")
        
        if pvalue < 0.05:
            print("-> Conclusion: Model performs significantly better than chance (p < 0.05)!")
        else:
            print("-> Conclusion: Model performance is NOT statistically significant.")

        # =============================================================================
        # 6. FREEZE AND SAVE
        # =============================================================================
        model_artifact = {
            'model': best_svm,
            'scaler': scaler,
            'features': selected_features,
            'band': band_name.upper(),
            'training_accuracy': grid_search.best_score_,
            'p_value': pvalue
        }

        model_path = SVM_DATA_DIR / f"saved_model_{band_name_lower}.pkl"
        joblib.dump(model_artifact, model_path)
        print(f"\n-> Model completely frozen and saved to: svm_data/{model_path.name}")

        # =============================================================================
        # 7. PLOT BASIC BIOMARKER MAP (Binary Presence)
        # =============================================================================
        print("-> Generating Basic Biomarker Network Map...")
        montage = mne.channels.make_standard_montage('standard_1020')
        
        # Ensure we only plot the standard 19 channels to match the Riemannian visual style
        standard_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        info = mne.create_info(ch_names=standard_19, sfreq=500, ch_types='eeg')
        info.set_montage(montage)

        fig, ax = plt.subplots(figsize=(8, 8))
        mne.viz.plot_sensors(info, show_names=True, axes=ax)

        # Style the sensors to match the clean aesthetic
        for collection in ax.collections:
            collection.set_sizes([600])
            collection.set_facecolor('white')
            collection.set_edgecolor('#cccccc')
            collection.set_linewidth(1.5)

        sensor_offsets = ax.collections[0].get_offsets()
        ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

        for feat in selected_features:
            node1 = feat.split('-')[0]
            node2 = feat.split('-')[1].split('(')[0]
            try:
                x_coords = [ch_pos[node1][0], ch_pos[node2][0]]
                y_coords = [ch_pos[node1][1], ch_pos[node2][1]]
                ax.plot(x_coords, y_coords, color='#d62728', linewidth=3.5, alpha=0.9, zorder=0)
            except KeyError:
                pass 

        ax.set_title(f"mSFFS Selected Connectivity Features\n({band_name.upper()} Band)", fontsize=16, pad=20)
        plt.tight_layout()
        
        SVM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = SVM_FIGURES_DIR / f"Figure_mSFFS_Network_{band_name_lower}.png"
        plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"-> Map saved to svm_figures/{plot_path.name}")

    print(f"\n{'='*60}\n✅ ALL APPLICABLE BANDS PROCESSED SUCCESSFULLY!\n{'='*60}")

if __name__ == "__main__":
    train_all_svm_models()