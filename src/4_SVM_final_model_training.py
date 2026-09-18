"""
=============================================================================
4. Final SVM Training & Hyperparameter Tuning (ALL BANDS AUTOMATED)
=============================================================================
Overview:
    This script trains the definitive SVM models for ALL available frequency
    bands using ONLY the optimal features identified by mSFFS (Script 3). 

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
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, permutation_test_score

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, 
                    SVM_DATA_DIR, SVM_FIGURES_DIR, BANDS, SAVED_MODELS_DIR)

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
        print(f"\n{'='*60}\n PROCESSING BAND: {band_name.upper()}\n{'='*60}")
        
        # Check if mSFFS features exist for this band
        features_path = SVM_DATA_DIR / f"final_msffs_selected_features_{band_name_lower}.csv"
        if not features_path.exists():
            print(f" Skipping {band_name.upper()}: No mSFFS features found (Run Script 3 for this band first).")
            continue

        selected_features = pd.read_csv(features_path)['Selected_Features'].tolist()
        print(f"-> Loaded {len(selected_features)} optimal features from mSFFS.")

        X_train_final = train_df[selected_features]

        # =============================================================================
        # 3. SCALING & STRATIFIED GROUP K-FOLD 10 repeats van 5-fold
        # =============================================================================
        
        cv_splits = []
        for seed_offset in range(10): 
            cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + seed_offset)
            cv_splits.extend(list(cv_strategy.split(X_train_final, y_train, groups=groups_train)))
            
        print(f"-> Created {len(cv_splits)} cross-validation folds (10 repeats of 5-fold CV).")

        # =============================================================================
        # 4. GRID SEARCH CV (Balanced Optimization)
        # =============================================================================
        print("-> Commencing GridSearchCV for C and gamma optimization...")

        # Maak een pipeline: schalen gebeurt nu pas BINNEN de CV-fold
        pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', 
                probability=True, 
                random_state=RANDOM_STATE)
        )

        param_grid = {
            'svc__C': [0.01, 0.1, 1, 10],
            'svc__gamma': np.logspace(-4, 1.5, 20), 
            'svc__class_weight': ['balanced'] 
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_splits,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Train op de ONGESCHAALDE data
        grid_search.fit(X_train_final, y_train)
        best_pipeline = grid_search.best_estimator_

        print("\nGRID SEARCH RESULTS:")
        clean_params = {k.replace('svc__', ''): v for k, v in grid_search.best_params_.items()}
        print(f"-> Best Parameters: {clean_params}")
        print(f"-> Final Internal CV Balanced Accuracy: {grid_search.best_score_:.4f}")

        # cv_results_ (instead of best_params_) shows results of each split in an ndarray for a pandas dataframe
        # The key 'params' is used to store a list of parameter settings dicts for all the parameter candidates.
        # The mean_fit_time, std_fit_time, mean_score_time and std_score_time are all in seconds.
        # For multi-metric evaluation, the scores for all the scorers are available in the cv_results_ dict 
        # at the keys ending with that scorer’s name ('_<scorer_name>') instead of '_score' shown above. 
        # (‘split0_test_precision’, ‘mean_train_precision’ etc.)
        
        

        # =============================================================================
        # 6. FREEZE AND SAVE
        # =============================================================================
        best_idx = grid_search.best_index_
        std_score = grid_search.cv_results_['std_test_score'][best_idx]

        model_artifact = {
            'pipeline': best_pipeline, 
            'features': selected_features,
            'band': band_name.upper(),
            'training_accuracy': grid_search.best_score_,
            'training_std': std_score 
        }

        model_path = SAVED_MODELS_DIR / f"saved_model_{band_name_lower}.pkl"
        joblib.dump(model_artifact, model_path)
        print(f"\n-> Model strictly frozen and saved to: saved_models/{model_path.name}")

    print(f"\n{'='*60}\n✅ ALL APPLICABLE BANDS PROCESSED SUCCESSFULLY!\n{'='*60}")

if __name__ == "__main__":
    train_all_svm_models()