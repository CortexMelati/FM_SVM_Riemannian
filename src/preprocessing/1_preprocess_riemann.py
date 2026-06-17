"""
=============================================================================
1. PREPROCESS RIEMANN (TRAIN/TEST INHERITANCE & DUAL LAYOUT)
=============================================================================
Overview:
    Enforces the exact train/test split based on the SVM dataset
    (final_dataset_train.csv and final_dataset_test.csv) to prevent data leakage 
    and guarantee a perfect 'apples-to-apples' comparison.
    
    *NEW: Now processes two spatial layouts: Whole-Brain (19 channels) and 
    Central ROI (9 channels).

Execution:
    python 1_preprocess_riemann.py
=============================================================================
"""

import numpy as np
import pandas as pd
import mne
from pyriemann.estimation import Covariances
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RESULTS_DIR, PROCESSED_DATA_DIR, SFREQ_MAP, ACTIVE_DATASET_NAME, BANDS
from config import CHANNELS_1020, BEST_CHANNELS_EVALUATE, RIEMANN_DATA_DIR


CONDITION = 'EC'  # or EO
SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]

def apply_bandpass_filter(epochs_data, l_freq, h_freq):
    # Use a 4th-order zero-phase IIR Butterworth filter for short epochs
    iir_params = dict(order=4, ftype='butter', output='sos')
    return mne.filter.filter_data(
        epochs_data.astype(np.float64), 
        sfreq=SFREQ, 
        l_freq=l_freq, 
        h_freq=h_freq, 
        method='iir', 
        iir_params=iir_params,
        verbose=False
    ) 

def preprocess_and_split():
    print("🚀 STARTING STEP 1: STRICT TRAIN/TEST PREPROCESSING (DUAL LAYOUT)")
    
    # 1. Load the "Source of Truth" from the SVM pipeline
    train_csv = PROCESSED_DATA_DIR / "final_dataset_train.csv"
    test_csv = PROCESSED_DATA_DIR / "final_dataset_test.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("🚨 SVM CSV files missing. Run build_dataset.py first.")
        
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    train_subjects = df_train['Subject'].unique()
    test_subjects = df_test['Subject'].unique()
    
    # Check for data leakage
    overlap = set(train_subjects).intersection(set(test_subjects))
    if overlap:
        raise ValueError(f"🚨 CRITICAL ERROR: Data leakage detected. Subjects in both sets: {overlap}")

    print(f"  ✓ Inherited Train Subjects: {len(train_subjects)}")
    print(f"  ✓ Inherited Test Subjects:  {len(test_subjects)}")

    # 2. Read and distribute .npy files
    file_pattern = f"*_{CONDITION}_cleaned.npy"
    subject_files = list(RESULTS_DIR.rglob(file_pattern))
    
    X_train_list, y_train_list, group_train_list = [], [], []
    X_test_list, y_test_list, group_test_list = [], [], []

    for file_path in subject_files:
        subject_id = file_path.name.split('_')[0]
        
        # Determine label (0 = Control, 1 = Pain)
        label = 0 if ("hc" in subject_id.lower() or "control" in str(file_path).lower()) else 1
        
        try:
            data = np.load(file_path)
            
            if subject_id in train_subjects:
                # Multiply macro-segments by 30 to get the correct number of micro-epochs
                target_macro_segments = len(df_train[df_train['Subject'] == subject_id])
                target_micro_epochs = target_macro_segments * 30
                
                data_trunc = data[:target_micro_epochs]
                X_train_list.append(data_trunc)
                y_train_list.extend([label] * data_trunc.shape[0])
                group_train_list.extend([subject_id] * data_trunc.shape[0])
                
            elif subject_id in test_subjects:
                # Test set is strictly 5 macro-segments (5 * 30 = 150 micro-epochs)
                data_trunc = data[:150]
                X_test_list.append(data_trunc)
                y_test_list.extend([label] * data_trunc.shape[0])
                group_test_list.extend([subject_id] * data_trunc.shape[0])
                
        except Exception as e:
            print(f"  ⚠️ Error processing {file_path.name}: {e}")

    X_train, y_train, groups_train = np.concatenate(X_train_list), np.array(y_train_list), np.array(group_train_list)
    X_test, y_test, groups_test = np.concatenate(X_test_list), np.array(y_test_list), np.array(group_test_list)

    print(f"\n  📊 Train Tensor: {X_train.shape} | Test Tensor: {X_test.shape}")

    # =========================================================
    # 🛡️ BUILT-IN SAFETY CHECK (FAIL-SAFE)
    # =========================================================
    expected_train_epochs = len(df_train) * 30
    expected_test_epochs = len(df_test) * 30

    if X_train.shape[0] != expected_train_epochs:
        raise ValueError(f"🚨 DATA LEAKAGE / LOSS: Train tensor has {X_train.shape[0]} epochs, "
                         f"but based on SVM-CSV we expect {expected_train_epochs} ({len(df_train)} macro-segments * 30s).")
    
    if X_test.shape[0] != expected_test_epochs:
        raise ValueError(f"🚨 DATA LEAKAGE / LOSS: Test tensor has {X_test.shape[0]} epochs, "
                         f"but based on SVM-CSV we expect {expected_test_epochs} ({len(df_test)} macro-segments * 30s).")
    
    print("  ✅ Safety check passed: Data volumes perfectly match the SVM baseline.")
    # =========================================================

    # Save base labels
    np.save(RIEMANN_DATA_DIR / "y_train_riemann.npy", y_train)
    np.save(RIEMANN_DATA_DIR / "groups_train_riemann.npy", groups_train)
    np.save(RIEMANN_DATA_DIR / "y_test_riemann.npy", y_test)
    np.save(RIEMANN_DATA_DIR / "groups_test_riemann.npy", groups_test)

    # 3. Filter and compute covariances per band AND spatial layout
    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"  ⏳ Processing band: {band_name.upper()}...")
        
        X_tr_filt = apply_bandpass_filter(X_train, l_freq, h_freq)
        X_te_filt = apply_bandpass_filter(X_test, l_freq, h_freq)
        
        # --- Layout 1: Whole Brain (19 Channels) ---
        covs_train_whole = Covariances(estimator='oas').transform(X_tr_filt)
        covs_test_whole = Covariances(estimator='oas').transform(X_te_filt)
        np.save(RIEMANN_DATA_DIR / f"covs_train_{band_name}_whole.npy", covs_train_whole)
        np.save(RIEMANN_DATA_DIR / f"covs_test_{band_name}_whole.npy", covs_test_whole)
        
        # --- Layout 2: Central ROI (9 Channels) ---
        covs_train_roi = Covariances(estimator='oas').transform(X_tr_filt[:, ROI_INDICES, :])
        covs_test_roi = Covariances(estimator='oas').transform(X_te_filt[:, ROI_INDICES, :])
        np.save(RIEMANN_DATA_DIR / f"covs_train_{band_name}_roi.npy", covs_train_roi)
        np.save(RIEMANN_DATA_DIR / f"covs_test_{band_name}_roi.npy", covs_test_roi)

    print("\n✅ Strict train/test preprocessing completed for both spatial layouts.")

if __name__ == "__main__":
    preprocess_and_split()