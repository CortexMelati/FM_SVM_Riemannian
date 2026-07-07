"""
=============================================================================
1. PREPROCESS RIEMANN (SOURCE & TARGET DOMAINS, DUAL LAYOUT)
=============================================================================
Overview:
    This unified script handles ALL covariance matrix generation.
    
    Part A (Source Domain): Enforces the exact train/test split based on the SVM 
    dataset to prevent data leakage.
    
    Part B (Target Domain): Processes the external unseen dataset (e.g., NCCP)
    to ensure it is ready for Cross-Domain Validation (TrAdaBoost).
    
    Generates two spatial layouts: Whole-Brain (19 channels) and Central ROI (9 channels).

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
from config import (PROJECT_ROOT, RESULTS_DIR, PROCESSED_DATA_DIR, SFREQ_MAP, 
                    ACTIVE_DATASET_NAME, BANDS, CHANNELS_1020, BEST_CHANNELS_EVALUATE, 
                    RIEMANN_DATA_DIR, CROSS_TARGET_DATASET)

CONDITION = 'EC'  # or EO
SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)
TARGET_SFREQ = SFREQ_MAP.get(CROSS_TARGET_DATASET, 500) 
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]

def apply_bandpass_filter(epochs_data, l_freq, h_freq, sfreq=500):
    iir_params = dict(order=4, ftype='butter', output='sos')
    return mne.filter.filter_data(
        epochs_data.astype(np.float64), 
        sfreq=sfreq, 
        l_freq=l_freq, 
        h_freq=h_freq, 
        method='iir', 
        iir_params=iir_params,
        verbose=False
    ) 

def process_source_domain():
    print("🚀 STARTING PART A: SOURCE DOMAIN PREPROCESSING (TRAIN/TEST SPLIT)")
    
    train_csv = PROCESSED_DATA_DIR / "final_dataset_train.csv"
    test_csv = PROCESSED_DATA_DIR / "final_dataset_test.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("🚨 SVM CSV files missing. Run build_dataset.py first.")
        
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    train_subjects = df_train['Subject'].unique()
    test_subjects = df_test['Subject'].unique()
    
    overlap = set(train_subjects).intersection(set(test_subjects))
    if overlap:
        raise ValueError(f"🚨 CRITICAL ERROR: Data leakage detected. Subjects in both sets: {overlap}")

    file_pattern = f"*_{CONDITION}_cleaned.npy"
    subject_files = list(RESULTS_DIR.rglob(file_pattern))
    
    X_train_list, y_train_list, group_train_list = [], [], []
    X_test_list, y_test_list, group_test_list = [], [], []

    for file_path in subject_files:
        subject_id = file_path.name.split('_')[0]
        label = 0 if ("hc" in subject_id.lower() or "control" in str(file_path).lower()) else 1
        
        try:
            data = np.load(file_path)
            if subject_id in train_subjects:
                target_macro_segments = len(df_train[df_train['Subject'] == subject_id])
                data_trunc = data[:(target_macro_segments * 30)]
                X_train_list.append(data_trunc)
                y_train_list.extend([label] * data_trunc.shape[0])
                group_train_list.extend([subject_id] * data_trunc.shape[0])
                
            elif subject_id in test_subjects:
                data_trunc = data[:150]
                X_test_list.append(data_trunc)
                y_test_list.extend([label] * data_trunc.shape[0])
                group_test_list.extend([subject_id] * data_trunc.shape[0])
        except Exception as e:
            print(f"  ⚠️ Error processing {file_path.name}: {e}")

    X_train = np.concatenate(X_train_list)
    y_train = np.array(y_train_list)
    groups_train = np.array(group_train_list)
    X_test = np.concatenate(X_test_list)
    y_test = np.array(y_test_list)
    groups_test = np.array(group_test_list)

    print(f"  📊 Train Tensor: {X_train.shape} | Test Tensor: {X_test.shape}")

    # Save base labels and RAW DATA
    np.save(RIEMANN_DATA_DIR / "X_train_raw.npy", X_train)
    np.save(RIEMANN_DATA_DIR / "X_test_raw.npy", X_test)  
    np.save(RIEMANN_DATA_DIR / "y_train_riemann.npy", y_train)
    np.save(RIEMANN_DATA_DIR / "groups_train_riemann.npy", groups_train)
    np.save(RIEMANN_DATA_DIR / "y_test_riemann.npy", y_test)
    np.save(RIEMANN_DATA_DIR / "groups_test_riemann.npy", groups_test)

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"  ⏳ Processing band: {band_name.upper()}...")
        X_tr_filt = apply_bandpass_filter(X_train, l_freq, h_freq, SFREQ)
        X_te_filt = apply_bandpass_filter(X_test, l_freq, h_freq, SFREQ)
        
        # Whole Brain
        np.save(RIEMANN_DATA_DIR / f"covs_train_{band_name}_whole.npy", Covariances(estimator='oas').transform(X_tr_filt))
        np.save(RIEMANN_DATA_DIR / f"covs_test_{band_name}_whole.npy", Covariances(estimator='oas').transform(X_te_filt))
        # ROI
        np.save(RIEMANN_DATA_DIR / f"covs_train_{band_name}_roi.npy", Covariances(estimator='oas').transform(X_tr_filt[:, ROI_INDICES, :]))
        np.save(RIEMANN_DATA_DIR / f"covs_test_{band_name}_roi.npy", Covariances(estimator='oas').transform(X_te_filt[:, ROI_INDICES, :]))

    print("✅ Source Preprocessing Complete.\n")

def process_target_domain():
    print(f"🚀 STARTING PART B: TARGET DOMAIN PREPROCESSING ({CROSS_TARGET_DATASET})")
    
    target_csv_path = PROCESSED_DATA_DIR / f"target_domain_{CROSS_TARGET_DATASET.lower()}.csv"
    if not target_csv_path.exists():
        print(f"⚠️ Target CSV niet gevonden. Cross-Domain data wordt overgeslagen.")
        return

    df_target = pd.read_csv(target_csv_path)
    if 'Condition' in df_target.columns:
        df_target = df_target[df_target['Condition'] == 'EC'].copy()

    target_subjects = df_target['Subject'].unique()
    X_target_list, y_target_list, group_target_list = [], [], []
    
    for subject_id in target_subjects:
        subject_files = list(PROJECT_ROOT.rglob(f"{subject_id}*.npy"))
        valid_files = [f for f in subject_files if 'covs' not in f.name and 'y_' not in f.name and 'groups_' not in f.name]
        
        if not valid_files: continue
        file_path = valid_files[0]
        label = df_target[df_target['Subject'] == subject_id]['Target'].values[0]
        
        try:
            data = np.load(file_path)
            target_micro_epochs = len(df_target[df_target['Subject'] == subject_id]) * 30
            data_trunc = data[:target_micro_epochs]
            
            X_target_list.append(data_trunc)
            y_target_list.extend([label] * data_trunc.shape[0])
            group_target_list.extend([subject_id] * data_trunc.shape[0])
        except Exception as e:
            pass

    if not X_target_list:
        print("🚨 FOUT: Geen target data succesvol ingeladen.")
        return

    X_target = np.concatenate(X_target_list)
    np.save(RIEMANN_DATA_DIR / f"target_y_{CROSS_TARGET_DATASET.lower()}.npy", np.array(y_target_list))
    np.save(RIEMANN_DATA_DIR / f"target_groups_{CROSS_TARGET_DATASET.lower()}.npy", np.array(group_target_list))
    np.save(RIEMANN_DATA_DIR / f"target_X_{CROSS_TARGET_DATASET.lower()}_raw.npy", X_target)
    
    print(f"  📊 Target Tensor: {X_target.shape}")

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"  ⏳ Processing target band: {band_name.upper()}...")
        X_filt = apply_bandpass_filter(X_target, l_freq, h_freq, TARGET_SFREQ)
        
        np.save(RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band_name}_whole.npy", Covariances(estimator='oas').transform(X_filt))
        np.save(RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band_name}_roi.npy", Covariances(estimator='oas').transform(X_filt[:, ROI_INDICES, :]))

    print("✅ Target Preprocessing Complete.")

if __name__ == "__main__":
    RIEMANN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    process_source_domain()
    process_target_domain()