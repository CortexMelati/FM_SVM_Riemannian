"""
=============================================================================
1. PREPROCESS RIEMANN (TRAIN/TEST INHERITANCE)
=============================================================================
Overview:
    Dwingt de exacte train/test splitsing af op basis van de SVM-dataset
    (final_dataset_train.csv en final_dataset_test.csv) om data leakage
    te voorkomen en een perfecte 'apples-to-apples' vergelijking te garanderen.
    
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

CONDITION = 'EC'  # or EO
SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)

def apply_bandpass_filter(epochs_data, l_freq, h_freq):
    # Gebruik een 4e-orde zero-phase IIR Butterworth filter voor korte epochs
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
    print("🚀 START STAP 1: STRICT TRAIN/TEST PREPROCESSING")
    
    # 1. Laad de "Source of Truth" vanuit de SVM pipeline
    train_csv = PROCESSED_DATA_DIR / "final_dataset_train.csv"
    test_csv = PROCESSED_DATA_DIR / "final_dataset_test.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("🚨 SVM CSV-bestanden ontbreken. Draai eerst build_dataset.py.")
        
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    train_subjects = df_train['Subject'].unique()
    test_subjects = df_test['Subject'].unique()
    
    # Controleer op leakage
    overlap = set(train_subjects).intersection(set(test_subjects))
    if overlap:
        raise ValueError(f"🚨 KRITIEKE FOUT: Data leakage gedetecteerd. Subjecten in beide sets: {overlap}")

    print(f"  ✓ Inherited Train Subjects: {len(train_subjects)}")
    print(f"  ✓ Inherited Test Subjects:  {len(test_subjects)}")

    # 2. Inlezen en verdelen van .npy bestanden
    file_pattern = f"*_{CONDITION}_cleaned.npy"
    subject_files = list(RESULTS_DIR.rglob(file_pattern))
    
    X_train_list, y_train_list, group_train_list = [], [], []
    X_test_list, y_test_list, group_test_list = [], [], []

    for file_path in subject_files:
        subject_id = file_path.name.split('_')[0]
        
        # Bepaal label (0 = Control, 1 = Pain)
        label = 0 if ("hc" in subject_id.lower() or "control" in str(file_path).lower()) else 1
        
        try:
            data = np.load(file_path)
            
            if subject_id in train_subjects:
                # Vermenigvuldig macro-segmenten met 30 om het juiste aantal micro-epochs te krijgen
                target_macro_segments = len(df_train[df_train['Subject'] == subject_id])
                target_micro_epochs = target_macro_segments * 30
                
                data_trunc = data[:target_micro_epochs]
                X_train_list.append(data_trunc)
                y_train_list.extend([label] * data_trunc.shape[0])
                group_train_list.extend([subject_id] * data_trunc.shape[0])
                
            elif subject_id in test_subjects:
                # Test set is strikt 5 macro-segmenten (5 * 30 = 150 micro-epochs)
                data_trunc = data[:150]
                X_test_list.append(data_trunc)
                y_test_list.extend([label] * data_trunc.shape[0])
                group_test_list.extend([subject_id] * data_trunc.shape[0])
                
        except Exception as e:
            print(f"  ⚠️ Fout bij {file_path.name}: {e}")

    X_train, y_train, groups_train = np.concatenate(X_train_list), np.array(y_train_list), np.array(group_train_list)
    X_test, y_test, groups_test = np.concatenate(X_test_list), np.array(y_test_list), np.array(group_test_list)

    print(f"\n  📊 Train Tensor: {X_train.shape} | Test Tensor: {X_test.shape}")

    # =========================================================
    # 🛡️ INGEBOUWDE VEILIGHEIDSCHECK (FAIL-SAFE)
    # =========================================================
    expected_train_epochs = len(df_train) * 30
    expected_test_epochs = len(df_test) * 30

    if X_train.shape[0] != expected_train_epochs:
        raise ValueError(f"🚨 DATA LEAKAGE / VERLIES: Train tensor heeft {X_train.shape[0]} epochs, "
                         f"maar op basis van de SVM-CSV verwachten we er {expected_train_epochs} ({len(df_train)} macro-segmenten * 30s).")
    
    if X_test.shape[0] != expected_test_epochs:
        raise ValueError(f"🚨 DATA LEAKAGE / VERLIES: Test tensor heeft {X_test.shape[0]} epochs, "
                         f"maar op basis van de SVM-CSV verwachten we er {expected_test_epochs} ({len(df_test)} macro-segmenten * 30s).")
    
    print("  ✅ Veiligheidscheck geslaagd: Datavolumes komen 100% overeen met de SVM baseline.")
    # =========================================================




    # Sla basis labels op
    np.save(PROCESSED_DATA_DIR / "y_train_riemann.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "groups_train_riemann.npy", groups_train)
    np.save(PROCESSED_DATA_DIR / "y_test_riemann.npy", y_test)
    np.save(PROCESSED_DATA_DIR / "groups_test_riemann.npy", groups_test)

    # 3. Filter en bereken covarianties per band
    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"  ⏳ Verwerken band: {band_name.upper()}...")
        
        X_tr_filt = apply_bandpass_filter(X_train, l_freq, h_freq)
        X_te_filt = apply_bandpass_filter(X_test, l_freq, h_freq)
        
        covs_train = Covariances(estimator='oas').transform(X_tr_filt)
        covs_test = Covariances(estimator='oas').transform(X_te_filt)
        
        np.save(PROCESSED_DATA_DIR / f"covs_train_{band_name}.npy", covs_train)
        np.save(PROCESSED_DATA_DIR / f"covs_test_{band_name}.npy", covs_test)

    print("\n✅ Strict train/test preprocessing voltooid.")

if __name__ == "__main__":
    preprocess_and_split()