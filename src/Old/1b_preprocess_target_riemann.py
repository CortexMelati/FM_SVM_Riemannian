"""
=============================================================================
1b. PREPROCESS TARGET DOMAIN (RIEMANNIAN CROSS-DOMAIN)
=============================================================================
Overview:
    This script acts as a bridge. It takes the external target dataset 
    (e.g., NCCP) and processes its subject-level .npy epoch files into 
    Covariance matrices, using the exact same filtering parameters as the 
    Source Domain.
    
    It uses your target_domain_nccp.csv as a strict blueprint to ensure 
    perfect alignment with the SVM cross-domain validation.

Execution:
    python 1b_preprocess_target_riemann.py
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
from config import (PROJECT_ROOT, PROCESSED_DATA_DIR, RIEMANN_DATA_DIR, 
                    CROSS_TARGET_DATASET, BANDS, CHANNELS_1020, 
                    BEST_CHANNELS_EVALUATE, SFREQ_MAP)

# Fallback sampling frequency if not found in map
TARGET_SFREQ = SFREQ_MAP.get(CROSS_TARGET_DATASET, 500) 
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]

def apply_bandpass_filter(epochs_data, l_freq, h_freq):
    iir_params = dict(order=4, ftype='butter', output='sos')
    return mne.filter.filter_data(
        epochs_data.astype(np.float64), 
        sfreq=TARGET_SFREQ, 
        l_freq=l_freq, 
        h_freq=h_freq, 
        method='iir', 
        iir_params=iir_params,
        verbose=False
    )

def preprocess_target_domain():
    print(f"🚀 STARTING STEP 1B: TARGET DOMAIN PREPROCESSING ({CROSS_TARGET_DATASET})")

    # 1. LOAD THE BLUEPRINT (SVM CSV)
    target_csv_path = PROCESSED_DATA_DIR / f"target_domain_{CROSS_TARGET_DATASET.lower()}.csv"
    if not target_csv_path.exists():
        print(f"🚨 FOUT: Blauwdruk CSV niet gevonden op:\n{target_csv_path}")
        sys.exit()

    df_target = pd.read_csv(target_csv_path)
    if 'Condition' in df_target.columns:
        df_target = df_target[df_target['Condition'] == 'EC'].copy()

    target_subjects = df_target['Subject'].unique()
    print(f" -> Found {len(target_subjects)} unique subjects in target domain CSV.")

    # 2. LOCATE .NPY FILES FOR THESE SUBJECTS
    # We zoeken in de hele 'results' map naar .npy bestanden die matchen met deze subjects
    # We gaan er vanuit dat ze eindigen op _cleaned.npy of iets vergelijkbaars.
    X_target_list, y_target_list, group_target_list = [], [], []
    
    for subject_id in target_subjects:
        # Zoek het .npy bestand voor deze specifieke proefpersoon in de target dataset map
        subject_files = list(PROJECT_ROOT.rglob(f"{subject_id}*.npy"))
        
        # Filter bestanden die waarschijnlijk de ruwe epochs bevatten (geen covs of y_ arrays)
        valid_files = [f for f in subject_files if 'covs' not in f.name and 'y_' not in f.name and 'groups_' not in f.name]
        
        if not valid_files:
            print(f" ⚠️ Waarschuwing: Geen .npy bestand gevonden voor subject {subject_id}")
            continue
            
        file_path = valid_files[0] # Pak het eerste geldige bestand
        
        # Bepaal label uit de blueprint CSV (zodat we 100% zeker dezelfde labels gebruiken als SVM)
        label = df_target[df_target['Subject'] == subject_id]['Target'].values[0]
        
        try:
            data = np.load(file_path)
            
            # Match the amount of epochs to the SVM macro-segments (1 macro = 30 micro)
            target_macro_segments = len(df_target[df_target['Subject'] == subject_id])
            target_micro_epochs = target_macro_segments * 30
            
            # Truncate to ensure exact mathematical alignment
            data_trunc = data[:target_micro_epochs]
            
            X_target_list.append(data_trunc)
            y_target_list.extend([label] * data_trunc.shape[0])
            group_target_list.extend([subject_id] * data_trunc.shape[0])
            
        except Exception as e:
            print(f" ⚠️ Fout bij inladen van {file_path.name}: {e}")

    if not X_target_list:
        print("🚨 FOUT: Geen target data succesvol ingeladen.")
        sys.exit()

    X_target = np.concatenate(X_target_list)
    y_target = np.array(y_target_list)
    groups_target = np.array(group_target_list)

    print(f"\n -> Target Tensor Compiled: {X_target.shape} (Epochs x Channels x Time)")

    # Save the base arrays
    np.save(RIEMANN_DATA_DIR / f"target_y_{CROSS_TARGET_DATASET.lower()}.npy", y_target)
    np.save(RIEMANN_DATA_DIR / f"target_groups_{CROSS_TARGET_DATASET.lower()}.npy", groups_target)

    # 3. FILTER & COMPUTE COVARIANCES PER BAND & LAYOUT
    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f" ⏳ Berekenen van covarianties voor: {band_name.upper()}...")
        
        X_filt = apply_bandpass_filter(X_target, l_freq, h_freq)
        
        # --- Layout 1: Whole Brain (19 Channels) ---
        covs_whole = Covariances(estimator='oas').transform(X_filt)
        np.save(RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band_name}_whole.npy", covs_whole)
        
        # --- Layout 2: Central ROI (9 Channels) ---
        covs_roi = Covariances(estimator='oas').transform(X_filt[:, ROI_INDICES, :])
        np.save(RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band_name}_roi.npy", covs_roi)

    print(f"\n✅ Target pre-processing compleet! De bestanden staan klaar in {RIEMANN_DATA_DIR.name}/")

if __name__ == "__main__":
    preprocess_target_domain()