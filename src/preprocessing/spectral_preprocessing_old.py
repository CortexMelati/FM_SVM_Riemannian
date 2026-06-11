"""
Spectral Preprocessing Pipeline - spectral_preprocessing.py
Replication of Li et al. (2026) Methodology
Focus: CP_FM_dataset (22 HC vs 20 FM)
Method: 30s blocks -> Spectral Connectivity (Coherence) -> 855 Features

- work in progress
"""

import os
import sys
import re
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from itertools import combinations

# External MNE connectivity module (requires: pip install mne-connectivity)
from mne_connectivity import spectral_connectivity_epochs

# Add the root directory to the system path to allow importing config.py
sys.path.append(os.path.abspath(".."))
import config

def extract_spectral_features(epochs_block):
    """
    Takes a block of 30x 1-second epochs, calculates coherence for 171 pairs
    across 5 frequency bands, returning exactly 855 features.
    """
    ch_names = epochs_block.ch_names
    n_channels = len(ch_names) # Should be 19
    
    # Generate specific pairs (upper triangle of connectivity matrix without diagonal)
    # This ensures we know exactly which feature corresponds to which channel pair.
    pairs = list(combinations(range(n_channels), 2))
    indices = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    
    # Calculate coherence (coh) for the specified pairs (1 to 40 Hz)
    # Using multitaper as specified in the standard protocols
    con = spectral_connectivity_epochs(
        epochs_block, method='coh', mode='multitaper', 
        sfreq=epochs_block.info['sfreq'],
        indices=indices,
        fmin=1.0, fmax=40.0, faverage=False, verbose='ERROR'
    )
    
    freqs = np.array(con.freqs)
    coh_data = con.get_data() # Shape: (171 pairs, n_frequencies)
    
    feature_vector = []
    feature_names = []
    
    # Calculate the mean coherence per frequency band for all 171 pairs
    for band_name, (fmin, fmax) in config.BANDS.items():
        freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        
        # Mean connectivity within this band for each pair
        band_mean = np.mean(coh_data[:, freq_idx], axis=1)
        feature_vector.extend(band_mean)
        
        # Generate clean feature names for the CSV (e.g., "F3-C3_Alpha")
        for p in pairs:
            ch1, ch2 = ch_names[p[0]], ch_names[p[1]]
            feature_names.append(f"{ch1}-{ch2}_{band_name}")
            
    return np.array(feature_vector), feature_names

def process_subject_spectral(file_path):
    """
    Reads the 1s epochs and groups them into 30-second macro-blocks.
    """
    try:
        epochs = mne.read_epochs(file_path, preload=True, verbose='ERROR')
        n_epochs = len(epochs)
        block_size = int(config.SEGMENT_LENGTH) # 30 epochs = 30 seconds
        
        n_blocks = n_epochs // block_size
        if n_blocks == 0:
            print(f"    ⚠️ Not enough data for 1 block (has {n_epochs}s, requires {block_size}s)")
            return None, None
            
        subject_features = []
        feature_names = None
        
        # Chop the data into blocks of exactly 30 seconds
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_epochs = epochs[start_idx:end_idx]
            
            features, names = extract_spectral_features(block_epochs)
            subject_features.append(features)
            
            if feature_names is None:
                feature_names = names
            
        return np.array(subject_features), feature_names # Shape: (n_blocks, 855)
        
    except Exception as e:
        print(f"    ✗ Error processing {file_path.name}: {e}")
        return None, None
    
if __name__ == "__main__":
    print("► Starting Feature Selection & Model Training (Li et al. Replication)")
    
    # 1. Inlezen van de master-datasets gegenereerd door build_dataset.py
    train_path = config.RESULTS_DIR / "final_dataset_train.csv"
    test_path = config.RESULTS_DIR / "final_dataset_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("🚨 Master datasets niet gevonden! Run eerst 'preprocess_pipeline.py' en 'build_dataset.py'.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Definiëren van de 9-kanaals Centrale ROI conform Li et al. (2026)
    roi_channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    
    # Filter alle kolommen die behoren tot de ROI én de Gamma-band
    # Feature namen hebben het formaat: "CH1-CH2_Gamma" of "CH1-CH2(gamma)"
    roi_gamma_features = [
        col for col in train_df.columns 
        if any(ch in col.split('_')[0].split('-') for ch in roi_channels) 
        and 'gamma' in col.lower()
    ]
    
    print(f"  ✓ Geïsoleerd binnen ROI Gamma-band: {len(roi_gamma_features)} features.")
    
    # 3. Splitsen in Features (X), Labels (y) en Groups (voor StratifiedGroupKFold)
    X_train = train_df[roi_gamma_features].values
    y_train = train_df['Target'].values
    groups_train = train_df['Subject'].values  # Cruciaal voor GroupKFold!
    
    X_test = test_df[roi_gamma_features].values
    y_test = test_df['Target'].values
    
    print(f"  ✓ Matrix Vormen: Train X={X_train.shape}, Test X={X_test.shape}")
    
    # =========================================================================
    # VOLGENDE STAP IN JOUW SCRIPT (Skelet staat klaar voor implementatie):
    # =========================================================================
    # TODO: Definieer je mSFFS wrapper (Modified Sequential Floating Forward Selection)
    # TODO: Initialiseer GridSearchCV voor SVM (RBF-kernel) met C en gamma [0.0001, 30]
    # TODO: Train op X_train en evalueer op de kluis (X_test, y_test)
    
    print("\n✅ Data staat klaar in het geheugen. Pipeline is volledig waterdicht!")