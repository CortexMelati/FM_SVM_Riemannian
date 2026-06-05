"""
Spectral Preprocessing Pipeline - spectral_preprocessing.py
Replication of Li et al. (2026) Methodology
Focus: CP_FM_dataset (22 HC vs 22 FM)
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
    print("► Starting Spectral Feature Extraction (Li et al. Replication)")
    
    # Target the processed Riemannian data (the cleaned .fif files)
    dataset_dir = config.PROCESSED_DATA_DIR / "CP_FM_dataset"
    fif_files = list(dataset_dir.rglob("*_riemann.fif"))
    
    print(f"  ✓ Found {len(fif_files)} processed subjects to extract from.")
    
    all_features = []
    all_labels = []
    all_groups = []
    all_subject_names = []
    final_feature_names = None
    
    for subject_idx, f in enumerate(fif_files):
        # Dynamic label assignment via config
        filename_upper = f.name.upper()
        if any(tag in filename_upper for tag in config.LABEL_MAPPING['Healthy_0']):
            label = 0
            status = "HC"
        elif any(tag in filename_upper for tag in config.LABEL_MAPPING['Patient_1']):
            label = 1
            status = "FM"
        else:
            print(f"  ⚠️ Skipping {f.name}: Could not assign class label.")
            continue
            
        # Smart subject ID extraction (re-using your logic)
        match = re.search(r'(sub-[a-zA-Z0-9]+)', f.name)
        if match:
            real_subject_id = match.group(1)
        else:
            real_subject_id = f.name.replace('_riemann.fif', '')
        
        if real_subject_id.startswith("sub-NCCPhc"):
            num_match = re.search(r'\d+', real_subject_id)
            if num_match:
                sub_num = int(num_match.group())
                if 1 <= sub_num <= 69:
                    print(f"  ⏭️ Skipping excluded subject: {real_subject_id}")
                    continue
        
        
        
            
        print(f"  > Processing [{status}]: {real_subject_id}...")
        
        feature_matrix, f_names = process_subject_spectral(f)
        
        if feature_matrix is not None:
            n_blocks = feature_matrix.shape[0]
            all_features.append(feature_matrix)
            
            all_labels.extend([label] * n_blocks)
            all_groups.extend([subject_idx] * n_blocks)
            all_subject_names.extend([real_subject_id] * n_blocks)
            
            if final_feature_names is None:
                final_feature_names = f_names
            
    if all_features:
        X_spectral = np.vstack(all_features)
        y = np.array(all_labels)
        groups = np.array(all_groups)
        
        out_dir = config.PROCESSED_DATA_DIR / "CP_FM_dataset"
        out_dir.mkdir(parents=True, exist_ok=True)
        
########### Train en test set
        
        unique_subjects = np.unique(all_subject_names)

        # Gebruik de variabelen uit config.py
        train_subs, test_subs = train_test_split(
            unique_subjects, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE
        )

        is_train = np.isin(all_subject_names, train_subs)
        is_test = np.isin(all_subject_names, test_subs)

        # Filter de spectrale data
        X_train = X_spectral[is_train]
        y_train = y[is_train]
        groups_train = groups[is_train]

        X_test = X_spectral[is_test]
        y_test = y[is_test]
        groups_test = groups[is_test]

        # Sla op in submappen
        train_dir = out_dir / "train"
        test_dir = out_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Save Train (met unieke spectrale namen om overschrijven te voorkomen)
        np.save(train_dir / "X_train_spectral.npy", X_train)
        np.save(train_dir / "y_train_spectral.npy", y_train)

        # Save Test
        np.save(test_dir / "X_test_spectral.npy", X_test)
        np.save(test_dir / "y_test_spectral.npy", y_test)

        print(f"\n► Split complete: {len(train_subs)} training subjects, {len(test_subs)} test subjects.")
        
####################

# --- Save de volledige data voor inspectie ---
        np.save(out_dir / "X_spectral_full.npy", X_spectral)
        np.save(out_dir / "y_labels_spectral_full.npy", y)
        np.save(out_dir / "groups_spectral_full.npy", groups)
        
        # Save CSV met exacte feature namen
        df = pd.DataFrame(X_spectral, columns=final_feature_names)
        df.insert(0, "Subject_ID", all_subject_names)
        df.insert(1, "Group_Int", groups)
        df.insert(2, "Label", y)
        
        csv_path = out_dir / "spectral_features_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n✓ Succesfully saved Spectral features (855 per block)!")
        print(f"  - Arrays saved for ML processing in /train and /test.")
        print(f"  - CSV dataset saved at: {csv_path.name}")
        print(f"  - X shape: {X_spectral.shape} (Blocks x Features)")
        print(f"  - Total unique subjects processed: {len(np.unique(groups))}")