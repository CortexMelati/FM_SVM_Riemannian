"""
SVM Preprocessing Pipeline - svm_preprocessing.py
Focus: CP_FM_dataset (22 HC vs 22 FM)
Method: Gamma-band -> ROI (9 channels) -> Riemannian SCM -> Tangent Space

- work in progress
"""

import os
import sys
import mne
import numpy as np
import pyriemann
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add the root directory to the system path to allow importing config.py
sys.path.append(os.path.abspath(".."))
import config


def extract_features_per_subject(file_path):
    """
    Reads 1 subject, filters for Gamma band + ROI, calculates SCMs and returns them.
    """
    # The 9 central electrodes (Region of Interest) based on Li et al. top features
    roi_channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    
    try:
        # 1. Read the clean 1s epochs
        epochs = mne.read_epochs(file_path, preload=True, verbose='ERROR')
        
        # 2. Filter specifically for the Gamma band (via config.py: 30-40 Hz)
        gamma_min, gamma_max = config.BANDS['Gamma']
        epochs.filter(l_freq=gamma_min, h_freq=gamma_max, verbose='ERROR')
        
        # 3. Isolate the 9 central electrodes
        epochs.pick(roi_channels, verbose='ERROR')
        
        # 4. Calculate Spatial Covariance Matrices (SCM) per epoch (9x9 matrix)
        cov_est = pyriemann.estimation.Covariances(estimator='oas')
        cov_mats = cov_est.fit_transform(epochs.get_data(copy=False))
        
        return cov_mats # Shape: (n_epochs, 9, 9)
        
    except Exception as e:
        print(f"    ✗ Error processing {file_path.name}: {e}")
        return None

if __name__ == "__main__":
    print("► Starting Feature Extraction for SVM (CP_FM_dataset)")
    
    # Dynamically fetch paths from config
    dataset_dir = config.PROCESSED_DATA_DIR / "CP_FM_dataset"
    fif_files = list(dataset_dir.rglob("*_riemann.fif"))
    
    print(f"  ✓ Found {len(fif_files)} processed subjects.")
    
    all_covs = []
    all_labels = []
    all_groups = [] # Integer mapping for Scikit-Learn GroupKFold
    all_subject_names = [] # String names for visual inspection in CSV
    
    for subject_idx, f in enumerate(fif_files):
        # Dynamic label assignment via config.LABEL_MAPPING
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
            
        # Extract the real subject ID from the parent folder name
        real_subject_id = f.parent.name
        
        # --- NIEUWE UITSLUITINGSREGEL ---
        # Sla sub-NCCPhc01 t/m sub-NCCPhc69 over
        if real_subject_id.startswith("sub-NCCPhc"):
            # Zoek naar de cijfers in de bestandsnaam
            num_match = re.search(r'\d+', real_subject_id)
            if num_match:
                sub_num = int(num_match.group()) # Maakt er een echt getal van (bijv. 5 of 60)
                if 1 <= sub_num <= 69:
                    print(f"  ⏭️ Skipping excluded subject: {real_subject_id}")
                    continue # Dit commando breekt de loop af en gaat door naar de volgende patiënt
        # --------------------------------
        
        print(f"  > Processing [{status}]: {real_subject_id}...")
        cov_mats = extract_features_per_subject(f)
        
        if cov_mats is not None:
            n_epochs = cov_mats.shape[0]
            all_covs.append(cov_mats)
            
            all_labels.extend([label] * n_epochs)
            all_groups.extend([subject_idx] * n_epochs) # Anti-Data-Leakage tracker!
            all_subject_names.extend([real_subject_id] * n_epochs) # For the CSV
            
    # Combine all patients
    X_cov = np.vstack(all_covs)
    y = np.array(all_labels)
    groups = np.array(all_groups)
    
    print(f"\n► Projecting {X_cov.shape[0]} total SCMs to Tangent Space...")
    
    # 5. Project the 9x9 SCMs to the flat Tangent Space for the SVM
    ts = pyriemann.tangentspace.TangentSpace(metric='riemann')
    X_tangent = ts.fit_transform(X_cov)
    
    # Save the mathematical features via config paths
    out_dir = config.RESULTS_DIR / "svm_features" / "CP_FM_dataset" 
    out_dir.mkdir(parents=True, exist_ok=True)
    
####### Train en test set
    
    # 1. Bepaal de unieke subjecten
    unique_subjects = np.unique(all_subject_names)

    # 2. Split ze (10% test, 90% train)
    train_subs, test_subs = train_test_split(unique_subjects, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    # 3. Maak maskers aan
    is_train = np.isin(all_subject_names, train_subs)
    is_test = np.isin(all_subject_names, test_subs)

    # 4. Filter je data
    X_train = X_cov[is_train] # Of X_spectral
    y_train = y[is_train]
    groups_train = groups[is_train]

    X_test = X_cov[is_test] # Of X_spectral
    y_test = y[is_test]
    groups_test = groups[is_test]

    # 5. Sla op in submappen
    base_dir = config.PROCESSED_DATA_DIR / "CP_FM_dataset"
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

####################

    # Save Train
    np.save(train_dir / "X_train.npy", X_train)
    np.save(train_dir / "y_train.npy", y_train)

    # Save Test
    np.save(test_dir / "X_test.npy", X_test)
    np.save(test_dir / "y_test.npy", y_test)

    print(f"Split complete: {len(train_subs)} training subjects, {len(test_subs)} test subjects.")
    
    
    # --- 1. Save as lightning-fast Numpy arrays for the ML Pipeline ---
    np.save(out_dir / "X_tangent.npy", X_tangent)
    np.save(out_dir / "y_labels.npy", y)
    np.save(out_dir / "groups.npy", groups)
    
    # --- 2. Save as an organized CSV file for inspection ---
    # Create column names for the features (Feature_1, Feature_2, ... Feature_45)
    roi_channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    feature_cols = []
    
    for i in range(len(roi_channels)):
        for j in range(i, len(roi_channels)):
            if i == j:
                # De diagonaal: het pure vermogen van 1 kanaal
                feature_cols.append(f"{roi_channels[i]}_Power")
            else:
                # De off-diagonaal: de connectie tussen 2 kanalen
                feature_cols.append(f"{roi_channels[i]}-{roi_channels[j]}")
    
    # Build DataFrame
    df = pd.DataFrame(X_tangent, columns=feature_cols)
    
    # Add metadata at the front (String ID, Integer Group, and Label)
    df.insert(0, "Subject_ID", all_subject_names)
    df.insert(1, "Group_Int", groups)
    df.insert(2, "Label", y)
    
    # Export to CSV
    csv_path = out_dir / "svm_features_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n✓ Succesfully saved Tangent Space features!")
    print(f"  - Numpy arrays saved for ML processing.")
    print(f"  - CSV dataset saved at: {csv_path.name}")
    print(f"  - X shape: {X_tangent.shape} (Epochs x Features)")
    print(f"  - Total unique subjects processed: {len(np.unique(groups))}")