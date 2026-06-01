"""
Script to convert preprocessed MATLAB (.mat) EEG structures to BIDS-compliant 
BrainVision (.vhdr) format.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat
import pybv  # Library for exporting BrainVision files

# Import the centralized configuration to avoid hardcoding personal paths
import config

# ==============================================================================
# STEP 1: DEFINE PATHS USING CENTRAL CONFIGURATION
# ==============================================================================
# Use the dynamically resolved paths from config.py
# This ensures privacy (no "C:\Users\Jasmyne\...") and cross-platform compatibility
source_dirs = {
    "FM": os.path.join(config.FM_DIR, "Fibromyalgia"),
    "HC": os.path.join(config.FM_DIR, "Control")
}

# The central derivatives folder within the specific dataset directory
base_output_dir = os.path.join(config.FM_DIR, "derivatives")

# ==============================================================================
# STEP 2: LOOP OVER GROUPS AND FILES
# ==============================================================================
for group_label, source_path in source_dirs.items():
    print(f"\n{'='*80}\n🚀 Starting scan for the {group_label} group in:\n{source_path}\n{'='*80}")
    
    # Locate all MATLAB files in the current directory
    mat_files = sorted(glob.glob(os.path.join(source_path, "*.mat")))
    total_files = len(mat_files)
    
    if total_files == 0:
        print(f"⚠️ Warning: No .mat files found in: {source_path}")
        continue

    for index, mat_file in enumerate(mat_files, start=1):
        filename = os.path.basename(mat_file)
        
        try:
            # Extract the EEG structure from the MATLAB file
            dat = loadmat(mat_file)
            eeg_struct = dat['EEG'][0, 0]
            
            # Extract the numerical subject identifier (e.g., '06')
            digits = "".join(filter(str.isdigit, filename))
            if not digits:
                continue
            
            sub_number = int(digits)
            
            # ADOPTED STRATEGY: sub-FM001 or sub-HC001
            # Ensures BIDS compliance and prevents ID collisions in the Riemannian pipeline
            sub_id = f"sub-{group_label}{sub_number:03d}"
            
            # Construct target directory following the standard pipeline structure
            target_output_dir = os.path.join(base_output_dir, sub_id, "ses-1", "eeg")
            os.makedirs(target_output_dir, exist_ok=True)
            
            # Define file base name with task-EO (Eyes Open)
            fname_base = f"{sub_id}_task-EO_eeg"
            
            # ==============================================================================
            # STEP 3: DATA TRANSFORMATION AND EXPORT
            # ==============================================================================
            echte_data_uv = eeg_struct['data']
            n_channels, n_times, n_epochs = echte_data_uv.shape
            
            # 3D -> 2D matrix flattening to create a continuous signal for BrainVision
            data_2d_uv = np.concatenate([echte_data_uv[:, :, i] for i in range(n_epochs)], axis=1)
            data_2d_volts = data_2d_uv * 1e-6  # Convert microVolts to Volts for pybv
            
            srate = eeg_struct['srate'].item()
            chan_info = eeg_struct['chanlocs'][0]
            ch_names = [chan['labels'][0] for chan in chan_info]
            
            # Generate artificial stimulus markers to map the epoch boundaries
            events = []
            for i in range(n_epochs):
                sample_idx = i * n_times
                events.append({
                    'onset': int(sample_idx),
                    'duration': 1,
                    'description': int(i + 1),
                    'type': 'Stimulus'
                })
                
            # Export to BrainVision format (.vhdr, .vmrk, .eeg)
            pybv.write_brainvision(
                data=data_2d_volts,
                sfreq=srate,
                ch_names=ch_names,
                fname_base=fname_base,
                folder_out=target_output_dir,
                events=events,
                overwrite=True
            )
            print(f"[{group_label}] [{index}/{total_files}] Converted ➔ {sub_id}_task-EO_eeg.vhdr")
            
        except Exception as e:
            print(f"❌ ERROR processing {filename}: {str(e)}")

print("\n🏁 Conversion complete! Derivatives directory is structured with sub-FMxxx and sub-HCxxx folders.")