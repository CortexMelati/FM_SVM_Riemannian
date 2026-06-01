"""
Master Preprocessing Script for EEG Data.
Aligned with the methodology of Li et al. (2026) and customized for Riemannian Geometry.
"""

# work in progress

import os
import sys
import glob
import mne
from pathlib import Path

# ==============================================================================
# 1. IMPORT CONFIGURATION
# ==============================================================================
# Since this script is inside /preprocessing, we step one folder up ('..') to reach root
sys.path.append(os.path.abspath(".."))
import config

def apply_standard_filtering(raw):
    """
    Applies the standardized Li et al. (2026) filtering protocol to the raw data.
    """
    # 1. Channel renaming (sync legacy names to 10-20 standard)
    valid_mapping = {
        old: new for old, new in config.CHANNEL_RENAMING_MAP.items() if old in raw.ch_names
    }
    if valid_mapping:
        raw.rename_channels(valid_mapping)
        
    # 2. Select only the 19 target electrodes
    available_channels = [ch for ch in config.CHANNELS_1020 if ch in raw.ch_names]
    raw.pick(available_channels)
    
    # 3. Apply standard 10-20 montage for 3D coordinates
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')
    
    # 4. Spectral Filtering (Notch + Bandpass)
    raw.notch_filter(freqs=config.NOTCH_FREQ, verbose='ERROR')
    raw.filter(l_freq=config.FILTER_HP, h_freq=config.FILTER_LP, fir_design='firwin', verbose='ERROR')
    
    return raw

def process_continuous_data(file_path):
    """
    Processing pipeline for continuous datasets (TDBRAIN, Chronicpainset, CP_FM_dataset).
    Includes the 10-second truncation and fixed-length epoching.
    """
    raw = mne.io.read_raw(file_path, preload=True, verbose='ERROR')
    raw = apply_standard_filtering(raw)
    
    # Crop the first 10 seconds to remove initialization noise (Li et al., 2026)
    if raw.times[-1] > config.TRUNCATE_TIME:
        raw.crop(tmin=config.TRUNCATE_TIME)
    else:
        print(f"⚠️ Warning: File {os.path.basename(file_path)} is shorter than 10 seconds.")
        
    # Segment the continuous data into 1-second epochs for Riemannian analysis
    epochs = mne.make_fixed_length_epochs(raw, duration=config.EPOCH_LENGTH, preload=True, verbose='ERROR')
    return epochs

def process_itrs_data(file_path):
    """
    Processing pipeline specifically for pre-epoched itRS data (FM_EO_dataset).
    Skips the 10s crop and directly utilizes the 1350ms task boundaries.
    """
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
    raw = apply_standard_filtering(raw)
    
    # Extract existing markers
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    
    # Epoch according to the predefined config windows (0 to 1348ms)
    epochs = mne.Epochs(
        raw, events=events, event_id=event_id, 
        tmin=config.ITRS_TMIN, tmax=config.ITRS_TMAX, 
        baseline=None, preload=True, verbose='ERROR'
    )
    return epochs

# ==============================================================================
# 2. MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}\n🚀 STARTING THESIS EEG PREPROCESSING PIPELINE\n{'='*60}")
    
    # Define which folder requires which processing strategy
    processing_map = {
        "FM_EO_dataset": process_itrs_data,
        "CP_FM_dataset": process_continuous_data,
        "TDBRAIN-dataset": process_continuous_data,
        "Chronicpainset": process_continuous_data
    }
    
    # Loop over all datasets defined in the config
    for dataset_name, dataset_path in [
        ("FM_EO_dataset", config.FM_DIR), 
        ("CP_FM_dataset", config.CP_FM_DIR),
        ("TDBRAIN-dataset", config.TDBRAIN_DIR), 
        ("Chronicpainset", config.CHRONIC_PAIN_DIR)
    ]:
        print(f"\n📁 Scanning dataset: {dataset_name}...")
        
        if not dataset_path.exists():
            print(f"   ❌ Directory not found. Skipping.")
            continue
            
        # Find all raw EEG files (.vhdr or .edf/.bdf depending on your data)
        # Using .rglob to find them even if they are nested in subfolders like derivatives/
        eeg_files = []
        for ext in ["*.vhdr", "*.edf", "*.bdf", "*.set"]:
            eeg_files.extend([str(p) for p in dataset_path.rglob(ext)])
            
        if not eeg_files:
            print(f"   ⚠️ No valid EEG files found in {dataset_name}.")
            continue
            
        print(f"   ✅ Found {len(eeg_files)} files. Starting processing...")
        
        # Select the correct processing function
        process_func = processing_map[dataset_name]
        
        for file_path in eeg_files:
            subject_filename = os.path.basename(file_path)
            
            # Create a clean output filename replacing original extension with -epo.fif
            out_filename = os.path.splitext(subject_filename)[0] + "-epo.fif"
            out_filepath = config.PROCESSED_DATA_DIR / dataset_name / out_filename
            
            # Ensure the dataset output directory exists
            out_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already processed (saves time on re-runs)
            if out_filepath.exists():
                print(f"      ⏭️ Skipping {subject_filename} (Already processed)")
                continue
                
            try:
                # Process the file
                clean_epochs = process_func(file_path)
                
                # Save the processed MNE Epochs to the results folder
                # .fif is the standard MNE format for fast loading in Machine Learning
                clean_epochs.save(out_filepath, overwrite=True, verbose='ERROR')
                print(f"      ✅ Processed & Saved: {out_filename} (Trials: {len(clean_epochs)})")
                
            except Exception as e:
                print(f"      ❌ ERROR on {subject_filename}: {str(e)}")
                
    print(f"\n{'='*60}\n🎉 PIPELINE COMPLETED! All processed data is in /results/processed_data\n{'='*60}")