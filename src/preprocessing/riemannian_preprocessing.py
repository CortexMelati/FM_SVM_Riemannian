"""
Riemannian Preprocessing Pipeline
Input: .vhdr files
Output: Cleaned 3D Epochs for Riemannian Geometry
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
import logging
import warnings
from pathlib import Path


# ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="The unit for channel.*has changed from NA to V.")

# Add the root directory to the system path to allow importing config.py
sys.path.append(os.path.abspath(".."))
import config

# ==========================================
# ► LOGGING SETUP
# ==========================================
log_dir = config.PROCESSED_DATA_DIR / "riemannian"
log_dir.mkdir(parents=True, exist_ok=True) # Zorg dat de map bestaat voordat we loggen

logging.basicConfig(
    filename=log_dir / 'channel_check.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='a' # 'a' voor append, zodat hij niet overschrijft bij elke run
)

def apply_riemannian_filters(raw, file_name):
    """
    Applies the standardized filtering protocol: 19 channels (10-20 system)
    and a 0.5 - 44.0 Hz bandpass filter.
    """
    # ►►► DE FIX: Vertaal de config-lijst direct naar de moderne MNE namen (T3 -> T7 etc.)
    modern_target_channels = [config.CHANNEL_RENAMING_MAP.get(ch, ch) for ch in config.CHANNELS_1020]

    # 1. Synchronize legacy channel names using config map 
    mapping = {k: v for k, v in config.CHANNEL_RENAMING_MAP.items() if k in raw.ch_names}
    if mapping:
        raw.rename_channels(mapping)
    
    # 2. Fix Channel Types (Ensure everything is EEG)
    ch_types = {ch: 'eeg' for ch in raw.ch_names if ch not in ['ECG', 'EOG', 'EMG', 'STI']}
    if ch_types:
        raw.set_channel_types(ch_types)
        
    # 3. Log missing channels (Kijk nu naar de moderne lijst!)
    present_channels = [ch for ch in raw.ch_names if ch in modern_target_channels]
    missing = [ch for ch in modern_target_channels if ch not in present_channels]
    
    if missing:
        logging.warning(f"FILE: {file_name} | MISSING CHANNELS: {len(missing)} | {missing}")
    else:
        logging.info(f"FILE: {file_name} | ALL 19 CHANNELS PRESENT")
    
    # 4. Enforce exactly 19 target electrodes using a blank 19-channel canvas
    new_data = np.zeros((19, len(raw.times)))
    
    # Map existing channels from the source file into the new 19-channel data matrix
    for i, ch in enumerate(modern_target_channels):
        if ch in raw.ch_names:
            new_data[i, :] = raw.get_data(picks=ch)[0, :]
            
    # Create the new MNE object with the filled data matrix
    info = mne.create_info(ch_names=modern_target_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
    new_raw = mne.io.RawArray(data=new_data, info=info)
            
    # Copy the 'birth date' of the original data to the new canvas
    if raw.info.get('meas_date') is not None:
        new_raw.set_meas_date(raw.info['meas_date'])
            
    # Copy annotations (markers) for datasets that use them (like FM_EO)
    new_raw.set_annotations(raw.annotations)
            
    # 5. Filter and Montage on the standardized 19-channel object
    new_raw.notch_filter(freqs=config.NOTCH_FREQ, verbose='ERROR')
    new_raw.filter(l_freq=config.FILTER_HP, h_freq=config.FILTER_LP, fir_design='firwin', verbose='ERROR')
    
    montage = mne.channels.make_standard_montage('standard_1020')
    try:
        new_raw.set_montage(montage, on_missing='ignore')
    except TypeError:
        # Fallback voor oudere MNE versies
        new_raw.set_montage(montage)
    
    return new_raw

def process_riemannian_continuous(file_path):
    """
    Processing logic for continuous datasets.
    """
    file_name = os.path.basename(file_path)
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
    raw = apply_riemannian_filters(raw, file_name)
    
    if raw.times[-1] > 10.0:
        raw.crop(tmin=10.0)
        
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose='ERROR')
    return epochs

def process_riemannian_itrs(file_path):
    """
    Processing logic for pre-epoched itRS data (FM_EO_dataset).
    """
    file_name = os.path.basename(file_path)
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
    raw = apply_riemannian_filters(raw, file_name)
    
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    
    epochs = mne.Epochs(
        raw, events=events, event_id=event_id, 
        tmin=0.0, tmax=1.348, baseline=None, preload=True, verbose='ERROR'
    )
    return epochs

def get_valid_tdbrain_subjects():
    """
    Reads the generated Excel files to create a whitelist of valid TDBRAIN subjects.
    """
    valid_ids = set()
    target_files = [
        config.TDBRAIN_DIR / "TDBRAIN_participants_HEALTHY.xlsx",
        config.TDBRAIN_DIR / "TDBRAIN_participants_CHRONIC_PAIN.xlsx",
    ]
    
    for excel_path in target_files:
        if excel_path.exists():
            try:
                df = pd.read_excel(excel_path)
                id_col = next((c for c in df.columns if 'ID' in c or 'sub' in c), 'participants_ID')
                valid_ids.update([str(x).strip() for x in df[id_col].dropna()])
            except Exception as e:
                print(f"   ✗ Error reading {excel_path.name}: {e}")
                
    return valid_ids

if __name__ == "__main__":
    print("► Starting Riemannian Preprocessing Pipeline (Input: .vhdr)")
    
    valid_tdbrain_ids = get_valid_tdbrain_subjects()
    print(f"  ✓ Loaded {len(valid_tdbrain_ids)} valid TDBRAIN target subjects.")
    
    datasets = [
        ("FM_EO_dataset", config.FM_DIR, process_riemannian_itrs), 
        ("CP_FM_dataset", config.CP_FM_DIR, process_riemannian_continuous),
        ("TDBRAIN-dataset", config.TDBRAIN_DIR, process_riemannian_continuous), 
        ("Chronicpainset", config.CHRONIC_PAIN_DIR, process_riemannian_continuous)
    ]
    
    for dataset_name, dataset_path, process_func in datasets:
        print(f"\n> Scanning dataset directory: {dataset_name}...")
        
        vhdr_files = [str(p) for p in dataset_path.rglob("*.vhdr")]
        
        if not vhdr_files:
            print(f"   ! No .vhdr files located in this directory.")
            continue
            
        print(f"   ✓ Located {len(vhdr_files)} .vhdr files. Filtering...")
        
        valid_files_to_process = []
        for f in vhdr_files:
            filename = os.path.basename(f)
            subject_id = filename.split('_')[0]
            
            if dataset_name == "TDBRAIN-dataset" and subject_id not in valid_tdbrain_ids:
                continue
            if dataset_name == "Chronicpainset" and not filename.endswith("_new.vhdr"):
                continue
            
            valid_files_to_process.append(f)
            
        # ==========================================
        # ► TEST MODE: Nu geactiveerd voor 1 file. 
        # Verwijder deze regel of zet er een # voor als je alles wilt runnen!
        # ==========================================
        valid_files_to_process = valid_files_to_process[:1] 
        # ==========================================
            
        print(f"   ► Executing pipeline on {len(valid_files_to_process)} target files...")

        for file_path in valid_files_to_process:
            subject_filename = os.path.basename(file_path)
            out_filename = os.path.splitext(subject_filename)[0] + "_riemann.fif"
            out_filepath = config.PROCESSED_DATA_DIR / "riemannian" / dataset_name / out_filename
            
            out_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if out_filepath.exists():
                print(f"      - Skipping: {subject_filename} (Output already verified)")
                continue
                
            try:
                clean_epochs = process_func(file_path)
                clean_epochs.save(out_filepath, overwrite=True, verbose='ERROR')
                print(f"      ✓ Processed & Saved: {out_filename} ({len(clean_epochs)} epochs extracted)")
            except Exception as e:
                print(f"      ✗ ERROR processing {subject_filename}: {e}")

    print("\n✓ Riemannian Preprocessing Pipeline Completed Successfully!")