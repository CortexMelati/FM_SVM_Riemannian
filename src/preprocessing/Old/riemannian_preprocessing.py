"""
Riemannian Preprocessing Pipeline - riemannian_preprocessing.py
Input: .vhdr files
Output: Cleaned 3D Epochs & QC Plots per Subject

- work in progress

"""

import os
import sys
import re
import mne
import numpy as np
import pandas as pd
from preprocessing.Old.preprocessing_plotting_old import get_plots
import matplotlib.pyplot as plt
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
log_dir = config.PROCESSED_DATA_DIR
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_dir / 'channel_check.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='a'
)

def apply_riemannian_filters(raw, file_name):
    """
    Applies the standardized filtering protocol: 19 channels (10-20 system)
    and a 0.5 - 44.0 Hz bandpass filter.
    """
    modern_target_channels = [config.CHANNEL_RENAMING_MAP.get(ch, ch) for ch in config.CHANNELS_1020]

    # 1. Synchronize legacy channel names using config map 
    mapping = {k: v for k, v in config.CHANNEL_RENAMING_MAP.items() if k in raw.ch_names}
    if mapping:
        raw.rename_channels(mapping)
    
    # 2. Fix Channel Types (Ensure everything is EEG)
    ch_types = {ch: 'eeg' for ch in raw.ch_names if ch not in ['ECG', 'EOG', 'EMG', 'STI']}
    if ch_types:
        raw.set_channel_types(ch_types)
        
    # 3. Log missing channels
    present_channels = [ch for ch in raw.ch_names if ch in modern_target_channels]
    missing = [ch for ch in modern_target_channels if ch not in present_channels]
    
    if missing:
        logging.warning(f"FILE: {file_name} | MISSING CHANNELS: {len(missing)} | {missing}")
    else:
        logging.info(f"FILE: {file_name} | ALL 19 CHANNELS PRESENT")
    
    # 4. Enforce exactly 19 target electrodes using a blank 19-channel canvas
    new_data = np.zeros((19, len(raw.times)))
    
    for i, ch in enumerate(modern_target_channels):
        if ch in raw.ch_names:
            new_data[i, :] = raw.get_data(picks=ch)[0, :]
            
    info = mne.create_info(ch_names=modern_target_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
    new_raw = mne.io.RawArray(data=new_data, info=info)
            
    if raw.info.get('meas_date') is not None:
        new_raw.set_meas_date(raw.info['meas_date'])
            
    new_raw.set_annotations(raw.annotations)
            
    # 5. Filter and Montage
    new_raw.notch_filter(freqs=config.NOTCH_FREQ, verbose='ERROR')
    new_raw.filter(l_freq=config.FILTER_HP, h_freq=config.FILTER_LP, fir_design='firwin', verbose='ERROR')
    
    montage = mne.channels.make_standard_montage('standard_1020')
    try:
        new_raw.set_montage(montage, on_missing='ignore')
    except TypeError:
        new_raw.set_montage(montage)
    
    return new_raw

def process_riemannian_continuous(file_path, pdf_filepath=None):
    """
    Processing logic for continuous datasets. 
    Accepts an optional pdf_filepath to save the QC plot.
    """
    file_name = os.path.basename(file_path)
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
    raw = apply_riemannian_filters(raw, file_name)
    
    if raw.times[-1] > 10.0:
        raw.crop(tmin=10.0)
        
    # ==========================================
    # PLOTTING
    # ==========================================
    if pdf_filepath:
        try:
            fig = get_plots(raw, step=f"Cleaned_{file_name}")
            fig.savefig(pdf_filepath)
            plt.close(fig) # Geheugen vrijmaken
        except Exception as e:
            print(f"      ⚠️ Plotting failed for {file_name}: {e}")
    # ==========================================
        
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose='ERROR')
    return epochs

def process_riemannian_itrs(file_path, pdf_filepath=None):
    """
    Processing logic for pre-epoched itRS data (FM_EO_dataset).
    """
    file_name = os.path.basename(file_path)
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
    raw = apply_riemannian_filters(raw, file_name)
    
    # ==========================================
    # PLOTTING
    # ==========================================
    if pdf_filepath:
        try:
            fig = get_plots(raw, step=f"Cleaned_{file_name}")
            fig.savefig(pdf_filepath)
            plt.close(fig) 
        except Exception as e:
            print(f"      ⚠️ Plotting failed for {file_name}: {e}")
    # ==========================================

    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    
    epochs = mne.Epochs(
        raw, events=events, event_id=event_id, 
        tmin=0.0, tmax=1.348, baseline=None, preload=True, verbose='ERROR'
    )
    return epochs

def get_valid_tdbrain_subjects():
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
                print(f"  ✗ Error reading {excel_path.name}: {e}")
                
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
            print(f"  ! No .vhdr files located in this directory.")
            continue
            
        print(f"  ✓ Located {len(vhdr_files)} .vhdr files. Filtering...")
        
        valid_files_to_process = []
        
        # tags from config.py
        # tags: ['FMHC', 'HC', 'HEALTHY', 'FMPA', 'PA', 'FM']
        allowed_tags = []
        for tags in config.LABEL_MAPPING.values():
            allowed_tags.extend(tags)

        for f in vhdr_files:
            filename = os.path.basename(f)
            match = re.search(r'(sub-[a-zA-Z0-9]+)', filename)
            
            if match:
                subject_id = match.group(1)
            else:
                subject_id = filename.replace('.vhdr', '')
            
            is_valid = False # Standaard mag een bestand NIET door
            
            # 1: TDBRAIN (Check via de Excel whitelists)
            if dataset_name == "TDBRAIN-dataset":
                if subject_id in valid_tdbrain_ids:
                    is_valid = True
                    
            #  2: Chronicpainset (Check of het de '_new' bestanden zijn)
            elif dataset_name == "Chronicpainset":
                if filename.endswith("_new.vhdr"):
                    is_valid = True
                    
            # 3: De andere datasets (Check via de naam-tags uit config.py)
            else:
                filename_upper = filename.upper()
                if any(tag in filename_upper for tag in allowed_tags):
                    is_valid = True
            
            # Als hij door één van de regels is gekomen, voegen we hem toe!
            if is_valid:
                valid_files_to_process.append((f, subject_id))
            
        # ==========================================
        # ► TEST on one file 
        # Comment when processing all needed files
        # ==========================================
        # valid_files_to_process = valid_files_to_process[:1] 
        # ==========================================
            
        print(f"  ► Executing pipeline on {len(valid_files_to_process)} target files...")

        for file_path, subject_id in valid_files_to_process:
            subject_filename = os.path.basename(file_path)
            
            # Results of preprocessing
            subject_dir = config.PROCESSED_DATA_DIR / dataset_name / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            # Definieer de paden voor de .fif en de .pdf IN het subject mapje
            out_filename = os.path.splitext(subject_filename)[0] + "_riemann.fif"
            out_filepath = subject_dir / out_filename
            
            pdf_filename = f"{os.path.splitext(subject_filename)[0]}.pdf"
            pdf_filepath = subject_dir / pdf_filename
            
            if out_filepath.exists():
                print(f"      - Skipping: {subject_filename} (Output already verified)")
                continue
                
            try:
                # PDF path
                clean_epochs = process_func(file_path, pdf_filepath=pdf_filepath)
                
                clean_epochs.save(out_filepath, overwrite=True, verbose='ERROR')
                print(f"      ✓ Processed & Saved: {subject_id} ({len(clean_epochs)} epochs extracted)")
            except Exception as e:
                print(f"      ✗ ERROR processing {subject_filename}: {e}")

    print("\n✓ Riemannian Preprocessing Pipeline Completed Successfully!")