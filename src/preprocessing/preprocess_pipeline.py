"""
=============================================================================
🧠 EEG PREPROCESSING PIPELINE (Li et al., 2026)
=============================================================================
Overview:
    This script performs the preprocessing pipeline based strictly on the 
    methodology described by Li et al. (2026). It bypasses manual cleaning,
    extracts raw epochs, and computes functional spectral connectivity.

Key Steps based on the paper:
    1. SCALING:    Detects if data is in Volts and rescales to Microvolts (uV).
    2. RENAMING:   Standardizes 19 channels to the 10-20 system.
    3. CROPPING:   Discards the first 10 seconds of the recording.
    4. FILTERING:  Notch filter at 50 Hz, Bandpass 0.5 - 44 Hz.
    5. EPOCHING:   Segments the data into 1-second epochs.
    6. FEATURES:   Extracts mean Coherence (connectivity) across 5 frequency bands
                   for all 171 channel pairs, resulting in 855 features.

Outputs (per subject):
    1. .npy files -> 3D epoched data (Epochs x Channels x Time)
    2. .csv files -> 2D Features (Connectivity metrics for SVM)
    3. .pdf files -> Visual Quality Control reports
    4. .txt files -> Logs containing preprocessing statistics

Execution:
    python ./FM_SVM/src/Preprocessing/preprocess_pipeline.py
=============================================================================
"""

import mne
import numpy as np
import pandas as pd
import os
import glob
import random
from mne_connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import warnings
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import (
    RESULTS_DIR, 
    CP_FM_DIR, 
    CHANNELS_1020, 
    BANDS, 
    SFREQ_MAP, 
    EPOCH_LENGTH,
    FILTER_HP,
    FILTER_LP,
    NOTCH_FREQ,
    CHANNEL_RENAMING_MAP
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from preprocessing.preprocessing_plotting_old import get_plots
except ImportError:
    print("⚠️ Warning: 'preprocessing_plotting.py' not found. Plots will be skipped.")
    def get_plots(*args, **kwargs): return None

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================
OUTPUT_DIR = RESULTS_DIR

NUM_SUBJECTS_TO_PROCESS = 1 #change to None for all files or nr. of files to process

# We focus purely on the CP_FM_dataset for this script, filtering for fmpa/fmhc later.
DATASETS = [
    (str(CP_FM_DIR), "*.vhdr", "cp_fm_dataset")
]

COMMON_CHANNELS = CHANNELS_1020
FREQ_BANDS = BANDS

warnings.filterwarnings("ignore") 

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_condition(filename):
    fname_upper = filename.upper()
    if 'EC' in fname_upper or 'CLOSED' in fname_upper: return 'EC'
    elif 'EO' in fname_upper or 'OPEN' in fname_upper: return 'EO'
    else: return 'unknown'

def is_target_subject(filename):
    """Filters specifically for Fibromyalgia patients (fmpa) and Healthy Controls (fmhc)."""
    fname_lower = filename.lower()
    return 'fmpa' in fname_lower or 'fmhc' in fname_lower

def smart_rename_channels(raw):
    """
    Standardizes channel names to match the modern names in COMMON_CHANNELS.
    Uses CHANNEL_RENAMING_MAP from config.py to convert legacy (T3) to modern (T7).
    """
    current_names = raw.ch_names
    mapping = {}
    
    target_map = {ch.lower(): ch for ch in COMMON_CHANNELS}
    
    # Gebruik de originele map (niet omgedraaid): 't3' -> 'T7'
    forward_map = {k.lower(): v for k, v in CHANNEL_RENAMING_MAP.items()}
    
    for ch in current_names:
        clean_ch = ch.replace('EEG', '').replace('Ref', '').replace(' ', '').replace('-', '').replace('.', '').lower()
        
        # 1. Map oude namen (zoals 't3') naar moderne namen (zoals 'T7')
        if clean_ch in forward_map:
            mapping[ch] = forward_map[clean_ch]
            
        # 2. Fix puur de hoofdletters voor de overige kanalen
        elif clean_ch in target_map:
            std = target_map[clean_ch]
            if ch != std: 
                mapping[ch] = std
                
    if mapping:
        try: 
            raw.rename_channels(mapping)
        except Exception as e: 
            print(f"⚠️ Renaming failed: {e}")
            
    return raw

def fix_scaling_and_units(raw):
    data_sample = raw.get_data(start=0, stop=int(10*raw.info['sfreq']))
    mean_amp = np.mean(np.abs(data_sample))
    if mean_amp > 0.1: 
        raw.apply_function(lambda x: x * 1e-6, channel_wise=True)
    raw.apply_function(lambda x: x - np.mean(x), channel_wise=True)
    return raw

def load_raw_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(filepath, misc='auto', preload=True, verbose=False)
    else:
        raise ValueError(f"Unknown format: {ext}")
    
    try: raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if ch not in ['ECG', 'EOG']})
    except: pass
    
    raw = smart_rename_channels(raw)
    raw = fix_scaling_and_units(raw)
    return raw

def extract_connectivity_features(epochs, subject_id, condition):
    """
    Computes Spectral Connectivity (Coherence) using the method from Li et al. (2026).
    Returns a dataframe containing 855 features (171 pairs * 5 bands).
    """
    band_names = list(FREQ_BANDS.keys())
    ch_names = epochs.ch_names
    n_channels = len(ch_names)
    
    features = {'Subject': subject_id, 'Condition': condition}
    
    # Iterate over each frequency band separately to ensure robust computation
    for band_name in band_names:
        fmin, fmax = FREQ_BANDS[band_name]
        
        # Compute Coherence for this specific band
        # Using sfreq from the epochs to ensure proper scaling
        con = spectral_connectivity_epochs(
            epochs, 
            method='coh', 
            mode='multitaper',
            fmin=fmin, 
            fmax=fmax, 
            faverage=True,  # Average the coherence within this specific frequency band
            verbose=False
        )
        
        # Dense output gives a matrix of shape (n_channels, n_channels, 1) because faverage=True
        con_dense = con.get_data(output='dense')
        
        # Extract the upper triangle (171 pairs for 19 channels)
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Format exactly as in paper, e.g., "Fz-Cz(gamma)"
                ch_pair = f"{ch_names[i]}-{ch_names[j]}({band_name.lower()})"
                # con_dense has shape (n_channels, n_channels, 1) -> we take [i, j, 0]
                features[ch_pair] = con_dense[i, j, 0]
                
    return pd.DataFrame([features])

def process_subject(file_path, output_dir, dataset_name):
    filename = os.path.basename(file_path)
    
    # 1. Dataset & Subject Filtering
    if not is_target_subject(filename):
        return False, "Skipped (Not fmpa or fmhc target)"
        
    condition = get_condition(filename)
    if condition != 'EC':
        return False, "Skipped (Not Eyes-Closed resting state)"
        
    subject_id = filename.split('_')[0] 
    sub_folder = os.path.join('CP_FM', 'fmpa' if 'fmpa' in subject_id.lower() else 'fmhc')

    save_dir = os.path.join(output_dir, sub_folder, subject_id)
    os.makedirs(save_dir, exist_ok=True)
    
    csv_check = os.path.join(save_dir, f"{subject_id}_{condition}_features.csv")
    pdf_check = os.path.join(save_dir, f"{subject_id}_{condition}_report.pdf")
    
    if os.path.exists(csv_check) and os.path.exists(pdf_check):
       return True, "Skipped (Already fully processed)"

    try:
        # 2. LOAD DATA
        raw = load_raw_data(file_path)

        # 3. CHANNEL SELECTION
        available = raw.ch_names
        missing = [ch for ch in COMMON_CHANNELS if ch not in available]
        if missing: return False, f"Missing core channels: {missing}"
        
        raw.pick_channels(COMMON_CHANNELS)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # 4. PREPROCESSING (Li et al., 2026 guidelines)
        if raw.times[-1] <= 10.0: return False, "Recording too short (<= 10s)"
        
        # Plot before cropping
        try: fig_before = get_plots(raw, step="1. Raw (Pre-crop)", scalings={'eeg': 100e-6}, channel_idx=[9])
        except: fig_before = None

        # Discard first 10s
        raw.crop(tmin=10.0, tmax=None)
        
        # Apply filters
        raw.notch_filter(NOTCH_FREQ, verbose=False) 
        raw.filter(l_freq=FILTER_HP, h_freq=FILTER_LP, verbose=False)
        
        # Resample if needed
        sfreq = SFREQ_MAP.get(dataset_name, 500)
        if raw.info['sfreq'] != sfreq: raw.resample(sfreq, verbose=False)

        # 5. EPOCHING (1-second blocks)
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_LENGTH, overlap=0, preload=True, verbose=False)
        if len(epochs) < 10: return False, "Too few epochs generated"

        # 6. FEATURE EXTRACTION (Spectral Connectivity)
        df_features = extract_connectivity_features(epochs, subject_id, condition)
        df_features.to_csv(csv_check, index=False)

        # 7. REPORTING
        try:
            data_tmp = epochs.get_data(copy=True)
            raw_epoched = mne.io.RawArray(np.hstack(data_tmp), epochs.info, verbose=False)
            fig_after = get_plots(raw_epoched, step="2. Filtered & Epoched", scalings={'eeg': 40e-6}, channel_idx=[9])
        except: fig_after = None

        with PdfPages(pdf_check) as pdf:
            if fig_before: pdf.savefig(fig_before)
            if fig_after: pdf.savefig(fig_after)
        plt.close('all')

        # 8. CLEAN DATA SAVE (No RANSAC/AR)
        np.save(os.path.join(save_dir, f"{subject_id}_{condition}_cleaned.npy"), epochs.get_data(copy=True))

        # 9. LOGGING
        n_total = len(epochs)
        num_features = df_features.shape[1] - 2 # Exclude Subject & Condition cols
        
        log_lines = [
            f"Preprocessing Report for {subject_id} ({condition})",
            "="*45,
            f"Methodology:         Li et al. (2026) Raw EEG Analysis",
            f"Initial Cropping:    Discarded first 10 seconds",
            f"Filters applied:     {FILTER_HP} - {FILTER_LP} Hz bandpass, {NOTCH_FREQ} Hz notch",
            f"Artifact Rejection:  None (No RANSAC or AutoReject applied)",
            f"Total Epochs (1s):   {n_total}",
            f"Extracted Features:  {num_features} Connectivity pairs (Coherence)"
        ]
        with open(os.path.join(save_dir, f"{subject_id}_{condition}_processing_log.txt"), 'w') as f:
            f.write("\n".join(log_lines))

        return True, f"OK (Processed {n_total} epochs, {num_features} features)"

    except Exception as e:
        plt.close('all')
        return False, str(e)

# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"🚀 Starting Connectivity Preprocessing Pipeline (Li et al. 2026)")
    if NUM_SUBJECTS_TO_PROCESS is not None:
        print(f"⚠️ TEST MODE ACTIVE: Processing {NUM_SUBJECTS_TO_PROCESS} RANDOM subjects!")
    
    all_files = []
    for folder, pattern, ds_name in DATASETS:
        print(f"📂 Scanning {folder} for {pattern}...")
        
        # Gebruik Pathlib voor robuuster zoeken (werkt beter over mappen heen)
        search_dir = Path(folder)
        if not search_dir.exists():
            print(f"   ❌ FOUT: De map {search_dir} bestaat niet!")
            continue
            
        # Vind alle bestanden (zoekt recursief door alle submappen)
        found_paths = list(search_dir.rglob(pattern))
        found = [str(p) for p in found_paths]
        
        print(f"   -> Ruwe {pattern} bestanden gevonden: {len(found)}")
        
        valid_found = []
        for f in found:
            f_lower = f.lower()
            
            # Sla output folders over (vermijd dat hij eerder geprocessde data pakt)
            if 'clean' in f_lower or 'results' in f_lower:
                continue
            
            # --- PRE-FILTERING ---
            filename = os.path.basename(f).lower()
            
            # Check of het een FM patiënt of HC is
            if 'fmhc' not in filename and 'fmpa' not in filename:
                continue
                
            # Check of het de ogen-dicht conditie is
            if 'closed' not in filename and 'ec' not in filename:
                continue
            # ---------------------------------------
                
            valid_found.append(f)
            
        print(f"   -> Bestanden over na filteren (alleen fmpa/fmhc + closed): {len(valid_found)}")
        
        # Nu we alleen de juiste bestanden over hebben, pakken we er random 1 (of meer)
        if NUM_SUBJECTS_TO_PROCESS is not None and len(valid_found) > 0:
            aantal = min(NUM_SUBJECTS_TO_PROCESS, len(valid_found)) 
            valid_found = random.sample(valid_found, aantal)
            
        for f in valid_found:
            all_files.append((f, ds_name))

    print(f"\n🔍 Total files to evaluate: {len(all_files)}")
    
    results = []
    for file_path, ds_name in tqdm(all_files, desc="Processing"):
        success, msg = process_subject(file_path, OUTPUT_DIR, ds_name)
        results.append((os.path.basename(file_path), success, msg))

    # Summary
    print("\n" + "="*50)
    print("📊 FINAL SUMMARY")
    print("="*50)
    
    successes = [r for r in results if r[1] and "Skipped" not in r[2]]
    skipped = [r for r in results if "Skipped" in r[2]]
    failures = [r for r in results if not r[1]]

    print(f"Total Found: {len(results)}")
    print(f"✅ Processed: {len(successes)}")
    print(f"⏭️  Skipped:   {len(skipped)}")
    print(f"❌ Failed:    {len(failures)}")

    # --- NIEUW: Print de daadwerkelijke foutmeldingen ---
    if len(failures) > 0:
        print("\n🚨 FOUTMELDINGEN:")
        for r in failures:
            bestand = r[0]
            fout = r[2]
            print(f" -> {bestand}: {fout}")