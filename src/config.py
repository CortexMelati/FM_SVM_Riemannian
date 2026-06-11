"""
Central Configuration for Thesis EEG Pipeline.
Aligned with the methodology of Li et al. (2026).
"""

import os
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# This file is located in: .../FM_SVM_RIEMANNIAN/src
CURRENT_DIR = Path(__file__).resolve().parent

# PROJECT_ROOT is one directory up: .../FM_SVM_RIEMANNIAN
PROJECT_ROOT = CURRENT_DIR.parent

# DATA_ROOT is two directories up (Thesis) and then to Data: .../Thesis/Data
DATA_ROOT = Path.home() / "Documents" / "Thesis" / "Data"

# Input Directories
FM_DIR = DATA_ROOT / "FM_EO_dataset"           # Epoched itRS dataset
CP_FM_DIR = DATA_ROOT / "CP_FM_dataset"        # PainMunich dataset
TDBRAIN_DIR = DATA_ROOT / "TDBRAIN-dataset"    # TDBRAIN dataset
CHRONIC_PAIN_DIR = DATA_ROOT / "Chronicpainset" # Chronic Pain dataset
# NOT_USABLE_DIR = DATA_ROOT / "OSF_mj9xr_notusable" # Ignored by the pipeline


# Output Directories (binnen FM_SVM_RIEMANNIAN/results)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PROCESSED_DATA_DIR = RESULTS_DIR / "processed_data"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. PREPROCESSING & TIMING PARAMETERS (Li et al., 2026)
# ==========================================
# Native sampling rates per hardware system (linked to the respective directory names)
SFREQ_MAP = {
    "FM_EO_dataset": 250,      # Native itRS data
    "CP_FM_dataset": 500,      # PainMunich after downsampling (1000Hz -> 500Hz)
    "TDBRAIN-dataset": 500,    # Native Mitsar system
    "Chronicpainset": 500      # Native continuous system
}

# Substrings in filenames used to automatically assign target labels 
# (0 = Healthy Control, 1 = Patient/Chronic Pain/Fibromyalgia)
LABEL_MAPPING = {
    'Healthy_0': ['FMHC'], #, 'HC', 'HEALTHY', 'CONTROL'],
    'Patient_1': ['FMPA'] #, 'FM', 'CP', 'PAIN', 'CHRONICPAIN'] 
}


# Filter specifications according to the reference paper
FILTER_HP = 0.5           # High-pass filter cutoff (Hz)
FILTER_LP = 44.0          # Low-pass filter cutoff (Hz)
NOTCH_FREQ = 50.0         # Line noise notch filter (Hz)


# Segmentation windows for CONTINUOUS data (TDBRAIN, Chronicpainset, CP_FM_dataset)
TRUNCATE_TIME = 10.0      # Initial duration to discard for continuous data (seconds)
SEGMENT_LENGTH = 30.0     # Macro-segments for continuous datasets (seconds)
EPOCH_LENGTH = 1.0        # Micro-epochs for connectivity or covariance analysis (seconds)


# Segmentation windows for PRE-EPOCHED itRS data (FM_EO_dataset)
ITRS_TMIN = 0.0           # Start point of the itRS trial
ITRS_TMAX = 1.348         # Exactly 337 samples at 250Hz (approx. 1350 ms)

# ==============================================================================
# 3. CHANNEL CONFIGURATION (Standard International 10-20)
# ==============================================================================
# The exact 19 channels utilized in the core feature space of the paper
CHANNELS_1020 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',   # T3 -> T7, T4 -> T8
    'P7', 'P3', 'Pz', 'P4', 'P8',   # T5 -> P7, T6 -> P8
    'O1', 'O2'
]

# Mapping dictionary to synchronize legacy channel names with
# modern MNE standard montages (e.g., 'standard_1020')
CHANNEL_RENAMING_MAP = {
    'T3': 'T7', 'T4': 'T8',
    'T5': 'P7', 'T6': 'P8'
}


# ==============================================================================
# 4. FREQUENCY BANDS OF INTEREST
# ==============================================================================
# Strictly defined spectral boundaries corresponding to Equation 1
BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),    
    'Beta':  (12, 30),
    'Gamma': (30, 40)     # capped at 40 as stated in the paper from li et al. 
}

# ==============================================================================
# 5. MACHINE LEARNING & VISUALIZATION METRICS
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2 # paper uses 0.1 
MIN_AGE = 18.0


# Documented core features identified via mSFFS for reference verification
PAPER_TOP_5_FEATURES = [
    ('Fz', 'Cz'), ('Pz', 'P4'), ('Fz', 'C3'), ('Cz', 'P4'), ('Cz', 'Pz')
]

# Consistent color scheme for plots
COLORS = {
    'Healthy': '#1f77b4',      # Blue
    'ChronicPain': '#d62728',  # Red
    'FM': '#9467bd',           # Purple
    'Unknown': '#7f7f7f'       # Grey
}

if __name__ == "__main__":
    print(f"✅ Configuration Loaded.")
    print(f"📂 Config Location: {CURRENT_DIR}")
    print(f"📂 Project Root:    {PROJECT_ROOT}")
    print(f"📂 Data Root:       {DATA_ROOT}")
    print(f"📂 Results Dir:     {RESULTS_DIR}\n")
    
    datasets = [
        ("FM_EO_dataset (itRS)", FM_DIR), 
        ("TDBRAIN-dataset (Cont)", TDBRAIN_DIR), 
        ("Chronicpainset (Cont)", CHRONIC_PAIN_DIR),
        ("CP_FM_dataset (Cont)", CP_FM_DIR)
    ]
    
    for name, directory in datasets:
        if directory.exists():
            print(f"   -> 📂 {name} path found! ✅")
        else:
            print(f"   -> ❌ {name} path NOT found at: {directory}")
            
    if (RESULTS_DIR / "final_dataset.csv").exists():
        print(f"   -> final_dataset.csv found! ✅")
    else:
        print(f"   -> ⚠️ final_dataset.csv not yet generated in the results folder.")
        
        
# ==============================================================================
# 6. CLASS LABELING / NAMING CONVENTIONS
# ==============================================================================
# Substrings in filenames used to automatically assign target labels 
# (0 = Healthy Control, 1 = Chronic Pain / Fibromyalgia)

LABEL_MAPPING = {
    'Healthy_0': ['FMHC', 'HC', 'HEALTHY'],
    'Patient_1': ['FMPA', 'FM'] 
}