"""
=============================================================================
DATASET AGGREGATION & TRAIN/TEST SPLIT (Li et al., 2026 Methodology)
=============================================================================
Overview:
    This script aggregates individual patient feature files (*_features.csv)
    into a single master dataset and handles the class imbalance via an 
    automated data-density calculation to ensure a ~1:1 class distribution.
    
    Key Steps:
        1. Aggregation & Labeling (0 = hc, 1 = patient)
        2. Subject-Level Split: Stratified 80/20 split based on unique subjects
           to prevent data leakage.
        3. Automated Segment Sampling: Dynamically calculates the optimal 
           number of segments per Healthy Control to match the total volume
           of patient segments.

Execution:
    python build_dataset.py
=============================================================================
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, LABEL_MAPPING, TEST_SIZE, RANDOM_STATE, PROCESSED_DATA_DIR

# =============================================================================
# 1. AGGREGATE INDIVIDUAL FILES & LABELING
# =============================================================================
print("🚀 Starting Dataset Aggregation...")

search_path = RESULTS_DIR 
file_pattern = os.path.join(search_path, "**", "*_features.csv")
feature_files = glob.glob(file_pattern, recursive=True)

if not feature_files:
    print(f"❌ No *_features.csv files found in {search_path}.")
    sys.exit()

print(f"📂 Found {len(feature_files)} individual feature files. Merging...")

all_data = []
for file in feature_files:
    try:
        df = pd.read_csv(file)
        all_data.append(df)
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

master_df = pd.concat(all_data, ignore_index=True)

def assign_label(subject_id):
    subject_upper = str(subject_id).upper()
    for label_key, identifiers in LABEL_MAPPING.items():
        for identifier in identifiers:
            if identifier in subject_upper:
                return int(label_key.split('_')[1])
    return None

master_df = master_df.copy()
master_df['Target'] = master_df['Subject'].apply(assign_label)
master_df = master_df.dropna(subset=['Target'])
master_df['Target'] = master_df['Target'].astype(int)

# Reorder columns
cols = master_df.columns.tolist()
cols.insert(2, cols.pop(cols.index('Target')))
master_df = master_df[cols]

# =============================================================================
# 2. SUBJECT-LEVEL TRAIN/TEST SPLIT (To prevent data leakage)
# =============================================================================
# Extract unique subjects and their targets
unique_subjects = master_df[['Subject', 'Target']].drop_duplicates()

print(f"📊 Total UNIQUE subjects: {len(unique_subjects)}")
print(f"   -> Patients (1): {sum(unique_subjects['Target'] == 1)}")
print(f"   -> Controls (0): {sum(unique_subjects['Target'] == 0)}")

# Split at the SUBJECT level (80/20) with stratification
train_subs, test_subs = train_test_split(
    unique_subjects['Subject'], 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=unique_subjects['Target']
)

# Allocate all segments from the selected subjects to their respective sets
train_full_df = master_df[master_df['Subject'].isin(train_subs)].copy()
test_df = master_df[master_df['Subject'].isin(test_subs)].copy()

# =============================================================================
# 3. BIDIRECTIONAL DYNAMIC SEGMENT SAMPLING (Li et al., 2026 Optimized)
# =============================================================================
print("\n⚖️ Calculating optimal segment ratio for training set balancing...")

total_patient_segments = len(train_full_df[train_full_df['Target'] == 1])
total_hc_segments = len(train_full_df[train_full_df['Target'] == 0])

sampled_train_data = []

# SCENARIO A: Patients are the minority 
if total_patient_segments <= total_hc_segments:
    print("   -> Imbalance detected: Healthy Controls are the majority.")
    anchor_volume = total_patient_segments
    majority_subjects = train_full_df[train_full_df['Target'] == 0]['Subject'].nunique()
    
    # Calculate how many segments per HC subject we can keep
    optimal_segments = max(1, round(anchor_volume / majority_subjects)) if majority_subjects > 0 else 5
    
    for subject, group in train_full_df.groupby('Subject'):
        target = group['Target'].iloc[0]
        if target == 1:
            n_samples = min(5, len(group)) # Anchor: Take all available minority segments
        else:
            n_samples = min(optimal_segments, len(group)) # Buffer: Downsample the majority
        sampled_train_data.append(group.sample(n=n_samples, random_state=RANDOM_STATE))

# SCENARIO B: Healthy Controls are the minority 
else:
    print("   -> Imbalance detected: Patients are the majority.")
    anchor_volume = total_hc_segments
    majority_subjects = train_full_df[train_full_df['Target'] == 1]['Subject'].nunique()
    
    # Calculate how many segments per Patient subject we can keep
    optimal_segments = max(1, round(anchor_volume / majority_subjects)) if majority_subjects > 0 else 5
    
    for subject, group in train_full_df.groupby('Subject'):
        target = group['Target'].iloc[0]
        if target == 0:
            n_samples = min(5, len(group)) # Anchor: Take all available minority segments
        else:
            n_samples = min(optimal_segments, len(group)) # Buffer: Downsample the majority
        sampled_train_data.append(group.sample(n=n_samples, random_state=RANDOM_STATE))

train_df = pd.concat(sampled_train_data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Hold-out testset validation remains strictly 5 segments per subject
sampled_test_data = []
for subject, group in test_df.groupby('Subject'):
    if len(group) >= 5:
        sampled_test_data.append(group.sort_values('Segment').head(5))
    else:
        print(f"⚠️ Subject {subject} removed from hold-out set: contains only {len(group)}/5 segments.")

if not sampled_test_data:
    raise ValueError("🚨 No subjects in the hold-out testset meet the 5-segment requirement!")

test_df_final = pd.concat(sampled_test_data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# =============================================================================
# 4. SAVE FINAL DATASETS
# =============================================================================
train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
test_path = PROCESSED_DATA_DIR / "final_dataset_test.csv"

train_df.to_csv(train_path, index=False)
test_df_final.to_csv(test_path, index=False)

print("\n" + "="*50)
print("✅ DATASET CREATION SUCCESSFUL (Methodology Aligned)")
print("="*50)
print(f"Train set saved: {train_path.name}")
print(f"   -> Rows: {len(train_df)} (Balanced sampling applied)")
print(f"   -> Unique Training Subjects: {train_df['Subject'].nunique()}")
print(f"Test set saved:  {test_path.name}")
print(f"   -> Rows: {len(test_df_final)} (Strictly 5 segments per subject)")
print(f"   -> Unique Test Subjects: {test_df_final['Subject'].nunique()}")