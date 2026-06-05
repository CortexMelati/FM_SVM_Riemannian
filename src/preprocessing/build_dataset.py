"""
=============================================================================
DATASET AGGREGATION & TRAIN/TEST SPLIT (Li et al., 2026 Methodology)
=============================================================================
Overview:
    This script aggregates individual patient feature files (*_features.csv)
    into a single master dataset and handles the class imbalance exactly as 
    described by Li et al. (2026).
    
    Key Steps:
        1. Aggregation & Labeling (0 = fmhc, 1 = fmpa)
        2. Subject-Level Split: Stratified 80/20 split based on unique subjects
           to prevent data leakage.
        3. Segment Sampling: Balances the training set by using 5 segments 
           for Fibromyalgia patients and 3 segments for Healthy Controls.
           (Note: Requires preprocessing pipeline to output multiple segments).

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

from config import RESULTS_DIR, LABEL_MAPPING, TEST_SIZE, RANDOM_STATE

# =============================================================================
# 1. AGGREGATE INDIVIDUAL FILES & LABELING
# =============================================================================
print("🚀 Starting Dataset Aggregation...")

search_path = RESULTS_DIR / "CP_FM"
file_pattern = os.path.join(search_path, "**", "*_features.csv")
feature_files = glob.glob(file_pattern, recursive=True)

if not feature_files:
    print(f"❌ Geen *_features.csv bestanden gevonden in {search_path}.")
    sys.exit()

print(f"📂 Found {len(feature_files)} individual feature files. Merging...")

all_data = []
for file in feature_files:
    try:
        df = pd.read_csv(file)
        all_data.append(df)
    except Exception as e:
        print(f"⚠️ Fout bij inlezen van {file}: {e}")

master_df = pd.concat(all_data, ignore_index=True)

def assign_label(subject_id):
    subject_upper = str(subject_id).upper()
    for label_key, identifiers in LABEL_MAPPING.items():
        for identifier in identifiers:
            if identifier in subject_upper:
                return int(label_key.split('_')[1])
    return None

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
# 3. SEGMENT SAMPLING FOR BALANCING (Li et al., 2026)
# =============================================================================
# Training set: 5 sections from fmpa, 3 sections from fmhc
sampled_train_data = []

for subject, group in train_full_df.groupby('Subject'):
    target = group['Target'].iloc[0]
    
    if target == 1:
        # Fibromyalgia: Keep 5 segments
        n_samples = min(5, len(group))
        sampled_train_data.append(group.sample(n=n_samples, random_state=RANDOM_STATE))
    else:
        # Healthy Control: Keep 3 segments
        n_samples = min(3, len(group))
        sampled_train_data.append(group.sample(n=n_samples, random_state=RANDOM_STATE))

# Concatenate and shuffle the final training set
train_df = pd.concat(sampled_train_data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# =============================================================================
# 4. SAVE FINAL DATASETS
# =============================================================================
train_path = RESULTS_DIR / "final_dataset_train.csv"
test_path = RESULTS_DIR / "final_dataset_test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n" + "="*50)
print("✅ DATASET CREATION SUCCESSFUL (Li et al. Methodology)")
print("="*50)
print(f"Train set saved: {train_path}")
print(f"   -> Rows: {len(train_df)} (Balanced sampling applied)")
print(f"Test set saved:  {test_path}")
print(f"   -> Rows: {len(test_df)} (All segments kept)")
print("\nReady for Feature Selection (mSFFS) and SVM Training!")