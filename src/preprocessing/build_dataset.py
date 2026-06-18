"""
=============================================================================
DATASET AGGREGATION & TRAIN/TEST SPLIT (Li et al., 2026 Methodology)
=============================================================================
Overview:
    Aggregates feature files, identifies the study cohort via participants.tsv,
    and STRICTLY isolates the primary cohort for the Train/Test split.
    The target domain (e.g., 'NCCP') is saved separately to prevent data leakage
    prior to cross-domain validation.
=============================================================================
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, LABEL_MAPPING, TEST_SIZE, RANDOM_STATE, 
                    PROCESSED_DATA_DIR, PROJECT_ROOT, ACTIVE_DATASET_NAME,
                    CROSS_SOURCE_DATASET, CROSS_TARGET_DATASET)

print("Starting Dataset Aggregation...")

# 1. LOAD FEATURES
file_pattern = os.path.join(RESULTS_DIR, "**", "*_features.csv")
feature_files = glob.glob(file_pattern, recursive=True)

if not feature_files:
    print(f"Error: No *_features.csv files found.")
    sys.exit()

all_data = []
for file in feature_files:
    all_data.append(pd.read_csv(file))
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

# 2. MERGE WITH METADATA TO PREVENT DATA MIXING
tsv_path = PROJECT_ROOT / "data" / ACTIVE_DATASET_NAME / "data" / "participants.tsv"
participants_df = pd.read_csv(tsv_path, sep='\t')
if 'participant_id' in participants_df.columns:
    participants_df['Subject'] = participants_df['participant_id'].str.replace('sub-', '')

merged_df = pd.merge(master_df, participants_df[['Subject', 'study']], on='Subject', how='inner')

# 3. ISOLATE PRIMARY COHORT AND TARGET COHORT
# Using variables defined in config.py (e.g., FM and NCCP)
source_cohort_name = "FM"
target_cohort_name = "NCCP"

source_df = merged_df[merged_df['study'] == source_cohort_name].copy()
target_df = merged_df[merged_df['study'] == target_cohort_name].copy()

# Drop the 'study' column now that we have isolated them
source_df = source_df.drop(columns=['study'])
target_df = target_df.drop(columns=['study'])

print("\nDATA SEPARATION COMPLETE:")
print(f"   -> Primary Cohort ({source_cohort_name}): {source_df['Subject'].nunique()} subjects isolated for Train/Test.")
print(f"   -> Target Cohort ({target_cohort_name}): {target_df['Subject'].nunique()} subjects isolated for Cross-Domain.")

# Save Target Domain
target_path = PROCESSED_DATA_DIR / f"target_domain_{target_cohort_name.lower()}.csv"
target_df.to_csv(target_path, index=False)

# =============================================================================
# 4. SUBJECT-LEVEL TRAIN/TEST SPLIT (ONLY ON THE SOURCE COHORT)
# =============================================================================
unique_subjects = source_df[['Subject', 'Target']].drop_duplicates()

train_subs, test_subs = train_test_split(
    unique_subjects['Subject'], test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, stratify=unique_subjects['Target']
)

train_full_df = source_df[source_df['Subject'].isin(train_subs)].copy()
test_df = source_df[source_df['Subject'].isin(test_subs)].copy()

# =============================================================================
# 5. DYNAMIC SEGMENT SAMPLING (Balancing the Training Set)
# =============================================================================
total_patient_segments = len(train_full_df[train_full_df['Target'] == 1])
total_hc_segments = len(train_full_df[train_full_df['Target'] == 0])
sampled_train_data = []

if total_patient_segments <= total_hc_segments:
    anchor_volume = total_patient_segments
    majority_subjects = train_full_df[train_full_df['Target'] == 0]['Subject'].nunique()
    optimal_segments = max(1, round(anchor_volume / majority_subjects)) if majority_subjects > 0 else 5
    
    for subject, group in train_full_df.groupby('Subject'):
        target = group['Target'].iloc[0]
        n_samples = min(5, len(group)) if target == 1 else min(optimal_segments, len(group))
        sampled_train_data.append(group.sample(n=n_samples, random_state=RANDOM_STATE))
else:
    anchor_volume = total_hc_segments
    majority_subjects = train_full_df[train_full_df['Target'] == 1]['Subject'].nunique()
    optimal_segments = max(1, round(anchor_volume / majority_subjects)) if majority_subjects > 0 else 5
    
    for subject, group in train_full_df.groupby('Subject'):
        target = group['Target'].iloc[0]
        n_samples = min(5, len(group)) if target == 0 else min(optimal_segments, len(group))
        sampled_train_data.append(group.sample(n=n_samples, random_state=RANDOM_STATE))

train_df_final = pd.concat(sampled_train_data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Hold-out testset (Strictly 5 segments)
sampled_test_data = []
for subject, group in test_df.groupby('Subject'):
    if len(group) >= 5:
        sampled_test_data.append(group.sort_values('Segment').head(5))

test_df_final = pd.concat(sampled_test_data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# =============================================================================
# 6. SAVE FINAL DATASETS
# =============================================================================
train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
test_path = PROCESSED_DATA_DIR / "final_dataset_test.csv"

train_df_final.to_csv(train_path, index=False)
test_df_final.to_csv(test_path, index=False)

print("\nDATASET CREATION SUCCESSFUL (Methodology Aligned)")
print(f"Train set ({source_cohort_name}): {train_path.name} | Rows: {len(train_df_final)}")
print(f"Test set ({source_cohort_name}):  {test_path.name} | Rows: {len(test_df_final)}")
print(f"Target ({target_cohort_name}):  {target_path.name} | Rows: {len(target_df)}")