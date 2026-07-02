"""
=============================================================================
 TDBRAIN METADATA SPLITTER (MUTUALLY EXCLUSIVE)
=============================================================================
Objective:
    Split the master TDBrain metadata file into four distinct, non-overlapping
    sub-lists based on 'formal_status' and 'indication'.

Criteria for Split:
    1. HEALTHY:        Subjects with formal_status = 'HEALTHY'.
    2. CHRONIC PAIN:   Subjects with formal_status = 'CHRONIC PAIN'.
    3. UNKNOWN (INF):  Subjects with formal_status = 'UNKNOWN' BUT have an 'indication'.
    4. UNKNOWN (NaN):  Subjects with formal_status = 'UNKNOWN' AND an empty 'indication'.

Input:
    - Data/TDBRAIN-dataset/TDBRAIN_participants_V2.xlsx

Output:
    - TDBRAIN_participants_HEALTHY.xlsx
    - TDBRAIN_participants_CHRONIC_PAIN.xlsx
    - TDBRAIN_participants_UNKNOWN.xlsx       (Informal Indications only)
    - TDBRAIN_participants_UNKNOWN_NaNs.xlsx  (Empty indications only)

Execution:
    python ./FM_thesis_ML/src/Preprocessing/split_participants_TDBRAIN.py
=============================================================================
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import TDBRAIN_DIR

# Define Source and Output Files
source_file = TDBRAIN_DIR / "TDBRAIN_participants_V2.xlsx"
output_healthy      = TDBRAIN_DIR / "TDBRAIN_participants_HEALTHY.xlsx"
output_pain         = TDBRAIN_DIR / "TDBRAIN_participants_CHRONIC_PAIN.xlsx"
output_unknown      = TDBRAIN_DIR / "TDBRAIN_participants_UNKNOWN.xlsx"
output_unknown_nans = TDBRAIN_DIR / "TDBRAIN_participants_UNKNOWN_NaNs.xlsx"

def split_participants():
    print(f"► Loading source file: {source_file.name}")
    
    if not source_file.exists():
        print(f"✗ ERROR: File not found at {source_file}")
        return

    df = pd.read_excel(source_file)
    df.columns = [c.strip() for c in df.columns]

    print(f"  - Total rows in master file: {len(df)}")

    # 1. Filter Logic: 'Formal Status'
    cond_healthy = df['formal_status'].astype(str).str.upper() == 'HEALTHY'
    cond_pain    = df['formal_status'].astype(str).str.upper() == 'CHRONIC PAIN'
    
    # 2. Filter Logic: 'Unknown Status'
    cond_status_unknown = (
        df['formal_status'].isna() | 
        (df['formal_status'].astype(str).str.upper() == 'UNKNOWN') |
        (df['formal_status'].astype(str).str.upper() == 'NAN')
    )

    # 3. Filter Logic: 'Missing Indication'
    if 'indication' in df.columns:
        cond_indication_missing = (
            df['indication'].isna() | 
            (df['indication'].astype(str).str.strip() == '') |
            (df['indication'].astype(str).str.upper() == 'NAN')
        )
    else:
        cond_indication_missing = pd.Series([True] * len(df))

    # Perform Mutually Exclusive Split
    df_healthy = df[cond_healthy].copy()
    df_pain = df[cond_pain].copy()
    df_unknown = df[cond_status_unknown & ~cond_indication_missing].copy()
    df_nans = df[cond_status_unknown & cond_indication_missing].copy()

    # Save and Report
    print("\n► Saving Results (No Overlap)...")
    
    df_healthy.to_excel(output_healthy, index=False)
    print(f"  ✓ Healthy                 : {len(df_healthy):<5} -> {output_healthy.name}")

    df_pain.to_excel(output_pain, index=False)
    print(f"  ✓ Chronic Pain            : {len(df_pain):<5} -> {output_pain.name}")

    df_unknown.to_excel(output_unknown, index=False)
    print(f"  ✓ Unknown (Informal Ind.) : {len(df_unknown):<5} -> {output_unknown.name}")
    
    df_nans.to_excel(output_unknown_nans, index=False)
    print(f"  ✓ Unknown (No Indication) : {len(df_nans):<5} -> {output_unknown_nans.name}")

    # Verification
    total_subs = len(df_healthy) + len(df_pain) + len(df_unknown) + len(df_nans)
    print("-" * 50)
    print(f"  ∑ Total Exported          : {total_subs}")
    print(f"  - Original File Count     : {len(df)}")
    
    if total_subs < len(df):
        print(f"  ! Note: {len(df) - total_subs} subjects were NOT exported (e.g., OCD/ADHD). This is expected behavior!")

if __name__ == "__main__":
    split_participants()