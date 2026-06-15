"""
=============================================================================
5. GLOBAL FEATURE RANKING MODEL (RIEMANNIAN TANGENT SPACE)
=============================================================================
Overview:
    This script extracts the global feature weights from the trained linear 
    TS-SVM models, maps the 1D Tangent Space dimensions back to the original 
    19x19 electrode pairs, and exports a ranked feature importance table.
    
python 5_riemann_feature_ranking.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, CHANNELS_1020, BANDS

def compute_global_ranking(target_band='BETA'):
    print(f"🚀 Extracting global feature rankings for {target_band} band...")
    
    model_path = PROCESSED_DATA_DIR / f"model_riemann_{target_band}_TSSVM.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"🚨 Model artifact missing: {model_path.name}")
        
    pipeline = joblib.load(model_path)
    svm_coefs = pipeline.named_steps['svm'].coef_[0]

    # Reconstruct the exact mathematical indexing scheme of PyRiemann TangentSpace
    n_channels = len(CHANNELS_1020)
    pair_map = []
    for i in range(n_channels):
        for j in range(i, n_channels):
            pair_map.append((CHANNELS_1020[i], CHANNELS_1020[j]))

    # Compile into a structured dataframe
    ranking_df = pd.DataFrame({
        'Feature_Index': range(len(svm_coefs)),
        'Channel_A': [p[0] for p in pair_map],
        'Channel_B': [p[1] for p in pair_map],
        'Raw_Weight': svm_coefs,
        'Absolute_Importance': np.abs(svm_coefs)
    })

    # Filter out auto-variance (diagonal elements where Channel_A == Channel_B)
    ranking_df = ranking_df[ranking_df['Channel_A'] != ranking_df['Channel_B']]

    # Sort hierarchically by absolute mathematical leverage
    ranking_df = ranking_df.sort_values(by='Absolute_Importance', ascending=False).reset_index(drop=True)
    ranking_df['Global_Rank'] = ranking_df.index + 1

    # Reorder columns for academic reporting
    cols = ['Global_Rank', 'Channel_A', 'Channel_B', 'Absolute_Importance', 'Raw_Weight']
    ranking_df = ranking_df[cols]

    # Export comprehensive CSV
    output_path = PROCESSED_DATA_DIR / f"riemann_feature_ranking_{target_band.lower()}.csv"
    ranking_df.to_csv(output_path, index=False)
    
    print(f"\n🏆 TOP 10 GLOBAL RIEMANNIAN FEATURES ({target_band}):")
    print("-" * 75)
    print(ranking_df.head(10).to_string(index=False))
    print("-" * 75)
    print(f"✅ Complete ranking exported to: {output_path.name}")

if __name__ == "__main__":
    # Run ranking for your top-performing predictive bands
    for band in ['BETA', 'GAMMA']:
        try:
            compute_global_ranking(target_band=band)
        except Exception as e:
            print(f"  ❌ Could not process {band}: {e}")