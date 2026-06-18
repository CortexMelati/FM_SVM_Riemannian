"""
=============================================================================
5. GLOBAL FEATURE RANKING MODEL (RIEMANNIAN TANGENT SPACE)
=============================================================================
Overview:
    This script extracts the global feature weights from the trained linear 
    TS-SVM models, maps the 1D Tangent Space dimensions back to the original 
    electrode pairs (adapting dynamically to 19 or 9 channels), and exports 
    a ranked feature importance table.
    
Execution:
    python 5_riemann_feature_ranking.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
# Toegevoegd: RIEMANN_FIGURES_DIR zodat we de plot kunnen opslaan
from config import PROCESSED_DATA_DIR, CHANNELS_1020, BANDS, BEST_CHANNELS_EVALUATE, RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR

ROI_CHANNELS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

def plot_riemann_bar_chart(ranking_df, target_band, layout):
    """
    Maakt een Bar Chart van de Riemann top-features, in de stijl van Fig 6A.
    """
    # Pak de top 10 features
    top_df = ranking_df.head(10).copy()
    top_df['Feature_Name'] = top_df['Channel_A'] + "-" + top_df['Channel_B'] + f"({target_band.lower()})"
    
    # Sorteer oplopend zodat de belangrijkste bovenaan staat in de horizontale plot
    top_df = top_df.sort_values(by='Absolute_Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    
    # De paper gebruikt een strakke, effen kleur
    bars = plt.barh(top_df['Feature_Name'], top_df['Absolute_Importance'], color='#5c8cbc')
    
    # Zet de waardes rechts naast de balkjes
    for bar in bars:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                 f"{bar.get_width():.2f}", 
                 va='center', ha='left', fontsize=10)

    plt.title(f"Riemannian Feature Importance - {target_band.upper()} Band ({layout.upper()})", pad=20)
    plt.xlabel('Absolute SVM Weight (Tangent Space)', fontsize=11)
    
    # Clean up axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(RIEMANN_FIGURES_DIR / f"riemann_bar_plot_{target_band.lower()}_{layout}.png", dpi=300)
    plt.close()   

def compute_global_ranking(target_band='BETA', layout='whole'):
    print(f"🚀 Extracting global feature rankings for {target_band} band ({layout.upper()} layout)...")
    
    model_path = RIEMANN_DATA_DIR / f"model_riemann_{target_band}_{layout}_TSSVM.pkl"
    if not model_path.exists():
        print(f"  🚨 Model artifact missing: {model_path.name}")
        return
        
    pipeline = joblib.load(model_path)
    svm_coefs = pipeline.named_steps['svm'].coef_[0]

    # Select target channel map based on layout setting
    active_channels = CHANNELS_1020 if layout == 'whole' else BEST_CHANNELS_EVALUATE

    # Reconstruct the exact mathematical indexing scheme of PyRiemann TangentSpace
    n_channels = len(active_channels)
    pair_map = []
    for i in range(n_channels):
        for j in range(i, n_channels):
            pair_map.append((active_channels[i], active_channels[j]))

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
    output_path = RIEMANN_DATA_DIR / f"riemann_feature_ranking_{target_band.lower()}_{layout}.csv"
    ranking_df.to_csv(output_path, index=False)
    
    print(f"\n🏆 TOP 5 GLOBAL RIEMANNIAN FEATURES ({target_band} - {layout.upper()}):")
    print("-" * 75)
    print(ranking_df.head(5).to_string(index=False))
    print("-" * 75)
    
    # ---- NIEUW: ROEP DE VISUALISATIE AAN ----
    plot_riemann_bar_chart(ranking_df, target_band, layout)
    print(f"  ✓ Bar chart saved to {RIEMANN_FIGURES_DIR.name}")

if __name__ == "__main__":
    # Run ranking for your top-performing predictive bands across both layouts
    for band in ['BETA', 'GAMMA']:
        for layout in ['whole', 'roi']:
            try:
                compute_global_ranking(target_band=band, layout=layout)
            except Exception as e:
                print(f"  ❌ Could not process {band} ({layout}): {e}")