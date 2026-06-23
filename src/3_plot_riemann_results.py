"""
=============================================================================
3. RIEMANNIAN BIOMARKER MAP (TOPOGRAPHY)
=============================================================================
Overview:
    This script opens the winning Riemannian model (TSSVM), extracts the 
    internal Linear SVM weights from the Tangent Space, maps them back to 
    the original 9-channel ROI matrix, and plots the top 5 connectivity 
    features on a standard 19-channel topographical brain map.

Execution:
    python 3_plot_riemann_results.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import joblib
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import SVM_DATA_DIR, RIEMANN_FIGURES_DIR, BEST_CHANNELS_EVALUATE

def plot_riemannian_weights():
    # 1. ZOEK HET WINNENDE MODEL
    model_files = list(SVM_DATA_DIR.glob("model_riemann_*.pkl"))
    if not model_files:
        print("🚨 Geen Riemannian model gevonden in svm_data/.")
        sys.exit()
        
    # Pak het laatst opgeslagen model
    model_path = model_files[-1]
    band_name = model_path.stem.split('_')[2]
    print(f"🚀 Genereren van Topografisch Netwerk voor TS-SVM ({band_name.upper()} Band)...")
    
    artifact = joblib.load(model_path)
    # Handle both direct pipeline saves and dictionary saves
    pipeline = artifact['model'] if isinstance(artifact, dict) else artifact

    # 2. HAAL DE GEWICHTEN UIT DE TANGENT SPACE SVM
    try:
        svm_coefs = pipeline.named_steps['svm'].coef_[0] 
    except (AttributeError, KeyError):
        print("🚨 Kan geen lineaire SVM coëfficiënten vinden in dit model.")
        sys.exit()

    # 3. RECONSTRUEER DE TANGENT SPACE INDEXERING (Voor de 9 ROI kanalen)
    roi_channels = BEST_CHANNELS_EVALUATE
    n_channels = len(roi_channels)
    pair_map = []
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            pair_map.append((roi_channels[i], roi_channels[j]))

    if len(svm_coefs) != len(pair_map):
        print(f"🚨 Dimensie mismatch! TS-SVM heeft {len(svm_coefs)} features, verwachtte er {len(pair_map)}.")
        sys.exit()

    # 4. KOPPEL EN FILTER DE GEWICHTEN
    weights_df = pd.DataFrame({
        'Node1': [p[0] for p in pair_map],
        'Node2': [p[1] for p in pair_map],
        'Weight': np.abs(svm_coefs)
    })
    
    # Filter variantie op hetzelfde kanaal (we willen connectiviteit)
    weights_df = weights_df[weights_df['Node1'] != weights_df['Node2']]
    
    # Top 5
    top_5 = weights_df.sort_values(by='Weight', ascending=False).head(5)
    
    print("\n-> Top 5 Riemannian Connecties (Tangent Space Weights):")
    for _, row in top_5.iterrows():
        print(f"   {row['Node1']:<3} - {row['Node2']:<3} | Gewicht: {row['Weight']:.4f}")

    # 5. MNE TOPOGRAFIE TEKENEN (Stijl van Figuur 4)
    standard_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=standard_19, sfreq=500, ch_types='eeg')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_sensors(info, show_names=True, axes=ax)
    
    for collection in ax.collections:
        collection.set_sizes([600])
        collection.set_facecolor('white')
        collection.set_edgecolor('#cccccc')
        collection.set_linewidth(1.5)
        
    sensor_offsets = ax.collections[0].get_offsets()
    ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

    max_weight = top_5['Weight'].max()
    top_5['Scaled_Importance'] = (top_5['Weight'] / max_weight) * 2.5

    for _, row in top_5.iterrows():
        try:
            x_coords = [ch_pos[row['Node1']][0], ch_pos[row['Node2']][0]]
            y_coords = [ch_pos[row['Node1']][1], ch_pos[row['Node2']][1]]
            
            scaled_val = row['Scaled_Importance']
            if scaled_val >= 2.0:
                color, lw = '#FF8C94', 5.0  # Roze
            elif scaled_val >= 1.0:
                color, lw = '#8B4513', 3.5  # Bruin
            else:
                color, lw = '#228B22', 2.0  # Groen
                
            ax.plot(x_coords, y_coords, color=color, linewidth=lw, alpha=0.9, zorder=0)
        except KeyError:
            pass

    ax.set_title(f"Riemannian TS-SVM Connectivity\n({band_name.upper()} Band - ROI)", fontsize=14, pad=20)
    plt.tight_layout()
    
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RIEMANN_FIGURES_DIR / f"Figure_Riemann_Network_{band_name}.png"
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close()
    print(f"\n✅ Hersenkaart opgeslagen: riemann_figures/{save_path.name}")

if __name__ == "__main__":
    plot_riemannian_weights()