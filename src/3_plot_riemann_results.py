"""
=============================================================================
3. RIEMANNIAN BIOMARKER MAP (TOPOGRAPHY)
=============================================================================
Overview:
    This script generates a physiological network map. Because the optimal
    predictive model (TSSVM_Xdawn, RBF) is non-linear and uses virtual 
    spatial filters, this script fits a surrogate Linear Tangent Space SVM 
    (TSSVM_Cov, Linear) strictly for spatial interpretability.
    
    python 3_plot_riemann_results.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from pathlib import Path
import sys

# Paden instellen
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import RIEMANN_FIGURES_DIR, BEST_CHANNELS_EVALUATE

def plot_surrogate_riemannian_weights():
    print("🚀 Genereren van Topografisch Netwerk via Linear Surrogate Model")
    
    # 1. RECONSTRUEER DE TANGENT SPACE INDEXERING (Voor de 9 ROI kanalen)
    roi_channels = BEST_CHANNELS_EVALUATE
    n_channels = len(roi_channels)
    pair_map = []
    
    # PyRiemann TangentSpace vectoriseert: eerst diagonaal, dan de off-diagonals per rij
    for i in range(n_channels):
        for j in range(i, n_channels):
            pair_map.append((roi_channels[i], roi_channels[j]))

    # -----------------------------------------------------------------------
    # 2. DATA INLADEN (Gebaseerd op jouw mapstructuur)
    # -----------------------------------------------------------------------
    riemann_data_dir = project_root / "results" / "CP_FM_dataset" / "processed_data" / "riemann_data"
    
    covs_path = riemann_data_dir / "covs_train_Theta_roi.npy"
    y_path = riemann_data_dir / "y_train_riemann.npy"
    
    if not covs_path.exists() or not y_path.exists():
        print(f"🚨 Kon de data niet vinden! Check of je het script vanuit de juiste map runt.")
        print(f"Verwacht pad: {covs_path}")
        sys.exit()

    X_cov_train = np.load(covs_path)
    y_train = np.load(y_path)

    # -----------------------------------------------------------------------
    # 3. TRAIN HET LINEAIRE SURROGATE MODEL
    # -----------------------------------------------------------------------
    print("🧠 Projecting Covariances to Tangent Space and fitting Linear SVM...")
    ts = TangentSpace(metric='riemann')
    X_ts = ts.fit_transform(X_cov_train)
    
    # Train een strikt lineaire SVM (C=1.0 is standaard voor interpretatie)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_ts, y_train)
    
    # Omdat het lineair is, kunnen we NU WEL de coef_ trekken!
    svm_coefs = clf.coef_[0]

    if len(svm_coefs) != len(pair_map):
        print(f"🚨 Dimensie mismatch! TS-SVM heeft {len(svm_coefs)} features, verwachtte er {len(pair_map)}.")
        sys.exit()

    # 4. KOPPEL EN FILTER DE GEWICHTEN
    weights_df = pd.DataFrame({
        'Node1': [p[0] for p in pair_map],
        'Node2': [p[1] for p in pair_map],
        'Weight': np.abs(svm_coefs)
    })
    
    # Filter variantie op hetzelfde kanaal (we willen de connecties tussen gebieden, niet kanaal-met-zichzelf)
    weights_df = weights_df[weights_df['Node1'] != weights_df['Node2']]
    
    # Selecteer de Top 5 belangrijkste connecties
    top_5 = weights_df.sort_values(by='Weight', ascending=False).head(5)
    
    # Maak de log-tekst aan
    log_text = "Top 5 Riemannian Connecties (Linear Surrogate Weights) - THETA Band\n"
    log_text += "="*65 + "\n"
    print(f"\n-> {log_text.strip()}")
    
    for _, row in top_5.iterrows():
        line = f"   {row['Node1']:<3} - {row['Node2']:<3} | Gewicht: {row['Weight']:.4f}\n"
        print(line, end="")
        log_text += line

    # Schrijf weg naar een .txt bestand in dezelfde map als de figuren
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    txt_save_path = RIEMANN_FIGURES_DIR / "Riemann_Network_Theta_Surrogate_Weights.txt"
    
    with open(txt_save_path, "w") as f:
        f.write(log_text)
        
    print(f"✅ Gewichten gelogd: {txt_save_path.name}")

    # 5. MNE TOPOGRAFIE TEKENEN
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
                color, lw = '#FF8C94', 5.0  # Roze (Belangrijkst)
            elif scaled_val >= 1.0:
                color, lw = '#8B4513', 3.5  # Bruin
            else:
                color, lw = '#228B22', 2.0  # Groen
                
            ax.plot(x_coords, y_coords, color=color, linewidth=lw, alpha=0.9, zorder=0)
        except KeyError:
            pass

    ax.set_title("Riemannian TS-SVM Connectivity\n(THETA Band - Linear Surrogate)", fontsize=14, pad=20)
    plt.tight_layout()
    
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RIEMANN_FIGURES_DIR / "Figure_Riemann_Network_Theta_Surrogate.png"
    plt.savefig(save_path, dpi=300, transparent=False)
    plt.close()
    print(f"\n✅ Hersenkaart opgeslagen: {save_path}")

if __name__ == "__main__":
    plot_surrogate_riemannian_weights()