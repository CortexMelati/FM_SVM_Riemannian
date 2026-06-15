"""
=============================================================================
3. VISUALISATIE & HERSENNETWERK (RIEMANNIAN MULTI-BAND)
=============================================================================
Overview:
    1. Genereert ROC-curves voor de Cross-Validatie (uit script 2).
    2. Laadt het getrainde TS-SVM model voor een specifieke band.
    3. Extraheert de SVM-gewichten en vertaalt de Tangent Space vector (190 features) 
       terug naar de originele 19x19 kanaalparen.
    4. Plot de top-5 netwerkconnecties op een topografische hersenkaart.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import joblib
from pathlib import Path
import sys

# Configuratie laden
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, FIGURES_DIR, CHANNELS_1020, BANDS

def plot_roc_curves():
    """Genereert een gecombineerde ROC curve plot voor TS-SVM over alle banden."""
    print("🎨 Genereren van ROC Curves...")
    
    plot_data_path = PROCESSED_DATA_DIR / "riemann_plot_data.pkl"
    if not plot_data_path.exists():
        raise FileNotFoundError("🚨 riemann_plot_data.pkl ontbreekt. Draai script 2.")
        
    plot_data = joblib.load(plot_data_path)
    roc_data = plot_data['roc']

    plt.figure(figsize=(8, 6))
    
    # Kleuren per band voor visueel onderscheid
    kleuren = {'DELTA': '#1f77b4', 'THETA': '#ff7f0e', 'ALPHA': '#2ca02c', 
               'BETA': '#d62728', 'GAMMA': '#9467bd'}
    
    # Plot alleen de TS-SVM modellen om de grafiek leesbaar te houden
    for run_name, data in roc_data.items():
        if 'TS-SVM' in run_name:
            band = run_name.split(' | ')[1]
            plt.plot(data['fpr'], data['tpr'], lw=2, color=kleuren.get(band, '#000'), 
                     label=f"{band} (AUC = {data['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title("ROC Curves - Tangent Space SVM per Frequentieband", fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "riemann_multiband_roc.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ Opgeslagen: {save_path.name}")

def plot_topographical_weights(target_band='BETA'):
    """Vertaalt TS-SVM weights naar kanaalparen en plot de topografische kaart."""
    print(f"\n🧠 Genereren van Topografisch Netwerk voor TS-SVM ({target_band})...")
    
    # 1. Laad het getrainde model
    model_path = PROCESSED_DATA_DIR / f"model_riemann_{target_band}_TSSVM.pkl"
    if not model_path.exists():
        print(f"  ⚠️ Model {model_path.name} niet gevonden. Kan hersenkaart niet genereren.")
        return
        
    model = joblib.load(model_path)
    
    # 2. Haal de gewichten uit de Lineaire SVM
    try:
        # TS-SVM pipeline: [covariances, ts, scaler, svm]
        svm_coefs = model.named_steps['svm'].coef_[0] 
    except (AttributeError, KeyError):
        print("  ⚠️ Kan geen lineaire SVM coëfficiënten vinden in dit model.")
        return

    # 3. Reconstrueer de index-mapping van PyRiemann's Tangent Space
    # TangentSpace zet de bovenste driehoek (upper triangle) van de covariantiematrix 
    # om naar een 1D vector, exact in deze volgorde:
    n_channels = len(CHANNELS_1020)
    pair_map = []
    for i in range(n_channels):
        for j in range(i, n_channels):
            pair_map.append((CHANNELS_1020[i], CHANNELS_1020[j]))
            
    if len(svm_coefs) != len(pair_map):
        print(f"  ⚠️ Dimensie mismatch: SVM heeft {len(svm_coefs)} features, layout verwacht {len(pair_map)}.")
        return

    # 4. Koppel de absolute gewichten aan de paren en filter diagonale waarden (variantie) eruit
    weights_df = pd.DataFrame({
        'Pair': pair_map,
        'Weight': np.abs(svm_coefs),
        'Node1': [p[0] for p in pair_map],
        'Node2': [p[1] for p in pair_map]
    })
    
    # We plotten alleen connecties (node1 != node2), geen auto-varianties
    weights_df = weights_df[weights_df['Node1'] != weights_df['Node2']]
    
    # Selecteer de top 5 sterkste connecties
    top_5 = weights_df.sort_values(by='Weight', ascending=False).head(5)
    
    print("  -> Top 5 geïdentificeerde verbindingen (TS-SVM Weights):")
    for _, row in top_5.iterrows():
        print(f"     {row['Node1']:<4} - {row['Node2']:<4} | Gewicht: {row['Weight']:.4f}")

    # 5. Teken de MNE Topografie
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=CHANNELS_1020, sfreq=500, ch_types='eeg')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_sensors(info, show_names=True, axes=ax)
    
    # Styling van de sensoren
    for collection in ax.collections:
        collection.set_sizes([300])
        collection.set_edgecolor('black')
        collection.set_linewidth(1.5)
        
    sensor_offsets = ax.collections[0].get_offsets()
    ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

    # Teken de lijnen, dikte gebaseerd op het genormaliseerde gewicht
    max_weight = top_5['Weight'].max()
    for _, row in top_5.iterrows():
        x_coords = [ch_pos[row['Node1']][0], ch_pos[row['Node2']][0]]
        y_coords = [ch_pos[row['Node1']][1], ch_pos[row['Node2']][1]]
        
        lijn_dikte = (row['Weight'] / max_weight) * 5.0
        ax.plot(x_coords, y_coords, color='#d62728', linewidth=lijn_dikte, zorder=1)

    ax.set_title(f"Sterkste Riemannian Connecties\n(TS-SVM - {target_band.upper()} Band)", fontsize=14, pad=20)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f"riemann_topography_{target_band.lower()}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ Opgeslagen: {save_path.name}")

if __name__ == "__main__":
    plot_roc_curves()
    
    # Pas de frequentieband hier aan naar de band die het best presteerde in je testset.
    plot_topographical_weights(target_band='BETA')