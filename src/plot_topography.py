"""
=============================================================================
TOPOGRAPHICAL FEATURE VISUALIZATION
=============================================================================
Overview:
    Genereert topografische hersenkaarten op basis van de centrale config.
    Figuur 2: De 9-kanaals Region of Interest (ROI) (Li et al., 2026).
    Figuur 4: Jouw empirische top 5 connectiviteits-features (CP_FM_dataset).
    
python plot_topography.py
=============================================================================
"""

import matplotlib.pyplot as plt
import mne
from pathlib import Path
import sys

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import FIGURES_DIR, CHANNELS_1020

# ==========================================
# 1. MNE SETUP (Gekoppeld aan config.py)
# ==========================================
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=CHANNELS_1020, sfreq=500, ch_types='eeg')
info.set_montage(montage)

# ==========================================
# 2. FUNCTIE: PLOT ROI (Figure 2)
# ==========================================
def plot_roi_map():
    print("🧠 Genereren van ROI Topografie (Figure 2)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # De 9 centrale sensoren (Li et al., 2026)
    roi_sensors = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    colors = ['#FFD700' if ch in roi_sensors else '#E0E0E0' for ch in info.ch_names]
    
    mne.viz.plot_sensors(
        info, show_names=True, axes=ax,
        ch_groups=[[info.ch_names.index(ch)] for ch in info.ch_names],
        cmap=plt.matplotlib.colors.ListedColormap(colors)
    )
    
    # MNE overschrijven met Matplotlib voor gegarandeerde styling
    for collection in ax.collections:
        collection.set_facecolor(colors)
        collection.set_edgecolor('black')
        collection.set_sizes([400])  # Fix voor de grootte
        collection.set_linewidth(2)  # Fix voor de randdikte

    ax.set_title("Region of Interest (ROI)\n9 Centrale Sensoren", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_2_ROI_Sensors.png", dpi=300)
    plt.close()
    print("  ✓ Figure 2 Opgeslagen")

# ==========================================
# 3. FUNCTIE: PLOT CONNECTIVITY (Figure 4)
# ==========================================
def plot_connectivity_map():
    print("\n🔗 Genereren van Connectivity Topografie (Jouw SHAP Top 5)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Jouw top 5 Beta-band features
    top_connections = [
        ('Cz', 'Pz', '#8B0000', 4.0), 
        ('P3', 'O1', '#B22222', 3.5),
        ('P7', 'Pz', '#B22222', 3.5),
        ('T7', 'C3', '#CD5C5C', 3.0),
        ('O1', 'O2', '#F08080', 2.0)
    ]
    
    # 1. Teken eerst de sensoren op de assen
    mne.viz.plot_sensors(info, show_names=True, axes=ax)
    
    # 2. Forceer jouw styling op de bolletjes
    for collection in ax.collections:
        collection.set_sizes([300])
        collection.set_edgecolor('black')
        collection.set_linewidth(1.5)
    
    # --- DE FIX VOOR DE ZWEVENDE LIJNEN ---
    # 3. Steel de exacte (x,y) coördinaten van de bolletjes direct uit de grafiek!
    sensor_offsets = ax.collections[0].get_offsets()
    ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}
    
    # 4. Teken de lijnen exact tussen deze opgehaalde coördinaten
    for ch1, ch2, color, width in top_connections:
        x1, y1 = ch_pos[ch1]
        x2, y2 = ch_pos[ch2]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)

    ax.set_title("Top 5 Connectiviteits Features\n(CP_FM_dataset - SHAP Ranking)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_4_Connectivity_CP_FM.png", dpi=300)
    plt.close()
    print("  ✓ Figure 4 Opgeslagen")

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roi_map()
    plot_connectivity_map()
    print("\n✅ Alle topografieën succesvol gegenereerd!")