"""
=============================================================================
TOPOGRAPHICAL FEATURE VISUALIZATION
=============================================================================
Overview:
    Generates topographical brain maps based on the central config.
    Figure 2: The 9-channel Region of Interest (ROI) (Li et al., 2026).
    Figure 4: Your empirical top 5 connectivity features (CP_FM_dataset).
    
Execution:
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
# 1. MNE SETUP (Linked to config.py)
# ==========================================
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=CHANNELS_1020, sfreq=500, ch_types='eeg')
info.set_montage(montage)

# ==========================================
# 2. FUNCTION: PLOT ROI (Figure 2)
# ==========================================
def plot_roi_map():
    print("Generating ROI Topography (Figure 2)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # The 9 central sensors (Li et al., 2026)
    roi_sensors = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    colors = ['#FFD700' if ch in roi_sensors else '#E0E0E0' for ch in info.ch_names]
    
    mne.viz.plot_sensors(
        info, show_names=True, axes=ax,
        ch_groups=[[info.ch_names.index(ch)] for ch in info.ch_names],
        cmap=plt.matplotlib.colors.ListedColormap(colors)
    )
    
    # Override MNE with Matplotlib for guaranteed styling
    for collection in ax.collections:
        collection.set_facecolor(colors)
        collection.set_edgecolor('black')
        collection.set_sizes([400])  # Fix for size
        collection.set_linewidth(2)  # Fix for edge width

    ax.set_title("Region of Interest (ROI)\n9 Central Sensors", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_2_ROI_Sensors.png", dpi=300)
    plt.close()
    print("  -> Figure 2 Saved")

# ==========================================
# 3. FUNCTION: PLOT CONNECTIVITY (Figure 4)
# ==========================================
def plot_connectivity_map():
    print("\nGenerating Connectivity Topography (Your SHAP Top 5)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Your top 5 Beta-band features
    top_connections = [
        ('Cz', 'Pz', '#8B0000', 4.0), 
        ('P3', 'O1', '#B22222', 3.5),
        ('P7', 'Pz', '#B22222', 3.5),
        ('T7', 'C3', '#CD5C5C', 3.0),
        ('O1', 'O2', '#F08080', 2.0)
    ]
    
    # 1. First draw the sensors on the axes
    mne.viz.plot_sensors(info, show_names=True, axes=ax)
    
    # 2. Force your styling on the scatter points
    for collection in ax.collections:
        collection.set_sizes([300])
        collection.set_edgecolor('black')
        collection.set_linewidth(1.5)
    
    # --- THE FIX FOR FLOATING LINES ---
    # 3. Extract the exact (x,y) coordinates of the scatter points directly from the plot!
    sensor_offsets = ax.collections[0].get_offsets()
    ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}
    
    # 4. Draw the lines exactly between these retrieved coordinates
    for ch1, ch2, color, width in top_connections:
        x1, y1 = ch_pos[ch1]
        x2, y2 = ch_pos[ch2]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)

    ax.set_title("Top 5 Connectivity Features\n(CP_FM_dataset - SHAP Ranking)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_4_Connectivity_CP_FM.png", dpi=300)
    plt.close()
    print("  -> Figure 4 Saved")

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roi_map()
    plot_connectivity_map()
    print("\nAll topographies generated successfully!")