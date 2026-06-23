"""
=============================================================================
2. ROI Feature Screening (Top 10 Selection)
=============================================================================
Overview:
    This script filters the dataset strictly to the 9-channel Central ROI 
    and the designated focus band (resulting in 36 potential features). 
    It trains a baseline SVM and uses SHAP to identify the Top 10 most 
    predictive connectivity features within this restricted space.

Execution:
    python 2_SVM_roi_feature_screening.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mne
import sys
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, 
                    FIGURES_DIR, BEST_CHANNELS_EVALUATE, FOCUS_BAND)

# ==========================================
# 1. SETUP AND FILTERING
# ==========================================
print(f"Starting ROI Feature Screening ({FOCUS_BAND.upper()} Band)...")

train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
if not train_path.exists():
    print(f"Error: {train_path.name} not found. Run build_dataset.py first.")
    sys.exit()

train_df = pd.read_csv(train_path)

y_train = train_df['Target'].values
meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_train_full = train_df.drop(columns=[c for c in meta_cols if c in train_df.columns])

# Filter to only include features where BOTH channels are in the ROI and in the Focus Band
roi_features = []
for col in X_train_full.columns:
    if f'({FOCUS_BAND})' in col:
        pair = col.replace(f'({FOCUS_BAND})', '').split('-')
        if pair[0] in BEST_CHANNELS_EVALUATE and pair[1] in BEST_CHANNELS_EVALUATE:
            roi_features.append(col)

X_train_roi = X_train_full[roi_features]
print(f"-> Filtered to {len(roi_features)} features (9 ROI channels, {FOCUS_BAND.upper()} band).")

# ==========================================
# 2. SCALING AND SVM TRAINING
# ==========================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_roi), columns=X_train_roi.columns)

print("-> Training intermediate SVM for SHAP screening...")
screening_svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
screening_svm.fit(X_train_scaled, y_train)

# ==========================================
# 3. SHAP ANALYSIS
# ==========================================
print("-> Calculating SHAP values for the ROI features...")
explainer = shap.KernelExplainer(screening_svm.predict_proba, shap.kmeans(X_train_scaled, 10))
shap_values = explainer.shap_values(X_train_scaled)

if isinstance(shap_values, list):
    shap_values_fm = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_values_fm = np.array(shap_values)[:, :, 1]
else:
    shap_values_fm = np.array(shap_values)

mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)

feature_importance = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

top_10_df = feature_importance.head(10)
print(f"\nTOP 10 FEATURES WITHIN ROI ({FOCUS_BAND.upper()} Band):")
print(top_10_df.to_string(index=False))

# Save the Top 10 list so the mSFFS script can load it
top_10_path = PROCESSED_DATA_DIR / "top_10_roi_features.csv"
top_10_df.to_csv(top_10_path, index=False)
print(f"-> Saved Top 10 features to {top_10_path.name}")

# ==========================================
# 4. PLOT TOPOGRAPHICAL NETWORK
# ==========================================
print("\nGenerating Topographical Map for Top 10 Features...")

# 1. Gebruik uitsluitend de 19 klassieke kanalen (zoals in de paper)
standard_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=standard_19, sfreq=500, ch_types='eeg')
info.set_montage(montage)

fig, ax = plt.subplots(figsize=(8, 8))
mne.viz.plot_sensors(info, show_names=True, axes=ax)

# 2. Clean up sensor styling (witte cirkels met subtiele grijze rand)
for collection in ax.collections:
    collection.set_sizes([600])
    collection.set_facecolor('white')
    collection.set_edgecolor('#cccccc')
    collection.set_linewidth(1.5)

sensor_offsets = ax.collections[0].get_offsets()
ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

max_shap = top_10_df['Mean_Abs_SHAP'].max()

# Schaal de SHAP waarden naar de logica van de paper (grenzen: 0.5, 1.0, 2.0)
# Door maal 2.5 te doen forceer je dat de absolute top-feature als "Roze" wordt geclassificeerd.
top_10_df = top_10_df.copy()
top_10_df['Scaled_Importance'] = (top_10_df['Mean_Abs_SHAP'] / max_shap) * 2.5

for _, row in top_10_df.iterrows():
    feat = row['Feature']
    node1 = feat.split('-')[0]
    node2 = feat.split('-')[1].split('(')[0]
    
    try:
        x_coords = [ch_pos[node1][0], ch_pos[node2][0]]
        y_coords = [ch_pos[node1][1], ch_pos[node2][1]]
        
        # 3. Pas de kleuren en dikte toe (Groen/Bruin/Roze)
        scaled_val = row['Scaled_Importance']
        if scaled_val >= 2.0:
            color, lw = '#FF8C94', 5.0  # Roze
        elif scaled_val >= 1.0:
            color, lw = '#8B4513', 3.5  # Bruin
        else:
            color, lw = '#228B22', 2.0  # Groen
            
        # Teken de lijn met zorder=0 zodat hij mooi onder de sensoren doorloopt
        ax.plot(x_coords, y_coords, color=color, linewidth=lw, alpha=0.9, zorder=0)
    except KeyError:
        pass # Negeer kanalen die niet in de 19-kanaals standaard vallen

ax.set_title(f"Top 10 Connectivity Features within ROI\n({FOCUS_BAND.upper()} Band - SHAP Importance)", fontsize=14, pad=20)
plt.tight_layout()

# Sla op met een transparante achtergrond voor in je LaTeX document
plot_path = FIGURES_DIR / f"Figure_Intermediate_Top10_ROI_{FOCUS_BAND}.png"
plt.savefig(plot_path, dpi=300, transparent=True)
plt.close()

print(f"-> Intermediate plot saved to {plot_path.name}")