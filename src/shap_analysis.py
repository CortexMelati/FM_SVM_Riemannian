"""
=============================================================================
5. SHAP ANALYSIS PIPELINE (Li et al., 2026 Replication)
=============================================================================
Overview:
    This script opens the frozen SVM model artifacts and the test dataset 
    to calculate SHapley Additive exPlanations (SHAP) values.
    
    It generates three figures replicating the paper:
    1. Mean Absolute SHAP values (Bar plot - Feature Importance, Fig 6A)
    2. SHAP values summary (Bee swarm plot - Impact on model output, Fig 6B)
    3. SVM Top-5 Connectivity Network (Topographical Map, Fig 4)

Execution:
    python shap_analysis.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mne
from pathlib import Path
import sys
import joblib

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RESULTS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, SVM_FIGURES_DIR
from config import USE_ROI, PREFIX

def plot_svm_network_map(shap_values_fm, X_test_final, target_band='gamma'):
    """
    Replicates Figure 4 from Li et al. (Topographical Network Map).
    Maps the top 5 connectivity features based on their mean absolute SHAP values
    onto an MNE 10-20 topographical map.
    """
    print(f"\nGenerating Topographical SVM Network Map (Fig 4) for {target_band.upper()} band...")
    
    # 1. Calculate absolute SHAP importance per feature
    feature_names = X_test_final.columns.tolist()
    mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)
    
    # Create DataFrame and isolate the top 5 features
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False).head(5)
    
    # Extract channel pairs (assumes format like "Fz-Cz(gamma)")
    shap_df['Node1'] = shap_df['Feature'].apply(lambda x: x.split('-')[0])
    shap_df['Node2'] = shap_df['Feature'].apply(lambda x: x.split('-')[1].split('(')[0])
    
    # 2. Setup MNE Topography using the standard 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types='eeg')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_sensors(info, show_names=True, axes=ax)
    
    # Clean up sensor styling (light gray dots, similar to the paper)
    for collection in ax.collections:
        collection.set_sizes([150])
        collection.set_color('#cccccc')
        
    sensor_offsets = ax.collections[0].get_offsets()
    ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

    # 3. Draw connections based on SHAP Importance
    max_importance = shap_df['Importance'].max()
    
    for _, row in shap_df.iterrows():
        try:
            x_coords = [ch_pos[row['Node1']][0], ch_pos[row['Node2']][0]]
            y_coords = [ch_pos[row['Node1']][1], ch_pos[row['Node2']][1]]
            
            # Normalize line width relative to the highest SHAP importance
            line_width = (row['Importance'] / max_importance) * 6.0
            
            # Paper uses red/purple hues for connectivity strength visualization
            ax.plot(x_coords, y_coords, color='#d62728', linewidth=line_width, zorder=1, alpha=0.8)
        except KeyError as e:
            print(f"  Warning: Channel {e} not found in the standard montage map.")

    ax.set_title(f"SVM Top-5 Connectivity Network\n({target_band.upper()} Band - SHAP Importance)", fontsize=16, pad=20)
    plt.tight_layout()
    
    save_path = SVM_FIGURES_DIR / f"SVM_network_map_{target_band.lower()}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> SVM Network Map (Fig 4 Replication) saved to: {save_path.name}")


def run_shap_analysis(target_band='gamma'): # Amend to the band needed? 
    print(f"\nStarting SHAP Analysis for the {target_band.upper()} band...")
    
    # 1. Load the frozen model artifact using centralized config logic
    model_path = PROCESSED_DATA_DIR / f"saved_model_{PREFIX}{target_band}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Error: Model {model_path.name} not found. Run train_svm.py first.")
        
    artifact = joblib.load(model_path)
    final_svm = artifact['model']
    scaler = artifact['scaler']
    roi_features = artifact['roi_features']
    selected_features = artifact['selected_features']
    
    print(f"  -> Frozen model loaded with {len(selected_features)} mSFFS selected features.")

    # 2. Load the test dataset (SHAP is exclusively evaluated on unseen data)
    test_path = PROCESSED_DATA_DIR / "final_dataset_test.csv"
    test_df = pd.read_csv(test_path)
    
    # Filter and scale strictly matching the training pipeline
    X_test_roi = test_df[roi_features]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_roi), columns=roi_features)
    X_test_final = X_test_scaled[selected_features]

    # 3. Initialize SHAP KernelExplainer for SVM (RBF)
    print("  -> Calculating SHAP values (This may take a moment for RBF kernels)...")
    explainer = shap.KernelExplainer(final_svm.predict_proba, X_test_final)
    shap_values = explainer.shap_values(X_test_final)
    
    # Safely handles both list and 3D-array outputs to a flat 2D matrix
    if isinstance(shap_values, list):
        shap_values_fm = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_fm = shap_values[:, :, 1]
    else:
        shap_values_fm = shap_values

    # 4. Generate & Save Plots
    print("  -> Generating SHAP visualizations...")
    features_display = X_test_final.columns.tolist()
    
    # A. Bar Plot (Mean Absolute SHAP) - Replicates Fig 6A
    plt.figure(figsize=(10, 8)) 
    shap.summary_plot(shap_values_fm, X_test_final, plot_type="bar", 
                      feature_names=features_display, show=False)
    plt.title(f"{target_band.upper()} Band - Mean Absolute SHAP Values (Feature Importance)", fontsize=14, pad=15)
    plt.xlabel("Mean |SHAP value| (Impact on model output)", fontsize=12)
    plt.tight_layout()
    plt.savefig(SVM_FIGURES_DIR / f"SHAP_bar_plot_{target_band}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # B. Summary Bee Swarm Plot - Replicates Fig 6B
    plt.figure(figsize=(10, 8)) 
    shap.summary_plot(shap_values_fm, X_test_final, 
                      feature_names=features_display, show=False)
    plt.title(f"{target_band.upper()} Band - SHAP Values Summary", fontsize=14, pad=15)
    plt.xlabel("SHAP value (Impact on specific prediction)", fontsize=12)
    plt.tight_layout()
    plt.savefig(SVM_FIGURES_DIR / f"SHAP_summary_plot_{target_band}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # C. Network Map - Replicates Fig 4
    plot_svm_network_map(shap_values_fm, X_test_final, target_band)

    print(f"SHAP analysis complete. Plots saved to {SVM_FIGURES_DIR.name}/")


if __name__ == "__main__":
    # =========================================================================
    # EXECUTION SWITCH
    # Set to True to run all 5 bands automatically, or False to only run one.
    # =========================================================================
    RUN_ALL_BANDS = False 
    TARGET_BAND = 'gamma' # Used if RUN_ALL_BANDS is False

    if RUN_ALL_BANDS:
        all_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        for band in all_bands:
            try:
                run_shap_analysis(target_band=band)
            except Exception as e:
                print(f"  Warning: Could not complete SHAP analysis for the {band.upper()} band. Error: {e}")
    else:
        run_shap_analysis(target_band=TARGET_BAND)