"""
=============================================================================
5. SVM Model Evaluation & Interpretation
=============================================================================
Overview:
    This script performs the final, unbiased evaluation of the trained SVM model.
    It opens the strictly isolated 20% test dataset (the "vault") and applies
    the frozen model artifacts to calculate true generalization metrics.
    
    It also generates the final publication figures:
    - Confusion Matrix
    - Figure 4 (Topographical Biomarker Map based on SHAP)
    - Figure 5 (t-SNE Data Distribution)
    - Figure 6A & 6B (SHAP Value Summaries)

Execution:
    python 5_SVM_model_evaluation.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mne
from pathlib import Path
import sys
import joblib

from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score)

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (PROCESSED_DATA_DIR, SVM_FIGURES_DIR, SVM_DATA_DIR, FOCUS_BAND, CP_FM_DIR)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_weight = np.sum(in_bin) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

def plot_svm_network_map(shap_values_fm, feature_names, target_band):
    print(f"\n-> Generating Topographical SVM Network Map (Fig 4) for {target_band.upper()} band...")
    
    mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False)
    
    shap_df['Node1'] = shap_df['Feature'].apply(lambda x: x.split('-')[0])
    shap_df['Node2'] = shap_df['Feature'].apply(lambda x: x.split('-')[1].split('(')[0])
    
    # 1. Gebruik uitsluitend de 19 klassieke kanalen (zoals in de paper)
    standard_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=standard_19, sfreq=500, ch_types='eeg')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Teken de omtrek van het hoofd, maar pas de sensor-stijl aan
    mne.viz.plot_sensors(info, show_names=True, axes=ax)
    
    # Maak de sensoren wit met een dunne grijze rand, net als in de paper
    for collection in ax.collections:
        collection.set_sizes([600]) # Grotere cirkels
        collection.set_facecolor('white')
        collection.set_edgecolor('#cccccc')
        collection.set_linewidth(1.5)
        
    sensor_offsets = ax.collections[0].get_offsets()
    ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

    # 2. Vertaal jouw SHAP waarden naar de Kleur-logica van de paper
    max_importance = shap_df['Importance'].max()
    # We schalen de max_importance naar 2.5 om de grenzen van de paper (0.5, 1.0, 2.0) te matchen
    shap_df['Scaled_Importance'] = (shap_df['Importance'] / max_importance) * 2.5
    
    for _, row in shap_df.iterrows():
        try:
            x_coords = [ch_pos[row['Node1']][0], ch_pos[row['Node2']][0]]
            y_coords = [ch_pos[row['Node1']][1], ch_pos[row['Node2']][1]]
            
            # Bepaal kleur en dikte gebaseerd op de geschaalde waarde
            scaled_val = row['Scaled_Importance']
            if scaled_val >= 2.0:
                color, lw = '#FF8C94', 5.0 # Roze/Lichtrood
            elif scaled_val >= 1.0:
                color, lw = '#8B4513', 3.5 # Bruin
            else:
                color, lw = '#228B22', 2.0 # Groen
                
            ax.plot(x_coords, y_coords, color=color, linewidth=lw, zorder=0, alpha=0.9)
        except KeyError:
            # Negeer kanalen die toevallig niet in de 19-channel map zitten
            pass

    ax.set_title(f"Connectivity features associated with fibromyalgia\n({target_band.upper()} Band - SHAP Importance)", fontsize=14, pad=20)
    plt.tight_layout()
    plot_path = SVM_FIGURES_DIR / f"Figure_4_SVM_network_map_{target_band}.png"
    plt.savefig(plot_path, dpi=300, transparent=True) # Transparent background makes it look cleaner
    plt.close()

# =============================================================================
# 1. LOAD TEST DATA & FILTER
# =============================================================================
print(f"Opening the Vault: Loading Unseen Test Data ({FOCUS_BAND.upper()} Band)...")
test_path = PROCESSED_DATA_DIR / "final_dataset_test.csv"
test_df = pd.read_csv(test_path)

# Filter for Eyes Closed (EC) only to match training!
if 'Condition' in test_df.columns:
    test_df = test_df[test_df['Condition'] == 'EC'].copy()
    print(f"-> Filtered Test Set to EC only: {len(test_df)} segments remain.")

y_test = test_df['Target'].values

# =============================================================================
# 2. LOAD FROZEN MODEL ARTIFACT
# =============================================================================
model_path = SVM_DATA_DIR / f"saved_model_{FOCUS_BAND}.pkl"

if not model_path.exists():
    print(f"Error: Model {model_path.name} not found. Run Script 4 first.")
    sys.exit()
    
artifact = joblib.load(model_path)
final_svm = artifact['model']
scaler = artifact['scaler']
selected_features = artifact['features']

print(f"-> Loaded frozen model trained on {len(selected_features)} features.")

# Prepare exactly as during training
X_test_final = test_df[selected_features]
X_test_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=selected_features)

# =============================================================================
# 3. EXTERNAL VALIDATION (Metrics & Confusion Matrix)
# =============================================================================
print("\n-> Predicting on Unseen Data...")
y_pred = final_svm.predict(X_test_scaled)
y_prob = final_svm.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
auprc = average_precision_score(y_test, y_prob) 
brier = brier_score_loss(y_test, y_prob)
ece = expected_calibration_error(y_test, y_prob)

# --- NIEUW: Print én Sla op als .txt Rapport ---
report_text = (
    f"====================================================\n"
    f" FINAL TEST SET METRICS - {FOCUS_BAND.upper()} BAND \n"
    f"====================================================\n"
    f"Accuracy:        {acc:.4f}\n"
    f"Precision:       {prec:.4f}\n"
    f"Recall:          {rec:.4f}\n"
    f"ROC-AUC:         {auc:.4f}\n"
    f"AUPRC:           {auprc:.4f}\n"
    f"Brier Score:     {brier:.4f}\n"
    f"ECE:             {ece:.4f}\n"
    f"====================================================\n"
)

print(f"\n{report_text}")

report_path = SVM_DATA_DIR / f"final_test_metrics_report_{FOCUS_BAND}.txt"
with open(report_path, "w") as f:
    f.write(report_text)
print(f"-> Metrics report permanently saved to: svm_data/{report_path.name}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy (0)', 'Fibro (1)'], 
            yticklabels=['Healthy (0)', 'Fibro (1)'],
            annot_kws={"size": 16})
plt.title(f'{FOCUS_BAND.upper()} Band FINAL Validation\n(Accuracy: {acc:.2%})', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(SVM_FIGURES_DIR / f"final_confusion_matrix_{FOCUS_BAND}.png", dpi=300)
plt.close()

# =============================================================================
# 4. GENERATE FIGURE 5 (t-SNE Projection)
# =============================================================================
print(f"\n-> Generating t-SNE data distribution (Fig 5)...")
tsne = TSNE(n_components=2, perplexity=min(30, len(X_test_scaled)-1), random_state=42)
X_tsne = tsne.fit_transform(X_test_scaled)

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(
    x=X_tsne[:, 0], y=X_tsne[:, 1], 
    hue=y_test, 
    palette={0: '#5c8cbc', 1: '#d62728'},
    s=80, alpha=0.8, edgecolor='white'
)
plt.title(f"Data Distribution of Selected Connectivity Features\n({FOCUS_BAND.upper()} Band - t-SNE Projection)", fontsize=14, pad=15)
plt.xlabel("t-SNE Dimension 1", fontsize=11)
plt.ylabel("t-SNE Dimension 2", fontsize=11)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=['Healthy Control (HC)', 'Fibromyalgia (FM)'], title='Diagnosis', frameon=True)
plt.tight_layout()
plt.savefig(SVM_FIGURES_DIR / f"Figure_5_tsne_distribution_{FOCUS_BAND}.png", dpi=300)
plt.close()

# =============================================================================
# 5. SHAP ANALYSIS
# =============================================================================
print("\n-> Calculating SHAP values for interpretability...")
explainer = shap.KernelExplainer(final_svm.predict_proba, shap.kmeans(X_test_scaled, 10))
shap_values = explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):
    shap_values_fm = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_fm = shap_values[:, :, 1]
else:
    shap_values_fm = shap_values

features_display = X_test_scaled.columns.tolist()

# Plot Fig 6A (Bar Plot) with Data Labels
plt.figure(figsize=(10, 8)) 
shap.summary_plot(shap_values_fm, X_test_scaled, plot_type="bar", feature_names=features_display, show=False)

ax = plt.gca()
for patch in ax.patches:
    width = patch.get_width()
    if width > 0: # Zorg dat we alleen de SHAP bars pakken
        # Bepaal het midden van de bar qua hoogte
        y = patch.get_y() + patch.get_height() / 2
        # Zet de tekst nét iets rechts van de bar
        ax.text(width + (width * 0.01), y, f"{width:.4f}", va='center', ha='left', fontsize=11)

# Maak de x-as iets langer zodat de tekst niet van het scherm valt
xlim = ax.get_xlim()
ax.set_xlim(xlim[0], xlim[1] * 1.15)

plt.title(f"{FOCUS_BAND.upper()} Band - Mean Absolute SHAP Values (Feature Importance)", fontsize=14, pad=15)
plt.xlabel("Mean |SHAP value|", fontsize=12)
plt.tight_layout()
plt.savefig(SVM_FIGURES_DIR / f"Figure_6A_SHAP_bar_{FOCUS_BAND}.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot Fig 6B (Summary Plot)
plt.figure(figsize=(10, 8)) 
shap.summary_plot(shap_values_fm, X_test_scaled, feature_names=features_display, show=False)
plt.title(f"{FOCUS_BAND.upper()} Band - SHAP Values Summary", fontsize=14, pad=15)
plt.xlabel("SHAP value (Impact on specific prediction)", fontsize=12)
plt.tight_layout()
plt.savefig(SVM_FIGURES_DIR / f"Figure_6B_SHAP_summary_{FOCUS_BAND}.png", dpi=300, bbox_inches='tight')
plt.close()

plot_svm_network_map(shap_values_fm, features_display, FOCUS_BAND)

print("\nPIPELINE COMPLETE. All evaluation metrics and figures have been generated.")

# =============================================================================
# 6. ALGORITHMIC BIAS EVALUATION (Demographics)
# =============================================================================
print("\n-> Commencing Algorithmic Bias Evaluation on Test Set...")

# 6A. Add predictions to the dataframe
test_df['SVM_Pred'] = y_pred

# 6B. Load Metadata (participants.tsv)
# Zorg dat de pad-verwijzing overeenkomt met hoe het in je config/structuur staat
tsv_path = CP_FM_DIR / "data" / "participants.tsv"

if not tsv_path.exists():
    print(f"  Warning: participants.tsv not found at {tsv_path}. Skipping Bias Evaluation.")
else:
    participants_df = pd.read_csv(tsv_path, sep='\t')

    # FIX: Kopieer de ID's exact, zonder stukjes af te knippen!
    if 'participant_id' in participants_df.columns:
        participants_df['Subject'] = participants_df['participant_id']
        
    # --- DEBUGGING CHECK --- 
    # Dit print even de eerste 3 ID's van beide tabellen, zodat je met je eigen ogen 
    # kunt zien of ze nu perfect matchen voordat ze gemerged worden.
    print("\n  [Debug] First 3 Subjects in Test Set:", test_df['Subject'].unique()[:3])
    print("  [Debug] First 3 Subjects in TSV File:", participants_df['Subject'].unique()[:3])
    
    # Merge on Subject
    merged_df = pd.merge(test_df, participants_df, on='Subject', how='inner')

    if merged_df.empty:
        print("  Warning: Merge failed. The printed subjects above likely still don't match.")
    else:
        print(f"  -> Merge successful! Bias evaluation running on {len(merged_df)} segments.")
        # Handle Age formatting
        merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
        merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 40, 55, 100], labels=['<40', '40-55', '>55'])

        bias_results = []

        def evaluate_subgroup(df_sub, category_name, category_value):
            n_samples = len(df_sub)
            if n_samples == 0: return
            
            y_t = df_sub['Target']
            y_p = df_sub['SVM_Pred']
            
            acc = accuracy_score(y_t, y_p)
            sens = recall_score(y_t, y_p, pos_label=1, zero_division=0)
            spec = recall_score(y_t, y_p, pos_label=0, zero_division=0)
            
            bias_results.append({
                'Category': category_name,
                'Group': category_value,
                'N_Segments': n_samples,
                'Accuracy': round(acc, 4),
                'Sensitivity': round(sens, 4),
                'Specificity': round(spec, 4)
            })

        # Evaluate by SEX
        if 'sex' in merged_df.columns:
            for sex in merged_df['sex'].dropna().unique():
                evaluate_subgroup(merged_df[merged_df['sex'] == sex], 'Sex', sex.upper())

        # Evaluate by AGE GROUP
        if 'age_group' in merged_df.columns:
            for age_grp in merged_df['age_group'].dropna().unique():
                evaluate_subgroup(merged_df[merged_df['age_group'] == age_grp], 'Age', age_grp)

        # Evaluate by STUDY (Cohort)
        if 'study' in merged_df.columns:
            for study in merged_df['study'].dropna().unique():
                evaluate_subgroup(merged_df[merged_df['study'] == study], 'Study/Cohort', study)

        # 6C. Export and Print Results
        if bias_results:
            bias_df = pd.DataFrame(bias_results).sort_values(by=['Category', 'Group'])
            
            print("\n" + "-" * 75)
            print(" DEMOGRAPHIC BIAS REPORT (UNSEEN TEST SET)")
            print("-" * 75)
            print(bias_df.to_string(index=False))
            print("-" * 75)
            
            bias_path = SVM_DATA_DIR / f"svm_algorithmic_bias_report_{FOCUS_BAND}.csv"
            bias_df.to_csv(bias_path, index=False)
            print(f"-> Full bias report saved to: svm_data/{bias_path.name}")

            # 6D. Generate Bias Visualization
            print("-> Generating Algorithmic Bias Visualization...")
            plt.figure(figsize=(10, 6))
            
            # Create a bar plot grouped by demographic category
            sns.barplot(
                data=bias_df, 
                x='Group', 
                y='Accuracy', 
                hue='Category', 
                dodge=False, 
                palette='Set2',
                edgecolor='black'
            )
            
            # Add a red dashed line for the 50% chance level
            plt.axhline(y=0.5, color='#d62728', linestyle='--', alpha=0.8, label='Chance Level (50%)')
            
            plt.title(f"Algorithmic Bias Evaluation\n({FOCUS_BAND.upper()} Band - Test Set Accuracy per Subgroup)", fontsize=14, pad=15)
            plt.ylim(0, 1.05)
            plt.ylabel("Accuracy", fontsize=12)
            plt.xlabel("Demographic Subgroup", fontsize=12)
            
            # Style the legend and axes
            plt.legend(title="Demographic Category", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
            plt.xticks(rotation=45, ha='right')
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            bias_fig_path = SVM_FIGURES_DIR / f"Figure_Bias_Evaluation_{FOCUS_BAND}.png"
            plt.savefig(bias_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"-> Bias visualization saved to: svm_figures/{bias_fig_path.name}")