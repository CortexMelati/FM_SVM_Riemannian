"""
=============================================================================
6. RIEMANNIAN MODEL BIAS EVALUATION (Demographic Confounding Check)
=============================================================================
Overview:
    This script evaluates the frozen winning Riemannian model (TSSVM_Xdawn)
    for potential demographic bias across age groups and biological sex on the 
    unseen test set. It uses Subject-Level Majority Voting to match 
    the clinical reality and previous evaluation protocols.

Execution:
    python 6_Riemann_Bias_Evaluation.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib
import mne  # Toegevoegd voor de custom filter!
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin  # Toegevoegd voor uitpakken

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, CP_FM_DIR

# =========================================================================
# BLAUWDRUKKEN: Nodig om de Xdawn pipeline uit te pakken!
# =========================================================================
class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]
# =========================================================================

def evaluate_riemann_bias():
    print("🚀 STARTING STEP 6: RIEMANNIAN DEMOGRAPHIC BIAS EVALUATION (SUBJECT-LEVEL)")

    # 1. LAAD EXPLICIET HET WINNENDE MODEL
    model_name = "model_riemann_Theta_roi_TSSVM_Xdawn.pkl"
    model_path = RIEMANN_DATA_DIR / model_name
    
    if not model_path.exists():
        print(f"🚨 Het winnende model ({model_name}) is niet gevonden!")
        sys.exit()
        
    artifact = joblib.load(model_path)
    pipeline = artifact['model']
    band = artifact['band']
    layout = artifact['layout']
    
    print(f"-> Analyzing bias for: {band.upper()} Band ({layout.upper()} Layout)")
    print(f"-> Loaded champion model: {model_path.name}")

    # 2. LAAD TEST DATA EN METADATA
    y_test_path = RIEMANN_DATA_DIR / "y_test_riemann.npy"
    groups_test_path = RIEMANN_DATA_DIR / "groups_test_riemann.npy"
    tsv_path = CP_FM_DIR / "data" / "participants.tsv"
    
    # We weten dat Xdawn de ruwe data nodig heeft
    X_test_path = RIEMANN_DATA_DIR / "X_test_raw.npy"

    if not (y_test_path.exists() and X_test_path.exists() and groups_test_path.exists() and tsv_path.exists()):
        print("🚨 Essentiële testbestanden of participants.tsv ontbreken.")
        sys.exit()

    y_test = np.load(y_test_path)
    X_test = np.load(X_test_path)
    groups_test = np.load(groups_test_path)
    participants_df = pd.read_csv(tsv_path, sep='\t')

    # 3. GENEREER VOORSPELLINGEN (OP EPOCH NIVEAU)
    print("-> Predicting on Unseen Data (Epochs)...")
    y_pred_epochs = pipeline.predict(X_test)

    # 4. MAJORITY VOTING (SUBJECT-LEVEL AGGREGATION)
    print("-> Aggregating predictions to Subject-Level...")
    df_preds = pd.DataFrame({
        'Subject': groups_test,
        'True_Label': y_test,
        'Pred_Label': y_pred_epochs
    })

    # Groepeer per patiënt en pak de meest voorkomende voorspelling
    df_subject = df_preds.groupby('Subject').agg(
        True_Label=('True_Label', 'first'), 
        Pred_Label=('Pred_Label', lambda x: x.mode()[0])
    ).reset_index()
    
    df_subject['Is_Correct'] = (df_subject['True_Label'] == df_subject['Pred_Label']).astype(int)

    # 5. MERGE MET DEMOGRAFISCHE DATA
    if 'participant_id' in participants_df.columns:
        participants_df['Subject'] = participants_df['participant_id']

    merged_df = pd.merge(df_subject, participants_df[['Subject', 'sex', 'age']], on='Subject', how='inner')
    
    if merged_df.empty:
        print("🚨 Merge mislukt. Controleer of de Subject ID's overeenkomen.")
        sys.exit()

    merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
    merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 40, 55, 100], labels=['<40', '40-55', '>55'])

    print(f"-> Evaluation matrix built successfully across {len(merged_df)} unique subjects.")

    # 6. BEREKEN ACCURAATHEID PER SUBGROEP
    bias_results = []
    
    for sex in merged_df['sex'].unique():
        sub_df = merged_df[merged_df['sex'] == sex]
        acc = accuracy_score(sub_df['True_Label'], sub_df['Pred_Label'])
        bias_results.append({'Factor': 'Sex', 'Subgroup': str(sex).upper(), 'Accuracy': acc, 'Sample_Size': len(sub_df)})

    for age_g in merged_df['age_group'].cat.categories:
        sub_df = merged_df[merged_df['age_group'] == age_g]
        if len(sub_df) > 0:
            acc = accuracy_score(sub_df['True_Label'], sub_df['Pred_Label'])
            bias_results.append({'Factor': 'Age Group', 'Subgroup': age_g, 'Accuracy': acc, 'Sample_Size': len(sub_df)})

    bias_df = pd.DataFrame(bias_results)
    print("\n🏆 RIEMANNIAN DEMOGRAPHIC PERFORMANCE MATRIX:")
    print("-" * 60)
    print(bias_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 60)

    bias_csv_path = RIEMANN_DATA_DIR / f"riemann_demographic_bias_report_{band}.csv"
    bias_df.to_csv(bias_csv_path, index=False, float_format='%.3f')
    
    # 7. VISUALISATIE
    plt.figure(figsize=(8, 5))
    sns.barplot(data=bias_df, x='Subgroup', y='Accuracy', hue='Factor', palette='Oranges_r')
    plt.axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='Chance Level (50%)')
    plt.ylim(0, 1.0)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Demographic Subgroup', fontsize=12)
    plt.title(f'Riemannian Model Fairness Check ({band.upper()} Band)\nArchitecture: TSSVM_Xdawn', fontsize=14, pad=15)
    plt.legend(frameon=True, loc='upper right')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.02, f'{height:.3f}', ha="center", fontsize=10)

    plt.tight_layout()
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RIEMANN_FIGURES_DIR / f"Figure_Riemann_Demographic_Bias_{band}_Xdawn.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Visual fairness chart saved to: riemann_figures/{plot_path.name}\n")

if __name__ == "__main__":
    evaluate_riemann_bias()