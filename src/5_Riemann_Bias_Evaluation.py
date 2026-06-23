"""
=============================================================================
5. RIEMANNIAN MODEL BIAS EVALUATION (Demographic Confounding Check)
=============================================================================
Overview:
    This script evaluates the frozen Riemannian model (Delta TSSVM) for 
    potential demographic bias across age groups and biological sex on the 
    unseen test set. It mirrors the evaluation protocol applied to the SVM.

Execution:
    python 5_Riemann_Bias_Evaluation.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib
from sklearn.metrics import accuracy_score

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, SVM_DATA_DIR, RIEMANN_FIGURES_DIR, CP_FM_DIR

def evaluate_riemann_bias():
    print("🚀 STARTING STEP 5: RIEMANNIAN DEMOGRAPHIC BIAS EVALUATION")

    # 1. LAAD HET BEVROREN MODEL
    model_files = list(SVM_DATA_DIR.glob("model_riemann_*.pkl"))
    if not model_files:
        print("🚨 Geen bevroren Riemannian model gevonden in svm_data/.")
        sys.exit()
        
    model_path = model_files[-1]
    artifact = joblib.load(model_path)
    pipeline = artifact['model']
    band = artifact['band']
    layout = artifact['layout']
    
    print(f"-> Analyzing bias for: {band.upper()} Band ({layout.upper()} Layout)")

    # 2. LAAD TEST DATA EN METADATA
    y_test_path = RIEMANN_DATA_DIR / "y_test_riemann.npy"
    covs_test_path = RIEMANN_DATA_DIR / f"covs_test_{band}_{layout}.npy"
    groups_test_path = RIEMANN_DATA_DIR / "groups_test_riemann.npy"
    tsv_path = CP_FM_DIR / "data" / "participants.tsv"
    
    if not (y_test_path.exists() and covs_test_path.exists() and groups_test_path.exists() and tsv_path.exists()):
        print("🚨 Essentiële testbestanden of participants.tsv ontbreken.")
        sys.exit()

    y_test = np.load(y_test_path)
    X_covs_test = np.load(covs_test_path)
    groups_test = np.load(groups_test_path)
    participants_df = pd.read_csv(tsv_path, sep='\t')

    # 3. GENEREER VOORSPELLINGEN
    y_pred = pipeline.predict(X_covs_test)

    # Bouw een tijdelijke dataframe met de testsegmenten
    test_results_df = pd.DataFrame({
        'Subject': groups_test,
        'True_Label': y_test,
        'Pred_Label': y_pred,
        'Is_Correct': (y_test == y_pred).astype(int)
    })

    # Fix participant_id mapping conform eerdere scripts
    if 'participant_id' in participants_df.columns:
        participants_df['Subject'] = participants_df['participant_id']

    # Merge met demografische data
    merged_df = pd.merge(test_results_df, participants_df[['Subject', 'sex', 'age']], on='Subject', how='inner')
    
    if merged_df.empty:
        print("🚨 Merge mislukt. Controleer of de Subject ID's overeenkomen.")
        sys.exit()

    # Formatteer leeftijdscategorieën
    merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
    merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 40, 55, 100], labels=['<40', '40-55', '>55'])

    print(f"-> Evaluation matrix built successfully across {len(merged_df)} test segments.")

    # 4. BEREKEN ACCURAATHEID PER SUBGROEP
    bias_results = []
    
    # Sex Subgroups
    for sex in merged_df['sex'].unique():
        sub_df = merged_df[merged_df['sex'] == sex]
        acc = accuracy_score(sub_df['True_Label'], sub_df['Pred_Label'])
        bias_results.append({'Factor': 'Sex', 'Subgroup': str(sex).upper(), 'Accuracy': acc, 'Sample_Size': len(sub_df)})

    # Age Subgroups
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

    # Exporteer rapport naar csv (max 3 decimalen voor LaTeX integratie)
    bias_csv_path = RIEMANN_DATA_DIR / f"riemann_demographic_bias_report_{band}.csv"
    bias_df.to_csv(bias_csv_path, index=False, float_format='%.3f')
    print(f"-> Bias matrix report saved to: riemann_data/{bias_csv_path.name}")

    # 5. VISUALISATIE (Bar plot conform Figuur 7 logic)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=bias_df, x='Subgroup', y='Accuracy', hue='Factor', palette='Oranges_r')
    plt.axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='Chance Level (50%)')
    plt.ylim(0, 1.0)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Demographic Subgroup', fontsize=12)
    plt.title(f'Riemannian Model Fairness Check ({band.upper()} Band)', fontsize=14, pad=15)
    plt.legend(frameon=True, loc='upper right')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Getallen boven de bars zetten
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.02, f'{height:.3f}', ha="center", fontsize=10)

    plt.tight_layout()
    plot_path = RIEMANN_FIGURES_DIR / f"Figure_Riemann_Demographic_Bias_{band}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Visual fairness chart saved to: riemann_figures/{plot_path.name}\n")

if __name__ == "__main__":
    evaluate_riemann_bias()