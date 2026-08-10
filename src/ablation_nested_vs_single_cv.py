"""
=============================================================================
ABLATION STUDY: Nested CV vs Single CV (SVM)
=============================================================================
Dit script toetst empirisch de methodologische robuustheid van de pijplijn.
Het vergelijkt de over-optimistische 'Single CV' methode (vaak gebruikt in 
standaard literatuur zoals Li et al.) met de strikte 'Nested CV' methode 
om hyperparameter-lekkage aan te tonen.

Output:
    - Een tekstueel rapport (.txt) met de exacte scores per band.
    - Een publicatie-klare staafdiagram (.png) van de resultaten.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, SVM_DATA_DIR, SVM_FIGURES_DIR, RANDOM_STATE, BANDS

def run_cv_ablation_study():
    print("🚀 STARTING ABLATION STUDY: NESTED VS SINGLE CV ACROSS ALL BANDS\n" + "="*70)

    train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
    if not train_path.exists():
        sys.exit("🚨 Error: Training dataset niet gevonden.")

    train_df = pd.read_csv(train_path)
    y = train_df['Target'].values
    groups = train_df['Subject'].values

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': np.logspace(-4, 1.5, 20),
        'class_weight': ['balanced']
    }
    base_svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    
    cv_inner = StratifiedGroupKFold(n_splits=5)
    cv_outer = StratifiedGroupKFold(n_splits=5)

    results = []

    for band_name in BANDS.keys():
        band_name_lower = band_name.lower()
        features_path = SVM_DATA_DIR / f"final_msffs_selected_features_{band_name_lower}.csv"
        
        if not features_path.exists():
            print(f"⚠️ Skipping {band_name.upper()}: Geen mSFFS features gevonden.")
            continue

        print(f"📡 PROCESSING BAND: {band_name.upper()}")
        selected_features = pd.read_csv(features_path)['Selected_Features'].tolist()
        
        X = train_df[selected_features]
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)

        # --- 1. Single CV (Biased) ---
        clf_single = GridSearchCV(
            estimator=base_svm, param_grid=param_grid, cv=cv_inner, 
            scoring='balanced_accuracy', n_jobs=-1
        )
        clf_single.fit(X_scaled, y, groups=groups)
        single_cv_score = clf_single.best_score_
        
        # --- 2. Nested CV (Unbiased - Handmatige Outer Loop om 0-d array fout te voorkomen) ---
        nested_scores = []
        for train_idx, test_idx in cv_outer.split(X_scaled, y, groups=groups):
            X_tr, X_te = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            g_tr = groups[train_idx]

            # Inner GridSearch voor hyperparameter tuning op de train-folds
            inner_gs = GridSearchCV(
                estimator=base_svm, param_grid=param_grid, cv=cv_inner, 
                scoring='balanced_accuracy', n_jobs=-1
            )
            inner_gs.fit(X_tr, y_tr, groups=g_tr)
            
            # Evalueer het beste model op de ongeziene outer test fold
            best_model = inner_gs.best_estimator_
            y_pred = best_model.predict(X_te)
            
            from sklearn.metrics import balanced_accuracy_score
            fold_acc = balanced_accuracy_score(y_te, y_pred)
            nested_scores.append(fold_acc)

        nested_cv_score_mean = np.mean(nested_scores)
        bias = single_cv_score - nested_cv_score_mean
        
        print(f"   Single CV: {single_cv_score:.4f} | Nested CV: {nested_cv_score_mean:.4f} | Overestimation: {bias:.4f}\n")
        
        results.append({
            'Band': band_name.upper(),
            'Single CV (Biased)': single_cv_score,
            'Nested CV (Unbiased)': nested_cv_score_mean,
            'Bias (Overestimation)': bias
        })

    if not results:
        sys.exit("Geen resultaten om te plotten. Zorg dat Script 3 is gedraaid.")

    df_results = pd.DataFrame(results)

    # =============================================================================
    # OUTPUT 1: TEXT REPORT (.txt)
    # =============================================================================
    report_path = SVM_DATA_DIR / "ablation_study_cv_report.txt"
    with open(report_path, "w") as f:
        f.write("=====================================================================\n")
        f.write(" ABLATION STUDY: NESTED VS SINGLE CROSS-VALIDATION\n")
        f.write("=====================================================================\n")
        f.write(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n=====================================================================\n")
    print(f"✅ Tekstrapport opgeslagen in: {report_path}")

    # =============================================================================
    # OUTPUT 2: VISUALIZATION (.png)
    # =============================================================================
    df_melted = df_results.melt(id_vars='Band', value_vars=['Single CV (Biased)', 'Nested CV (Unbiased)'], 
                                var_name='Method', value_name='Balanced Accuracy')

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(x='Band', y='Balanced Accuracy', hue='Method', data=df_melted, 
                     palette=['#d62728', '#5c8cbc'], edgecolor='black', alpha=0.8)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=10)

    plt.title('Methodological Ablation: Single CV vs. Nested CV Hyperparameter Tuning', fontsize=14, pad=20)
    plt.ylabel('Training Balanced Accuracy', fontsize=12)
    plt.xlabel('EEG Frequency Band', fontsize=12)
    plt.ylim(0.4, max(df_melted['Balanced Accuracy']) + 0.1)
    plt.legend(title='Validation Methodology', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    SVM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = SVM_FIGURES_DIR / "Figure_Ablation_CV_Comparison.png"
    plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figuur opgeslagen in: {plot_path}")
    print("="*70)

if __name__ == "__main__":
    run_cv_ablation_study()