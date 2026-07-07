"""
=============================================================================
3. SVM Feature Selection (mSFFS)
=============================================================================
Overview:
    This script performs Sequential Forward Floating Selection (mSFFS) on the 
    restricted ROI feature space. It evaluates subsets of increasing sizes 
    (from 1 to 20 features) using Stratified Group K-Fold cross-validation.
    
    we select the top 5 to train on for the SVM model. 
    
    It replicates Figure 3 and exports the definitive list of optimal features.

Execution:
    python 3_SVM_feature_selection_msffs.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, SVM_DATA_DIR,
                    SVM_FIGURES_DIR, BEST_CHANNELS_EVALUATE, FOCUS_BAND, BANDS)

# =============================================================================
# PUBLICATION PLOT FUNCTION
# =============================================================================
def plot_msffs_curve(features_count, train_scores, cv_scores, cv_std, ci_margins, target_band):
    plt.figure(figsize=(12, 6))
    x_axis = np.array(features_count)
    
    # Het schaduwvlak blijft de ±1 SD (label aangepast voor statistische accuraatheid)
    plt.fill_between(x_axis, cv_scores - cv_std, cv_scores + cv_std, 
                     color='#e6eef4', alpha=0.7, label='Standard Deviation (±1 SD)')
    
    plt.plot(x_axis, cv_scores, marker='o', markersize=4, color='#5c8cbc', lw=1.5, label='Cross-validation accuracy')
    plt.plot(x_axis, train_scores, marker='o', markersize=4, color='#fba232', lw=1.5, label='Training accuracy')
    
    # AANGEPAST: We printen nu de CI marge (±) in plaats van de SD
    for i, (tr, cv, ci) in enumerate(zip(train_scores, cv_scores, ci_margins)):
        plt.text(x_axis[i], tr + 0.002, f"{tr:.3f}", color='#fba232', fontsize=9, ha='center', va='bottom')
        # We printen de 95% CI (±) direct onder de CV accuracy
        plt.text(x_axis[i], cv - 0.003, f"{cv:.3f}\n(±{ci:.3f})", color='#5c8cbc', fontsize=8, ha='center', va='top')

    plt.title(f"Classification accuracy scores when searching in ROI ({target_band.upper()} band)", fontsize=12, pad=20)
    plt.xlabel('Number of features used', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.xticks(np.arange(min(features_count), max(features_count)+1, 1.0))
    plt.ylim([min(cv_scores - cv_std) - 0.05, 1.05])
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    
    plot_path = SVM_FIGURES_DIR / f"Figure_3_mSFFS_curve_{target_band}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> mSFFS Learning curve (Figure 3) saved to {plot_path.name}")

# =============================================================================
# 1. LOAD TRAINING DATA & IMPORT SCRIPT 2 FEATURES
# =============================================================================
print(f"Starting mSFFS Feature Selection ({FOCUS_BAND.upper()} Band ROI)...")

train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
train_df = pd.read_csv(train_path)

y_train = train_df['Target'].values
groups_train = train_df['Subject'].values

# A. Load the Top 10 features discovered in Script 2
top_10_path = PROCESSED_DATA_DIR / f"top_10_roi_features_{FOCUS_BAND}.csv"
if not top_10_path.exists():
    print(f"Error: Could not find {top_10_path.name}. Please run Script 2 first.")
    sys.exit()
    
top_10_features = pd.read_csv(top_10_path)['Feature'].tolist()

# B. Get all potential ROI features to fill up the remaining 10 spots
meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_train_full = train_df.drop(columns=[c for c in meta_cols if c in train_df.columns])

all_roi_features = []
for col in X_train_full.columns:
    if f'({FOCUS_BAND})' in col:
        pair = col.replace(f'({FOCUS_BAND})', '').split('-')
        if pair[0] in BEST_CHANNELS_EVALUATE and pair[1] in BEST_CHANNELS_EVALUATE:
            all_roi_features.append(col)

# C. Isolate exactly 20 features (Top 10 + 10 additional ROI features)
remaining_roi = [f for f in all_roi_features if f not in top_10_features]
pool_of_20_features = top_10_features + remaining_roi[:10]

X_train_roi = X_train_full[pool_of_20_features]

print(f"-> Successfully loaded Top 10 features from Script 2.")
print(f"-> Added 10 remaining ROI features to create the search pool.")
print(f"-> Search space strictly constrained to {len(X_train_roi.columns)} features.")

# =============================================================================
# 2. SCALING & REPEATED STRATIFIED GROUP K-FOLD (5 Folds x 10 Repeats)
# =============================================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_roi), columns=X_train_roi.columns)

cv_splits = []
# We herhalen de 5-fold splitsing 10 keer met een verschuivende random state
for seed_offset in range(10): 
    cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + seed_offset)
    cv_splits.extend(list(cv_strategy.split(X_train_scaled, y_train, groups=groups_train)))

print(f"-> Created {len(cv_splits)} cross-validation folds (10 repeats of 5-fold CV).")


# =============================================================================
# 3. mSFFS ALGORITHM
# =============================================================================
print("-> Running mSFFS algorithm (Evaluating subsets from 1 to 20 features)...")
# Amend to settings from script 4 if needed 
base_svm = SVC(
    kernel='rbf', 
    # C=10, 
    gamma='scale', 
    class_weight='balanced', 
    random_state=RANDOM_STATE
)

sfs = SFS(
    base_svm, 
    k_features=(1, 20),
    forward=True,
    floating=True,
    scoring='balanced_accuracy', 
    cv=cv_splits,          
    n_jobs=-1              
)

sfs = sfs.fit(X_train_scaled, y_train)
metric_dict = sfs.get_metric_dict()


# # =============================================================================
# # 3. mSFFS ALGORITHM (Methodologically Aligned with Target Model)
# # =============================================================================
# print("-> Running mSFFS algorithm (Evaluating subsets from 1 to 20 features)...")

# # Definieer de optimale parameters die in Script 4 zijn gevonden (Kohavi & John, 1997)
# optimal_params = {
#     'gamma': {'C': 10, 'gamma': 0.2976},
#     'beta':  {'C': 1, 'gamma': 0.2976},
#     'default': {'C': 1.0, 'gamma': 'scale'} # Fallback voor andere banden
# }

# band_key = FOCUS_BAND.lower()
# current_params = optimal_params.get(band_key, optimal_params['default'])

# print(f"-> Applying optimal hyperparameter alignment for {band_key.upper()}: C={current_params['C']}, gamma={current_params['gamma']}")

# base_svm = SVC(
#     kernel='rbf', 
#     C=current_params['C'], 
#     gamma=current_params['gamma'], 
#     class_weight='balanced', # Cruciaal voor imbalanced segmenten
#     random_state=RANDOM_STATE
# )

# sfs = SFS(
#     base_svm, 
#     k_features=(1, 20),
#     forward=True,
#     floating=True,
#     scoring='balanced_accuracy', # Uitgelijnd met evaluatie-metric
#     cv=cv_splits,          
#     n_jobs=-1              
# )

# sfs = sfs.fit(X_train_scaled, y_train)
# metric_dict = sfs.get_metric_dict()

# =============================================================================
# 4. STATISTICAL EVALUATION (Finding the Maximum Performance)
# =============================================================================
print("\n" + "="*85)
print(f"{'k':<3} | {'Added/Changed Feature':<25} | {'Mean Acc':<9} | {'Std Dev':<8} | {'95% CI'}")
print("-" * 85)

# AANGEPAST: ci_margins lijst toegevoegd om door te geven aan de plot
f_counts, cv_scores, cv_stds, tr_scores, ci_margins = [], [], [], [], []
stats_results = [] 
max_acc = 0
optimal_k = 1

for k in range(1, len(metric_dict) + 1):
    if k not in metric_dict: continue
    
    # Bepaal welke feature bij deze stap is toegevoegd (of verwijderd via floating)
    if k == 1:
        step_feature = f"+{metric_dict[k]['feature_names'][0]}"
    else:
        prev_set = set(metric_dict[k-1]['feature_names'])
        curr_set = set(metric_dict[k]['feature_names'])
        added = curr_set - prev_set
        removed = prev_set - curr_set
        
        changes = []
        if added: changes.append(f"+{', '.join(added)}")
        if removed: changes.append(f"-{', '.join(removed)}")
        step_feature = " ".join(changes)

    # Extract fold scores
    fold_scores = metric_dict[k]['cv_scores']
    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    n_folds = 50
    
    # Calculate 95% Confidence Interval
    ci_margin = 1.96 * (std_acc / np.sqrt(n_folds))
    ci_lower, ci_upper = mean_acc - ci_margin, mean_acc + ci_margin
    ci_str = f"[{ci_lower:.4f} - {ci_upper:.4f}]"

    # Print inclusief de specifieke feature
    print(f"{k:<3} | {step_feature:<25} | {mean_acc:.4f}   | {std_acc:.4f}   | {ci_str}")
    
    # Sla de rij op voor de CSV export
    stats_results.append({
        'k': k,
        'Feature_Change': step_feature,
        'Current_Subset': ", ".join(metric_dict[k]['feature_names']),
        'Mean_Acc': round(mean_acc, 4),
        'Std_Dev': round(std_acc, 4),
        '95_percent_CI': ci_str
    })
    
    # Data verzamelen voor de plot
    f_counts.append(k)
    cv_scores.append(mean_acc)
    cv_stds.append(std_acc)
    ci_margins.append(ci_margin) # Sla de marge (bijv. 0.012) op voor de tekst in de plot
    
    subset = list(metric_dict[k]['feature_names'])
    base_svm.fit(X_train_scaled[subset], y_train)
    tr_scores.append(base_svm.score(X_train_scaled[subset], y_train))
    
    if mean_acc > max_acc:
        max_acc = mean_acc
        optimal_k = k

print("=" * 85)

final_features = list(metric_dict[optimal_k]['feature_names'])
final_acc = metric_dict[optimal_k]['avg_score']

print(f"\nOPTIMAL SUBSET DISCOVERED AT k={optimal_k}:")
print(f"-> Selected based on absolute maximum cross-validation performance.")
print(f"-> Selected Validation Accuracy: {final_acc:.4f}")
print(f"-> Selected Biomarkers: {', '.join(final_features)}")

# =============================================================================
# 5. PLOT AND SAVE EXPORTS
# =============================================================================
plot_msffs_curve(f_counts, np.array(tr_scores), np.array(cv_scores), np.array(cv_stds), np.array(ci_margins), FOCUS_BAND)

# 5A. Save the statistical table
stats_df = pd.DataFrame(stats_results)
stats_path = SVM_DATA_DIR / f"msffs_statistical_summary_{FOCUS_BAND}.csv"
stats_df.to_csv(stats_path, index=False)
print(f"-> Statistical summary table saved to svm_data/{stats_path.name}")

# 5B. Save the optimal feature list for Script 4
output_df = pd.DataFrame({'Selected_Features': final_features})
output_path = SVM_DATA_DIR / f"final_msffs_selected_features_{FOCUS_BAND}.csv"
output_df.to_csv(output_path, index=False)
print(f"-> Final feature selection securely saved to svm_data/{output_path.name}")