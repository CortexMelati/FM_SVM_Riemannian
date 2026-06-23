"""
=============================================================================
4. Final SVM Training & Hyperparameter Tuning (BALANCED)
=============================================================================
Overview:
    This script trains the definitive SVM model using ONLY the optimal 
    features identified by mSFFS (Script 3). 
    
    CRITICAL FIX: It strictly uses 'balanced_accuracy' and 'class_weight' 
    to prevent the model from lazily predicting the majority class in 
    imbalanced datasets. 

Execution:
    python 4_SVM_final_model_training.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib
import mne

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, permutation_test_score

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, 
                    SVM_DATA_DIR, SVM_FIGURES_DIR, FOCUS_BAND)

# =============================================================================
# 1. LOAD TRAINING DATA & SELECTED FEATURES
# =============================================================================
print(f"Starting Final SVM Training ({FOCUS_BAND.upper()} Band)...")

train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
train_df = pd.read_csv(train_path)

y_train = train_df['Target'].values
groups_train = train_df['Subject'].values

features_path = SVM_DATA_DIR / f"final_msffs_selected_features_{FOCUS_BAND}.csv"
if not features_path.exists():
    print(f"Error: Could not find {features_path.name}. Run Script 3 first.")
    sys.exit()

selected_features = pd.read_csv(features_path)['Selected_Features'].tolist()
print(f"-> Loaded {len(selected_features)} optimal features from mSFFS.")

X_train_final = train_df[selected_features]

# =============================================================================
# 2. SCALING & STRATIFIED GROUP K-FOLD
# =============================================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=selected_features)

cv_strategy = StratifiedGroupKFold(n_splits=5)
cv_splits = list(cv_strategy.split(X_train_scaled, y_train, groups=groups_train))

# =============================================================================
# 3. GRID SEARCH CV (Balanced Optimization)
# =============================================================================
print("\n-> Commencing GridSearchCV for C and gamma optimization...")

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],  
    'gamma': np.logspace(-4, 1.5, 20),
    'class_weight': ['balanced'] # FORCED BALANCED: Prevents lazy majority voting
}

base_svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=base_svm,
    param_grid=param_grid,
    cv=cv_splits,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_

print("\nGRID SEARCH RESULTS:")
print(f"-> Best Parameters: {grid_search.best_params_}")
print(f"-> Final Internal CV Balanced Accuracy: {grid_search.best_score_:.4f}")

# =============================================================================
# 4. PERMUTATION TEST
# =============================================================================
N_PERMUTATIONS = 1000
print(f"\n-> Running {N_PERMUTATIONS}-iteration Permutation Test...")

score, permutation_scores, pvalue = permutation_test_score(
    best_svm, X_train_scaled, y_train, 
    groups=groups_train, cv=cv_splits, 
    n_permutations=N_PERMUTATIONS, n_jobs=-1, random_state=RANDOM_STATE, 
    scoring='balanced_accuracy' # Must match GridSearch
)

print("\nPERMUTATION TEST RESULTS:")
print(f"-> True Model Score (Balanced): {score:.4f}")
print(f"-> Mean Permuted Score: {permutation_scores.mean():.4f}")
print(f"-> P-value: {pvalue:.4f}")
if pvalue < 0.05:
    print("-> Conclusion: Model performs significantly better than chance (p < 0.05)!")
else:
    print("-> Conclusion: Model performance is not statistically significant.")

# =============================================================================
# 5. FREEZE AND SAVE
# =============================================================================
model_artifact = {
    'model': best_svm,
    'scaler': scaler,
    'features': selected_features,
    'band': FOCUS_BAND,
    'training_accuracy': grid_search.best_score_,
    'p_value': pvalue
}

model_path = SVM_DATA_DIR / f"saved_model_{FOCUS_BAND}.pkl"
joblib.dump(model_artifact, model_path)
print(f"\n-> Model completely frozen and saved to: svm_data/{model_path.name}")

# =============================================================================
# 6. PLOT FIGURE 4
# =============================================================================
print("\n-> Generating Final Biomarker Network Map (Figure 4)...")
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types='eeg')
info.set_montage(montage)

fig, ax = plt.subplots(figsize=(8, 8))
mne.viz.plot_sensors(info, show_names=True, axes=ax)

for collection in ax.collections:
    collection.set_sizes([150])
    collection.set_color('#cccccc')

sensor_offsets = ax.collections[0].get_offsets()
ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

for feat in selected_features:
    node1 = feat.split('-')[0]
    node2 = feat.split('-')[1].split('(')[0]
    try:
        x_coords = [ch_pos[node1][0], ch_pos[node2][0]]
        y_coords = [ch_pos[node1][1], ch_pos[node2][1]]
        ax.plot(x_coords, y_coords, color='#d62728', linewidth=3.5, alpha=0.9, zorder=1)
    except KeyError:
        pass 

ax.set_title(f"Final SVM Connectivity Features\n({FOCUS_BAND.upper()} Band - Figure 4)", fontsize=16, pad=20)
plt.tight_layout()
plot_path = SVM_FIGURES_DIR / f"Figure_4_Final_Biomarkers_{FOCUS_BAND}.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"-> Final Map saved to svm_figures/{plot_path.name}")