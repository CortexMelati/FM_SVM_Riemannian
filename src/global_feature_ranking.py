"""
=============================================================================
Initial Global Feature Ranking
=============================================================================
This script trains a model on ALL 855 features (all bands, all channels)
to generate the initial SHAP-ranking and to justify the choice for the 
Gamma ROI.

Note: Because there are 855 features, the SHAP KernelExplainer 
may take 5 to 15 minutes to run!

python global_feature_ranking.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, FIGURES_DIR

print("Starting Global Feature Ranking (Section 3.1)...")

# 1. Load the full training set
train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
train_df = pd.read_csv(train_path)

y_train = train_df['Target'].values
meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_train_raw = train_df.drop(columns=[c for c in meta_cols if c in train_df.columns])

print(f"-> Number of features loaded: {X_train_raw.shape[1]} (This should be ~855)")

# 2. Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)

# 3. Train a quick baseline SVM on all features
print("-> Training baseline SVM on the full feature space...")
global_svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
global_svm.fit(X_train_scaled, y_train)

# 4. Calculate SHAP Values
print("-> Calculating SHAP values across 855 features (This will take some time!)...")
# We use K-means (k=10) to create a background distribution to halve the computation time
background = shap.kmeans(X_train_scaled, 10)
explainer = shap.KernelExplainer(global_svm.predict_proba, background)

# We sample the dataset to calculate the impact
shap_values = explainer.shap_values(X_train_scaled)

if isinstance(shap_values, list):
    shap_values_fm = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_values_fm = np.array(shap_values)[:, :, 1] # Extract only class 1
else:
    shap_values_fm = np.array(shap_values)

# 5. Extract the Top 10 Features
mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)
feature_names = X_train_scaled.columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

print("\nTOP 10 GLOBAL FEATURES (Compare this with Li et al. Section 3.1):")
print(feature_importance.head(10).to_string(index=False))

# 6. Horizontal Bar Plot (Figure 1 from the paper)
plt.figure(figsize=(10, 8))
# We only show the top 15 to keep it readable, exactly like Figure 1
shap.summary_plot(shap_values_fm, X_train_scaled, plot_type="bar", max_display=15, show=False)
plt.title("Initial Global Feature Ranking (Mean Absolute SHAP)")
plt.tight_layout()
plot_path = FIGURES_DIR / "Figure_1_Global_SHAP_Ranking.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# Channel Importance ("Without the connections")
# ====================================================================
print("\nCalculating individual channel importance (Node Importance)...")
channel_shap = {}

for index, row in feature_importance.iterrows():
    feat = row['Feature'] # looks like 'Fz-Cz(gamma)'
    val = row['Mean_Abs_SHAP']
    
    # Extract the channels
    channels_part = feat.split('(')[0] # 'Fz-Cz'
    ch1, ch2 = channels_part.split('-')
    
    # Add the SHAP value to both channels
    channel_shap[ch1] = channel_shap.get(ch1, 0) + val
    channel_shap[ch2] = channel_shap.get(ch2, 0) + val

node_importance_df = pd.DataFrame(list(channel_shap.items()), columns=['Channel', 'Total_SHAP'])
node_importance_df = node_importance_df.sort_values(by='Total_SHAP', ascending=False)

print("\nTOP 5 MOST IMPORTANT INDIVIDUAL CHANNELS:")
print(node_importance_df.head(5).to_string(index=False))

print(f"\nAnalysis complete. Plot saved to {plot_path.name}")