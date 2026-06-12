"""
=============================================================================
Initial Global Feature Ranking
=============================================================================
Dit script traint een model op ALLE 855 features (alle banden, alle kanalen)
om de initiële SHAP-ranking te genereren en de keuze voor de Gamma ROI te 
verantwoorden.

Let op: Omdat het 855 features zijn, kan de SHAP KernelExplainer 
hier 5 tot 15 minuten over doen!

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

print("🚀 Starten van Global Feature Ranking (Sectie 3.1)...")

# 1. Laad de volledige trainingsset
train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
train_df = pd.read_csv(train_path)

y_train = train_df['Target'].values
meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_train_raw = train_df.drop(columns=[c for c in meta_cols if c in train_df.columns])

print(f"-> Aantal features ingeladen: {X_train_raw.shape[1]} (Dit zouden er ~855 moeten zijn)")

# 2. Schalen
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)

# 3. Train een snelle baseline SVM op alle features
print("-> Trainen van baseline SVM op de volledige feature space...")
global_svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
global_svm.fit(X_train_scaled, y_train)

# 4. Bereken SHAP Values
print("-> Berekenen van SHAP waarden over 855 features (Dit duurt even!)...")
# We gebruiken K-means (k=10) om een achtergrond-distributie te maken om de rekentijd te halveren
background = shap.kmeans(X_train_scaled, 10)
explainer = shap.KernelExplainer(global_svm.predict_proba, background)

# We samplen de testset (of een deel van train) om de impact te berekenen
shap_values = explainer.shap_values(X_train_scaled)

if isinstance(shap_values, list):
    shap_values_fm = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_values_fm = np.array(shap_values)[:, :, 1] # Pak uitsluitend klasse 1
else:
    shap_values_fm = np.array(shap_values)

# 5. Extract de Top 10 Features
mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)
feature_names = X_train_scaled.columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

print("\n🏆 TOP 10 GLOBAL FEATURES (Vergelijk dit met Li et al. Sectie 3.1):")
print(feature_importance.head(10).to_string(index=False))

# 6. Horizontale Bar Plot (Figure 1 uit de paper)
plt.figure(figsize=(10, 8))
# We tonen alleen de top 15 om het leesbaar te houden, exact zoals Figure 1
shap.summary_plot(shap_values_fm, X_train_scaled, plot_type="bar", max_display=15, show=False)
plt.title("Initial Global Feature Ranking (Mean Absolute SHAP)")
plt.tight_layout()
plot_path = FIGURES_DIR / "Figure_1_Global_SHAP_Ranking.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# Kanaal Belangrijkheid ("Zonder de connecties")
# ====================================================================
print("\n🧠 Berekenen van individuele kanaal-belangrijkheid (Node Importance)...")
channel_shap = {}

for index, row in feature_importance.iterrows():
    feat = row['Feature'] # ziet eruit als 'Fz-Cz(gamma)'
    val = row['Mean_Abs_SHAP']
    
    # Haal de kanalen eruit
    channels_part = feat.split('(')[0] # 'Fz-Cz'
    ch1, ch2 = channels_part.split('-')
    
    # Tel de SHAP waarde op bij beide kanalen
    channel_shap[ch1] = channel_shap.get(ch1, 0) + val
    channel_shap[ch2] = channel_shap.get(ch2, 0) + val

node_importance_df = pd.DataFrame(list(channel_shap.items()), columns=['Channel', 'Total_SHAP'])
node_importance_df = node_importance_df.sort_values(by='Total_SHAP', ascending=False)

print("\n🥇 TOP 5 MEEST BELANGRIJKE INDIVIDUELE KANALEN:")
print(node_importance_df.head(5).to_string(index=False))

print(f"\n✅ Analyse compleet. Plot opgeslagen in {plot_path.name}")