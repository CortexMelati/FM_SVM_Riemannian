import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
import ast

from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paden opzetten
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, SVM_DATA_DIR, RANDOM_STATE

def freeze_cov_model():
    print("❄️ Freezing the missing TSSVM_Cov model...")

    # 1. Haal de perfecte parameters uit je scoreboard
    scoreboard_path = RIEMANN_DATA_DIR / "scoreboard_roi_ablation.csv"
    df_roi = pd.read_csv(scoreboard_path)
    
    # Zoek de rij voor Theta & TSSVM_Cov
    row = df_roi[(df_roi['Band'] == 'THETA') & (df_roi['Architecture'] == 'TSSVM_Cov')].iloc[0]
    p_dict = ast.literal_eval(row['Optimal_Params'])

    # 2. Laad de Theta ROI trainingsdata in
    X_covs = np.load(RIEMANN_DATA_DIR / "covs_train_theta_roi.npy")
    y = np.load(RIEMANN_DATA_DIR / "y_train_riemann.npy")

    # 3. Bouw de Pipeline met de winnende parameters
    pipe = Pipeline([
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE, **p_dict))
    ])

    # 4. Train hem (dit duurt ~1 seconde op je i7)
    pipe.fit(X_covs, y)

    # 5. Sla hem op als .pkl voor Script 5
    model_name = "model_riemann_Theta_roi_TSSVM_Cov.pkl"
    joblib.dump({
        'model': pipe,
        'band': 'theta',
        'layout': 'roi',
        'training_balanced_accuracy': row['CV_Balanced_Accuracy']
    }, SVM_DATA_DIR / model_name)

    print(f"✅ Succes! {model_name} is succesvol weggeschreven.")
    print("Je kunt nu direct Script 5 (5_Riemann_Final_Evaluation.py) starten!")

if __name__ == "__main__":
    freeze_cov_model()