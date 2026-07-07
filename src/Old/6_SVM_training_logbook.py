"""
=============================================================================
6. HYPERPARAMETER & TRAINING LOGBOOK
=============================================================================
Overview:
    This utility script opens all frozen model artifacts (.pkl) in the 
    svm_data directory. It extracts the exact hyperparameters (C, gamma, 
    class_weight) selected by GridSearchCV, alongside the performance metrics, 
    and exports a clean logbook for the thesis appendix.

Execution:
    python 6_SVM_training_logbook.py
=============================================================================
"""

import pandas as pd
from pathlib import Path
import sys
import joblib
import glob
import os
import numpy as np

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import SVM_DATA_DIR

print("Starting Model Inspection & Logbook Generation...\n")

# Zoek alle opgeslagen modellen in de map
model_files = glob.glob(os.path.join(SVM_DATA_DIR, "saved_model_*.pkl"))

if not model_files:
    print(f"No .pkl models found in {SVM_DATA_DIR.name}.")
    sys.exit()

logbook_entries = []

for file_path in model_files:
    # 1. Open de tijdcapsule (.pkl)
    artifact = joblib.load(file_path)
    
    svm_model = artifact['model']
    band = artifact.get('band', Path(file_path).stem.split('_')[-1]) # Haal bandnaam uit bestandsnaam of dict
    
    # 2. Lees de hyperparameters direct uit het getrainde SVM object
    c_val = svm_model.C
    gamma_val = svm_model.gamma
    weight_val = svm_model.class_weight
    
    # 3. Lees de opgeslagen metrics en features
    n_features = len(artifact['features'])
    train_acc = artifact['training_accuracy']
    
    # VEILIGE FIX: Haalt p_value veilig op. Omdat we deze naar Script 5 hebben verplaatst 
    # (om datalekkage te voorkomen), vullen we hier een Not-a-Number in als hij ontbreekt.
    p_val = artifact.get('p_value', np.nan) 
    
    # 4. Voeg toe aan het logboek
    logbook_entries.append({
        'Frequency_Band': band.upper(),
        'Selected_Features (N)': n_features,
        'C_Parameter': round(c_val, 4) if isinstance(c_val, float) else c_val,
        'Gamma_Parameter': round(gamma_val, 6) if isinstance(gamma_val, float) else gamma_val,
        'Class_Weight': str(weight_val),
        'CV_Balanced_Accuracy': round(train_acc, 4),
        'Permutation_P_Value': round(p_val, 4) if not np.isnan(p_val) else "Zie Script 5"
    })

# =============================================================================
# EXPORT EN PRINT LOGBOEK
# =============================================================================
logbook_df = pd.DataFrame(logbook_entries).sort_values(by='Frequency_Band')

print("="*95)
print(" 📖 HYPERPARAMETER & TRAINING LOGBOOK ")
print("="*95)
print(logbook_df.to_string(index=False))
print("="*95)

output_path = SVM_DATA_DIR / "hyperparameter_training_logbook.csv"
logbook_df.to_csv(output_path, index=False)

print(f"\n-> Logbook successfully exported to: svm_data/{output_path.name}")