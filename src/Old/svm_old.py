"""
Machine Learning Pipeline - svm.py
Evaluates Chronic Pain classification using GridSearchCV on extracted features.
Outputs an automated hyperparameter report.

- work in progress
"""

import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Scikit-Learn modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    balanced_accuracy_score,
    roc_auc_score
)

# Zorg dat Python config.py kan vinden
sys.path.append(os.path.abspath(".."))
import config

# =============================================================================
# 1. MODEL CONFIGURATIE 
# =============================================================================
SVM_CONFIG = {
    'model': SVC(kernel='rbf', 
                 probability=True, 
                 class_weight='balanced', 
                 cache_size=1000, 
                 random_state=config.RANDOM_STATE),
    'params': {
        'clf__C': [0.1, 1, 5, 10, 50, 75, 100], 
        'clf__gamma': ['scale', 0.01, 1, 10]
    }
}

# =============================================================================
# 2. DATA LOADING & PLOTTING FUNCTIES
# =============================================================================
def load_split_data(dataset_dir, feature_type="riemannian"):
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    
    suffix = "_spectral" if feature_type == "spectral" else ""
    print(f"► Loading {feature_type.upper()} data from: {dataset_dir.name}")
    
    try:
        X_train = np.load(train_dir / f"X_train{suffix}.npy")
        y_train = np.load(train_dir / f"y_train{suffix}.npy")
        X_test = np.load(test_dir / f"X_test{suffix}.npy")
        y_test = np.load(test_dir / f"y_test{suffix}.npy")
        
        print(f"  ✓ Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  ✓ Test set:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"  ✗ Error: Could not find data files. ({e})")
        sys.exit(1)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', cbar=False,
                xticklabels=['Pred: HC', 'Pred: FM'],
                yticklabels=['True: HC', 'True: FM'])
    plt.title(title)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. HOOFD UITVOERING
# =============================================================================
def run_gridsearch_svm(feature_type="riemannian"):
    base_dir = config.PROCESSED_DATA_DIR / "CP_FM_dataset"
    X_train, X_test, y_train, y_test = load_split_data(base_dir, feature_type=feature_type)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', SVM_CONFIG['model'])
    ])
    
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.RANDOM_STATE)
    
    grid = GridSearchCV(
        pipeline, 
        param_grid=SVM_CONFIG['params'], 
        cv=inner_cv, 
        scoring='balanced_accuracy', 
        n_jobs=-1, 
        verbose=1  
    )
    
    print("\n► Starting Hyperparameter Tuning (GridSearchCV)...")
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # ==========================================
    # ► RAPPORTAGE GENEREREN
    # ==========================================
    # We slaan dit op in config.RESULTS_DIR (maak map aan als die niet bestaat)
    report_dir = config.RESULTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "hyperparameter_report.txt"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv_score = grid.best_score_ * 100
    
    # We gebruiken "a" (append) zodat oude rapporten niet worden overschreven
    with open(report_file, "a", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write(f"🏆 SVM HYPERPARAMETER REPORT - {feature_type.upper()}\n")
        f.write("====================================================\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Cross-Validation Balanced Accuracy: {cv_score:.2f}%\n\n")
        f.write("Best Parameters Found:\n")
        for param, value in grid.best_params_.items():
            clean_param = param.replace('clf__', '')
            f.write(f"  - {clean_param}: {value}\n")
            print(f"  - {clean_param}: {value}") # Print ook naar terminal
        f.write("\n\n")
        
    print(f"\n  📝 Report saved successfully to: {report_file}")
    # ==========================================

    print("\n► Evaluating winning model on unseen Test Set...")
    y_pred = best_model.predict(X_test)
    fm_class_index = np.where(best_model.classes_ == 1)[0][0]
    y_proba = best_model.predict_proba(X_test)[:, fm_class_index]
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n======================================")
    print(f"📊 TEST SET PERFORMANCE")
    print(f"======================================")
    print(f"Balanced Accuracy : {bal_acc * 100:.2f}%")
    print(f"ROC AUC Score     : {roc_auc:.3f}")
    print(f"======================================\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["HC (0)", "FM (1)"]))
    
    plot_confusion_matrix(y_test, y_pred, title=f"SVM Normalized CM - {feature_type.capitalize()}")

if __name__ == "__main__":
    # Schakelaar voor welk experiment je wilt draaien: 'riemannian' of 'spectral'
    experiment_type = "spectral" 
    
    print("=============================================================================")
    print(f"ML PIPELINE: SVM GRIDSEARCH - Feature type: {experiment_type.upper()}")
    print("=============================================================================\n")
    run_gridsearch_svm(feature_type=experiment_type)