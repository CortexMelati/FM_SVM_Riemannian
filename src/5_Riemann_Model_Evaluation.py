"""
=============================================================================
5. RIEMANNIAN MODEL EVALUATION (Test Set - Subject Level)
=============================================================================
Overview:
    This script evaluates the frozen Riemannian winning models 
    on the strictly isolated 20% hold-out test set across all frequency bands
    AND spatial layouts (ROI vs WHOLE). 
    Crucially, it applies Majority Voting to group the 1-second epochs back 
    into clinical predictions per subject.

Execution:
    python 5_Riemann_Model_Evaluation.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import ast
import mne
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances, XdawnCovariances, Coherences
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score, balanced_accuracy_score)

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, BANDS, BEST_CHANNELS_EVALUATE, 
                    CHANNELS_1020, RANDOM_STATE, CP_FM_DIR, SAVED_MODELS_DIR)


class AverageFrequencies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.mean(X, axis=-1) if X.ndim == 4 else X

try:
    from pyriemann.preprocessing import NearestSPD
except ImportError:
    try:
        from pyriemann.estimation import NearestSPD
    except ImportError:
        from pyriemann.utils.base import nearest_sym_pos_def
        class NearestSPD(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return np.array([nearest_sym_pos_def(x) for x in X])


# --- Custom Transformers ---
class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X):
        iir_params = dict(order=4, ftype='butter', output='sos')
        return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=iir_params, verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_weight = np.sum(in_bin) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece


def plot_permutation_distribution(permuted_scores, actual_acc, pvalue, target_band):
    print(f"  -> Generating Permutation Test Distribution Plot (KDE) for {target_band.upper()} band...")
    
    plt.figure(figsize=(8, 6))
    
    
    sns.kdeplot(
        permuted_scores, 
        fill=True, 
        color='#93c59e', 
        alpha=0.6, 
        linewidth=2.5,
        bw_adjust=1.5, 
        label='Permuted Scores (Null Distribution)'
    )
    
    
    plt.axvline(actual_acc, color='#d62728', linestyle='dashed', linewidth=2.5, 
                label=f'Actual Model Score ({actual_acc:.4f})')
    
    
    plt.axvline(np.mean(permuted_scores), color='black', linestyle='dotted', linewidth=2, 
                label=f'Chance Level (Mean: {np.mean(permuted_scores):.4f})')

    
    plt.title(f"Permutation Test Distribution (1000 Iterations)\n({target_band.upper()} Band - p = {pvalue:.4f})", fontsize=16, pad=15)
    plt.xlabel('Balanced Accuracy', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    
    plt.legend(frameon=False, loc='upper left', fontsize=12)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True) 
    plot_path = RIEMANN_FIGURES_DIR / f"Figure_Permutation_Distribution_{target_band}.png"
    plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  -> Permutation plot saved to riemann_figures/{plot_path.name}")


def evaluate_riemann_testset():
    print("STARTING STEP 5: RIEMANNIAN EVALUATION ON UNSEEN TEST SET (SUBJECT-LEVEL)")

    X_train_raw = np.load(SAVED_MODELS_DIR / "X_train_raw.npy")
    y_train = np.load(SAVED_MODELS_DIR / "y_train_riemann.npy")
    X_test_raw = np.load(SAVED_MODELS_DIR / "X_test_raw.npy")
    y_test = np.load(SAVED_MODELS_DIR / "y_test_riemann.npy")
    groups_test = np.load(SAVED_MODELS_DIR / "groups_test_riemann.npy") 
    
    scoreboard_path = RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard_10x_cv5.csv"
    if not scoreboard_path.exists():
        sys.exit(" Scoreboard not found! Please run Script 2/3 first.")
    scoreboard = pd.read_csv(scoreboard_path)
    
    
    if 'Layout' not in scoreboard.columns:
        sys.exit(" Kolom 'Layout' ontbreekt in scoreboard! Voeg deze toe in Script 3 (met waarden 'ROI' of 'WHOLE').")

    ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]
    valid_architectures = ['TSSVM_Cov', 'TSSVM_Xdawn', 'TSSVM_Coh'] # if different amend

    final_results = []
    
    
    layouts_to_test = ['ROI', 'WHOLE']

    for layout in layouts_to_test:
        for band_name, (l_freq, h_freq) in BANDS.items():
            print(f"\n{'='*60}\n📡 ANALYZING: {band_name.upper()} BAND | LAYOUT: {layout}\n{'='*60}")
            
            # Filter scoreboard op BAND én LAYOUT
            band_scores = scoreboard[(scoreboard['Band'] == band_name.upper()) & 
                                     (scoreboard['Layout'] == layout) & 
                                     (scoreboard['Architecture'].isin(valid_architectures))]
                                     
            if band_scores.empty:
                print(f"Geen getrainde modellen gevonden voor {band_name.upper()} - {layout}. Skipping...")
                continue
                
            best_row = band_scores.loc[band_scores['CV_Balanced_Accuracy_Mean'].idxmax()]
            arch = best_row['Architecture']
            params = ast.literal_eval(best_row['Optimal_Params'])
            
            print(f"-> Optimal Architecture: {arch}")
            print(f"-> Optimal Params: C={params['C']}, Kernel={params['kernel']}")
            
            # # Bouw de stappen old ======================================================================
            # steps = [('filter', MNEBandPass(l_freq, h_freq, 500))]
            
            # # ALLEEN toevoegen als de layout ROI is
            # if layout == 'ROI':
            #     steps.append(('roi', ROIExtractor(ROI_INDICES)))
            
            # if arch == 'TSSVM_Cov':
            #     steps.extend([('cov', Covariances(estimator='oas')), ('ts', TangentSpace(metric='riemann'))])
            # elif arch == 'TSSVM_Xdawn':
            #     steps.extend([('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), ('ts', TangentSpace(metric='riemann'))])
            # elif arch == 'TSSVM_Coh':
            #     steps.extend([
            #         ('coh', Coherences(coh='lagged')), 
            #         ('avg_freq', AverageFrequencies()), 
            #         ('spd', NearestSPD()), 
            #         ('ts', TangentSpace(metric='riemann'))
            #     ])
                
            # steps.extend([
            #     ('scaler', StandardScaler()),
            #     ('svm', SVC(C=params['C'], kernel=params['kernel'], class_weight='balanced', probability=True, random_state=RANDOM_STATE))
            # ])
            
            # pipeline = Pipeline(steps)
            # print(f"-> Training optimal pipeline on full training data ({layout} channels)...")
            # pipeline.fit(X_train_raw, y_train)
            #===================================================================================================

            
            layout_str = 'whole' if layout == 'WHOLE' else 'roi'
            model_filename = f"model_riemann_{band_name}_{layout_str}_{arch}_10x_cv5.pkl"
            model_path = SAVED_MODELS_DIR / model_filename
            
            try:
                artifact = joblib.load(model_path)
            except FileNotFoundError:
                print(f"⚠️ Waarschuwing: {model_filename} niet gevonden. Skipping...")
                continue
            
            
            pipeline = artifact['model']
            
            train_mean = artifact.get('training_balanced_accuracy', 0.0)
            train_std = artifact.get('training_std', 0.0)
            
            print(f"-> Ingeladen model: {arch} (Train Bal. Acc: {train_mean:.3f})")

            # ========================================================================================


            print("-> Predicting on Unseen Test Data (1-second epochs)...")
            y_pred_epochs = pipeline.predict(X_test_raw)
            y_prob_epochs = pipeline.predict_proba(X_test_raw)[:, 1]

            
            print("-> Aggregating predictions to Subject-Level...")
            df_preds = pd.DataFrame({
                'Subject': groups_test,
                'True_Label': y_test,
                'Pred_Class': y_pred_epochs,
                'Pred_Prob': y_prob_epochs
            })

            df_subject = df_preds.groupby('Subject').agg(
                True_Label=('True_Label', 'first'), 
                Pred_Class=('Pred_Class', lambda x: x.mode()[0]),
                Consistency=('Pred_Class', lambda x: (x == x.mode()[0]).mean()),
                Pred_Prob=('Pred_Prob', 'mean') 
            ).reset_index()

            y_test_sub = df_subject['True_Label'].values
            y_pred_sub = df_subject['Pred_Class'].values
            y_prob_sub = df_subject['Pred_Prob'].values

            
            acc = accuracy_score(y_test_sub, y_pred_sub)
            prec = precision_score(y_test_sub, y_pred_sub, zero_division=0)
            rec = recall_score(y_test_sub, y_pred_sub, zero_division=0)
            auc = roc_auc_score(y_test_sub, y_prob_sub)
            auprc = average_precision_score(y_test_sub, y_prob_sub) 
            brier = brier_score_loss(y_test_sub, y_prob_sub)
            ece = expected_calibration_error(y_test_sub, y_prob_sub)
            
            cm = confusion_matrix(y_test_sub, y_pred_sub)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                # Bescherm tegen delen door nul
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            else:
                print(f"Waarschuwing: Confusion matrix heeft onverwachte vorm voor {band_name}-{layout}: {cm.shape}")
                fpr, fnr = 0.0, 0.0

            # --- PERMUTATION TEST ---
            from sklearn.utils import shuffle
            n_permutations = 1000
            permuted_scores = []
            for i in range(n_permutations):
                y_test_shuffled = shuffle(y_test_sub, random_state=RANDOM_STATE + i)
                score = balanced_accuracy_score(y_test_shuffled, y_pred_sub)
                permuted_scores.append(score)

            pvalue = (np.sum(np.array(permuted_scores) >= acc) + 1) / (n_permutations + 1)
            print(f"-> Permutation P-value: {pvalue:.4f}")
            

            mean_consistency = df_subject['Consistency'].mean()
            
            report_text = (
                f"====================================================\n"
                f" FINAL TEST METRICS SUBJECT - {arch} ({layout}) {band_name.upper()} \n"
                f"====================================================\n"
                f"Intra-Subject Consistency: {mean_consistency:.2%}\n"
                f"Accuracy:        {acc:.4f}\n"
                f"Precision:       {prec:.4f}\n"
                f"Recall:          {rec:.4f}\n"
                f"FPR:             {fpr:.4f}\n"
                f"FNR:             {fnr:.4f}\n"
                f"ROC-AUC:         {auc:.4f}\n"
                f"AUPRC:           {auprc:.4f}\n"
                f"Brier Score:     {brier:.4f}\n"
                f"ECE:             {ece:.4f}\n"
                f"Permutation P:   {pvalue:.4f}\n"
                f"====================================================\n"
            )
            report_path = RIEMANN_DATA_DIR / f"final_test_metrics_report_{band_name.lower()}_{layout.lower()}_{arch}.txt"
            with open(report_path, "w") as f:
                f.write(report_text)

            plot_permutation_distribution(permuted_scores, acc, pvalue, band_name)
            
            # Reconstruct the filename exactly as saved in Script 3
            layout_str = 'whole' if layout == 'WHOLE' else 'roi'
            model_filename = f"model_riemann_{band_name.lower()}_{layout_str}_{arch}_10x_cv5.pkl"
            model_path = SAVED_MODELS_DIR / model_filename
            
            try:
                artifact = joblib.load(model_path)
            except FileNotFoundError:
                print(f"Waarschuwing: {model_filename} niet gevonden. CV scores worden op 0.0 gezet.")
                artifact = {}
            
        
            train_mean = artifact.get('training_balanced_accuracy', 0.0)
            train_std = artifact.get('training_std', 0.0)

            final_results.append({
                'Band': band_name.upper(),
                'Layout': layout, 
                'Optimal_Architecture': arch,
                'Optimal_Params': f"C={params['C']}, {params['kernel']}",
                'CV_Training_Score': f"{train_mean:.3f} ± {train_std:.3f}",
                'Bal_Accuracy': f"{acc:.2%}",
                'Sensitivity': f"{rec:.2%}",
                'Precision': f"{prec:.2%}",
                'FPR': f"{fpr:.2%}", 
                'FNR': f"{fnr:.2%}", 
                'AUPRC': f"{auprc:.4f}",
                'AUROC': f"{auc:.4f}",
                'Brier': f"{brier:.4f}",
                'ECE': f"{ece:.4f}",
                'Permutation_P': f"{pvalue:.4f}"
            })
            
            
            # --- GENERATE PLOTS ---
            annot_labels = np.empty_like(cm, dtype=object)
            
            if cm.shape == (2, 2):
                TN_subjs = df_subject[(df_subject['True_Label'] == 0) & (df_subject['Pred_Class'] == 0)]['Subject']
                FP_subjs = df_subject[(df_subject['True_Label'] == 0) & (df_subject['Pred_Class'] == 1)]['Subject']
                FN_subjs = df_subject[(df_subject['True_Label'] == 1) & (df_subject['Pred_Class'] == 0)]['Subject']
                TP_subjs = df_subject[(df_subject['True_Label'] == 1) & (df_subject['Pred_Class'] == 1)]['Subject']
                
                
                len_tn = len(df_preds[df_preds['Subject'].isin(TN_subjs)])
                len_fp = len(df_preds[df_preds['Subject'].isin(FP_subjs)])
                len_fn = len(df_preds[df_preds['Subject'].isin(FN_subjs)])
                len_tp = len(df_preds[df_preds['Subject'].isin(TP_subjs)])
                
                
                annot_labels[0, 0] = f"{cm[0, 0]}\n{len_tn}\n{len_tn // 30}"
                annot_labels[0, 1] = f"{cm[0, 1]}\n{len_fp}\n{len_fp // 30}"
                annot_labels[1, 0] = f"{cm[1, 0]}\n{len_fn}\n{len_fn // 30}"
                annot_labels[1, 1] = f"{cm[1, 1]}\n{len_tp}\n{len_tp // 30}"
            else:
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        annot_labels[i, j] = str(cm[i, j])

            
            display_arch = arch.replace('Xdawn', 'xDAWN')

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Oranges',
                        xticklabels=['Healthy (0)', 'Fibro (1)'], 
                        yticklabels=['Healthy (0)', 'Fibro (1)'],
                        annot_kws={"size": 14}) 
            
            
            plt.title(f'Riemannian FINAL Validation ({display_arch} - {band_name.upper()} - {layout})\nSubject-Level (Accuracy: {acc:.2%})', fontsize=14)
            plt.ylabel('True Clinical Diagnosis', fontsize=12)
            plt.xlabel('Predicted Diagnosis (Majority Vote)', fontsize=12)
            plt.tight_layout()
            
            RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plot_path = RIEMANN_FIGURES_DIR / f"final_confusion_matrix_riemann_{band_name}_{layout}_{arch}.png"
            plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
            plt.close()

            print(f"  -> Generating t-SNE data distribution for Riemannian {arch} ({layout})...")
            
            ts_pipeline = Pipeline(pipeline.steps[:-2]) 
            X_test_tangent = ts_pipeline.transform(X_test_raw)
            
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=min(30, len(X_test_tangent)-1), random_state=42)
            X_tsne_riemann = tsne.fit_transform(X_test_tangent)

            plt.figure(figsize=(8, 6))
            scatter = sns.scatterplot(
                x=X_tsne_riemann[:, 0], y=X_tsne_riemann[:, 1], 
                hue=y_test,
                palette={0: '#5c8cbc', 1: '#d62728'},
                s=80, alpha=0.8, edgecolor='white'
            )
            plt.title(f"Riemannian Tangent Space Distribution\n({band_name.upper()} Band - {layout} Layout)", fontsize=14, pad=15)
            plt.xlabel("t-SNE Dimension 1", fontsize=11)
            plt.ylabel("t-SNE Dimension 2", fontsize=11)

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            handles, labels = scatter.get_legend_handles_labels()
            plt.legend(handles=handles, labels=['Healthy Control (HC)', 'Fibromyalgia (FM)'], title='Diagnosis', frameon=False)
            plt.tight_layout()
            
            tsne_path = RIEMANN_FIGURES_DIR / f"Figure_5_tsne_riemann_{band_name}_{layout}_{arch}.png"
            plt.savefig(tsne_path, dpi=300, facecolor='white', bbox_inches='tight')
            plt.close()

    # 6. EXPORT MASTER TABLE FOR LATEX
    results_df = pd.DataFrame(final_results)
    csv_path = RIEMANN_DATA_DIR / "final_riemannian_test_table.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}\nALL BANDS & LAYOUTS EVALUATED SUCCESSFULLY!\n{'='*70}")
    print("Here is your final data for the LaTeX Table 2:\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_riemann_testset()