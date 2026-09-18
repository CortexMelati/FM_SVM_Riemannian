"""
=============================================================================
DYNAMISCHE TABEL GENERATOR (Train/Test Split, 5-Fold & NCCP)
=============================================================================
Dit script leest de werkelijke data in via de paden uit config.py en 
genereert automatisch de LaTeX code, inclusief correcte K-Fold ranges.
"""
import pandas as pd
import sys
from config import PROCESSED_DATA_DIR, SEGMENT_LENGTH, EPOCH_LENGTH

def generate_table():
    train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
    test_path = PROCESSED_DATA_DIR / "final_dataset_test.csv"
    nccp_path = PROCESSED_DATA_DIR / "target_domain_nccp.csv"

    if not train_path.exists() or not test_path.exists() or not nccp_path.exists():
        print(f"🚨 Error: Kan een of meerdere bestanden niet vinden in {PROCESSED_DATA_DIR}.")
        sys.exit()

    print("-> Data inladen via config paden...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_nccp = pd.read_csv(nccp_path)
    
    ratio = int(SEGMENT_LENGTH / EPOCH_LENGTH)

    # --- BEREKENINGEN TRAIN SET (Outer Partition) ---
    train_hc = df_train[df_train['Target'] == 0]['Subject'].nunique()
    train_fm = df_train[df_train['Target'] == 1]['Subject'].nunique()
    train_total_subs = df_train['Subject'].nunique()
    train_macro = len(df_train) 
    train_micro = train_macro * ratio 
    
    macro_per_sub = train_macro // train_total_subs

    # --- BEREKENINGEN 5-FOLD CV RANGES (Inner Partition) ---
    cv_splits = 5
    
    # Berekening van Val Fold ranges (hoeveel mensen kunnen in 1 val fold?)
    val_hc_min = train_hc // cv_splits
    val_hc_max = val_hc_min + (1 if train_hc % cv_splits != 0 else 0)
    val_fm_min = train_fm // cv_splits
    val_fm_max = val_fm_min + (1 if train_fm % cv_splits != 0 else 0)
    
    val_subs_min = val_hc_min + val_fm_min
    val_subs_max = val_hc_max + val_fm_max
    
    val_macro_min = val_subs_min * macro_per_sub
    val_macro_max = val_subs_max * macro_per_sub
    val_micro_min = val_macro_min * ratio
    val_micro_max = val_macro_max * ratio

    # Berekening van Train Folds ranges
    cv_train_hc_min = train_hc - val_hc_max
    cv_train_hc_max = train_hc - val_hc_min
    cv_train_fm_min = train_fm - val_fm_max
    cv_train_fm_max = train_fm - val_fm_min
    
    cv_train_subs_min = cv_train_hc_min + cv_train_fm_min
    cv_train_subs_max = cv_train_hc_max + cv_train_fm_max
    
    cv_train_macro_min = cv_train_subs_min * macro_per_sub
    cv_train_macro_max = cv_train_subs_max * macro_per_sub
    cv_train_micro_min = cv_train_macro_min * ratio
    cv_train_micro_max = cv_train_macro_max * ratio

    # Formatteer naar mooie range-strings
    v_hc = f"{val_hc_min}--{val_hc_max}"
    v_fm = f"{val_fm_min}--{val_fm_max}"
    v_sub = f"{val_subs_min}--{val_subs_max}"
    v_mac = f"{val_macro_min}--{val_macro_max}"
    v_mic = f"{val_micro_min}--{val_micro_max}"
    
    tr_hc = f"{cv_train_hc_min}--{cv_train_hc_max}"
    tr_fm = f"{cv_train_fm_min}--{cv_train_fm_max}"
    tr_sub = f"{cv_train_subs_min}--{cv_train_subs_max}"
    tr_mac = f"{cv_train_macro_min}--{cv_train_macro_max}"
    tr_mic = f"{cv_train_micro_min}--{cv_train_micro_max}"

    # --- BEREKENINGEN TEST SET (Outer Partition) ---
    test_hc = df_test[df_test['Target'] == 0]['Subject'].nunique()
    test_fm = df_test[df_test['Target'] == 1]['Subject'].nunique()
    test_total_subs = df_test['Subject'].nunique()
    test_macro = len(df_test)
    test_micro = test_macro * ratio

    # --- TOTALEN FM DATASET ---
    total_hc = train_hc + test_hc
    total_fm = train_fm + test_fm
    total_subs = train_total_subs + test_total_subs
    total_macro = train_macro + test_macro
    total_micro = train_micro + test_micro
    
    # --- BEREKENINGEN NCCP DATASET ---
    nccp_hc = df_nccp[df_nccp['Target'] == 0]['Subject'].nunique()
    nccp_pa = df_nccp[df_nccp['Target'] == 1]['Subject'].nunique()
    nccp_total_subs = df_nccp['Subject'].nunique()
    nccp_macro = len(df_nccp)
    nccp_micro = nccp_macro * ratio

    # --- LATEX TABEL OPBOUWEN ---
    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Participant and segment counts for the internal partitions (FM dataset) and the external validation cohort (NCCP dataset).}}
\\label{{tab:final_counts}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{l|ccc|cc}}
\\toprule
\\textbf{{Dataset Partition}} & \\textbf{{HC ($N$)}} & \\textbf{{Pa ($N$)}} & \\textbf{{Total ($N$)}} & \\textbf{{Total 30-sec}} & \\textbf{{Total 1-sec}} \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\textbf{{Internal Dataset (FM)}}}} \\\\
Total FM Master Dataset & {total_hc} & {total_fm} & {total_subs} & {total_macro} & {total_micro} \\\\
\\quad $\\llcorner$ Outer Training Partition (80\\%) & {train_hc} & {train_fm} & {train_total_subs} & {train_macro} & {train_micro} \\\\
\\quad\\quad \\textit{{-- Inner 5-Fold CV: Train Folds (Range)}} & \\textit{{{tr_hc}}} & \\textit{{{tr_fm}}} & \\textit{{{tr_sub}}} & \\textit{{{tr_mac}}} & \\textit{{{tr_mic}}} \\\\
\\quad\\quad \\textit{{-- Inner 5-Fold CV: Val Fold (Range)}} & \\textit{{{v_hc}}} & \\textit{{{v_fm}}} & \\textit{{{v_sub}}} & \\textit{{{v_mac}}} & \\textit{{{v_mic}}} \\\\
\\quad $\\llcorner$ Isolated Test Partition (20\\%) & {test_hc} & {test_fm} & {test_total_subs} & {test_macro} & {test_micro} \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\textbf{{External Target Dataset (NCCP)}}}} \\\\
Total Target Cohort & {nccp_hc} & {nccp_pa} & {nccp_total_subs} & {nccp_macro} & {nccp_micro} \\\\
\\quad \\textit{{$\\llcorner$ Iterative N-Fold Validation (2--47 Folds)*}} & \\textit{{varies}} & \\textit{{varies}} & \\textit{{varies}} & \\textit{{varies}} & \\textit{{varies}} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\vspace{{1ex}}
\\parbox{{\\textwidth}}{{\\footnotesize \\textit{{Note:}} Through dynamic downsampling in the preprocessing pipeline (\\texttt{{build\\_dataset.py}}), the data density is strictly calibrated to mitigate algorithmic bias. Consequently, every individual subject across all datasets provides exactly {macro_per_sub} macro-segments (30-sec). Each 30-sec segment is subsequently windowed into {ratio} non-overlapping micro-epochs (1-sec), yielding exactly {macro_per_sub * ratio} micro-epochs per participant. *The external validation cohort undergoes an iterative validation strategy, scaling from 2 to 47 stratified folds to systematically evaluate transfer learning efficacy across varying target training sizes.}}
\\end{{table}}"""

    print("\n" + "="*70)
    print("🏆 BEREKENING SUCCESVOL: Tabel én Note uitlijning geperfectioneerd!")
    print("="*70)
    print("\n👇 KOPIEER DEZE LATEX CODE VOOR JE THESIS 👇\n")
    print(latex_code)

if __name__ == "__main__":
    generate_table()