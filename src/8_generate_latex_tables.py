"""
=============================================================================
8. GENERATE OVERLEAF TABLES (LATEX FORMATTER)
=============================================================================
Overview:
    This script converts the CSV results from the test set evaluation into 
    publication-ready LaTeX table environments formatted for Overleaf.
=============================================================================
"""

import pandas as pd
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import PROCESSED_DATA_DIR, RIEMANN_DATA_DIR

def generate_latex():
    csv_path = RIEMANN_DATA_DIR / "riemann_testset_results.csv"
    if not csv_path.exists():
        print("🚨 Run Script 4 first to generate the testset results.")
        return

    df = pd.read_csv(csv_path)
    
    latex_content = """% =============================================================================
% LaTeX Table Templates for Overleaf - Riemannian Framework Results
% =============================================================================
\\begin{table}[htbp]
    \\centering
    \\caption{Riemannian Classification Performance on Unseen Test Set (Whole-Brain vs Central ROI)}
    \\label{tab:riemann_dual_layout_results}
    \\resizebox{\\textwidth}{!}{%
    \\begin{tabular}{llcccccc}
        \\toprule
        \\textbf{Layout} & \\textbf{Band} & \\textbf{Model} & \\textbf{Bal. Accuracy} & \\textbf{ROC-AUC} & \\textbf{Brier Score} & \\textbf{Sensitivity} & \\textbf{Specificity} \\\\
        \\midrule
"""
    
    # Iterate through the rows and format nicely
    for _, row in df.iterrows():
        layout = "Whole-Brain (19)" if row['Layout'] == 'whole' else "Central ROI (9)"
        
        latex_content += f"        {layout} & {row['Band'].capitalize()} & {row['Model']} & "
        latex_content += f"{row['Unseen_Accuracy']:.4f} & {row['ROC_AUC']:.4f} & "
        latex_content += f"{row['Brier_Score']:.4f} & {row['Sensitivity']:.4f} & {row['Specificity']:.4f} \\\\\n"
        
    latex_content += """        \\bottomrule
    \\end{tabular}%
    }
\\end{table}
"""

    output_path = current_dir / "riemann_overleaf_tables.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
        
    print(f"✅ LaTeX Overleaf tables successfully exported to: {output_path.name}")

if __name__ == "__main__":
    generate_latex()