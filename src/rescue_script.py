import pandas as pd
from pathlib import Path
import sys

# Paden opzetten
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, SVM_DATA_DIR

def rescue_files():
    print("🚑 DRAAIEN VAN RECOVERY SCRIPT...")

    # Lees de bestanden die gelukkig wél al waren opgeslagen
    df_whole = pd.read_csv(RIEMANN_DATA_DIR / "scoreboard_whole_brain.csv")
    df_roi = pd.read_csv(RIEMANN_DATA_DIR / "scoreboard_roi_ablation.csv")

    best_bands = ['theta', 'gamma']

    report_text = "====================================================\n"
    report_text += " FINAL ROI ABLATION RESULTS (9 CHANNELS)\n"
    report_text += "====================================================\n\n"

    for band_name in best_bands:
        band_rows = df_roi[df_roi['Band'] == band_name.upper()].sort_values(by='CV_Balanced_Accuracy', ascending=False)
        best_row = band_rows.iloc[0]
        
        report_text += f"WINNER: {band_name.upper()} BAND\n"
        report_text += f"Architecture:      {best_row['Architecture']}\n"
        report_text += f"Balanced Accuracy: {best_row['CV_Balanced_Accuracy']:.4f}\n"
        report_text += f"Optimal Params:    {best_row['Optimal_Params']}\n"
        report_text += "-"*52 + "\n"

    # DE FIX: We dwingen Windows om het als UTF-8 (met emoji-ondersteuning) op te slaan
    report_path = SVM_DATA_DIR / "riemann_roi_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # De overkoepelende CSV alsnog aanmaken
    df_final = pd.concat([df_whole, df_roi]).sort_values(by=['Band', 'CV_Balanced_Accuracy'], ascending=[True, False])
    df_final.to_csv(RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv", index=False)

    print("✅ Bestanden succesvol hersteld en opgeslagen in utf-8!")

if __name__ == "__main__":
    rescue_files()