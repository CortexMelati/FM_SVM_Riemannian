from pathlib import Path
import pandas as pd

def validate_and_analyze_data(tsv_path, base_results_dir):
    """
    Leest de TSV, valideert het bestaan van EC en EO feature files via pathlib,
    rapporteert het dataverlies, en berekent de finale demografische tabel.
    """
    # 1. Laad de dataset
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except FileNotFoundError:
        print(f"Fout: Kan TSV-bestand niet vinden op {tsv_path}")
        return

    # 2. Exclusiecriteria toepassen (Negeer sub-CBPpaXX)
    df = df[~df['participant_id'].str.contains('sub-CBPpa', na=False, case=False)].copy()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # 3. Controleer fysiek de feature bestanden
    has_ec = []
    has_eo = []
    is_valid = [] # Valid = Heeft minimaal 1 van de 2 bestanden
    
    print("Controleer lokale bestanden op EC en EO features...\n")
    
    for index, row in df.iterrows():
        sub_id = row['participant_id']
        group = row['group']
        
        folder_name = "Control" if group.lower() == 'hc' else "Patient"
        sub_dir = base_results_dir / folder_name / sub_id
        
        # Pathlib paden aanmaken
        ec_file = sub_dir / f"{sub_id}_EC_features.csv"
        eo_file = sub_dir / f"{sub_id}_EO_features.csv"
        
        # Check bestaan (geeft True of False terug)
        ec_exists = ec_file.exists()
        eo_exists = eo_file.exists()
        
        # Voeg 1 of 0 toe aan onze lijsten voor makkelijk rekenwerk
        has_ec.append(int(ec_exists))
        has_eo.append(int(eo_exists))
        is_valid.append(ec_exists or eo_exists)
        
        # Print specifieke waarschuwingen per subject (optioneel, kan je uitcommentariëren)
        if not ec_exists and not eo_exists:
            print(f" -> VOLLEDIG GEFILTERD: Geen data gevonden voor {sub_id}")
        elif not ec_exists:
            print(f" -> DEELS GEFILTERD: Wel EO, maar GEEN EC data voor {sub_id}")
        elif not eo_exists:
            print(f" -> DEELS GEFILTERD: Wel EC, maar GEEN EO data voor {sub_id}")

    # Voeg de checks als nieuwe kolommen toe aan de dataframe
    df['Found_EC'] = has_ec
    df['Found_EO'] = has_eo
    df['Valid_Subject'] = is_valid

    # 4. Bereken Dataverlies / Aantal Bestanden (Voor vs Na)
    # In eerste instantie (gebaseerd op de TSV) verwachten we per subject 1 EC en 1 EO bestand.
    file_summary = df.groupby(['study', 'group']).agg(
        Verwacht_In_1e_Instantie=('participant_id', 'count'), # Dit is de N uit de TSV
        Gevonden_EC_Na_Verwerking=('Found_EC', 'sum'),
        Gevonden_EO_Na_Verwerking=('Found_EO', 'sum'),
        Geldige_Subjecten_Over=('Valid_Subject', 'sum') # N waarbij minimaal 1 bestand is gevonden
    ).reset_index()

    print("\n" + "="*70)
    print(" OVERZICHT EC EN EO BESTANDEN PER CATEGORIE (DATA RETENTIE)")
    print("="*70)
    print(file_summary.to_string(index=False))

    # 5. Bereken de demografische gegevens voor de LaTeX tabel
    # We gebruiken alleen de subjecten die we bij de validatie als 'Valid' hebben gemarkeerd
    df_valid = df[df['Valid_Subject']].copy()
    
    summary = df_valid.groupby(['study', 'group']).agg(
        N=('participant_id', 'count'),
        Female=('sex', lambda x: (x.str.lower() == 'f').sum()),
        Male=('sex', lambda x: (x.str.lower() == 'm').sum()),
        Age_mean=('age', 'mean'),
        Age_std=('age', 'std')
    ).reset_index()

    # Formatteer LaTeX kolommen
    summary['Female / Male'] = summary['Female'].astype(str) + ' / ' + summary['Male'].astype(str)
    summary['Age (Mean ± SD)'] = summary.apply(
        lambda row: f"{row['Age_mean']:.2f} ± {row['Age_std']:.2f}" if pd.notnull(row['Age_std']) else f"{row['Age_mean']:.2f} ± 0.00",
        axis=1
    )

    final_table = summary[['study', 'group', 'N', 'Female / Male', 'Age (Mean ± SD)']]
    final_table.columns = ['Dataset', 'Group', 'N', 'Female / Male', 'Age (Mean ± SD)']

    print("\n" + "="*70)
    print(" BEREKENDE DEMOGRAFISCHE DATA VOOR LATEX TABEL (tab:demographics)")
    print("="*70)
    print(final_table.to_string(index=False))
    
    return df, final_table

# ==========================================
# Instellingen met pathlib
# ==========================================
# THESIS_ROOT is twee mappen omhoog vanaf Documents en dan naar Thesis
THESIS_ROOT = Path.home() / "Documents" / "Thesis"
DATA_ROOT = THESIS_ROOT / "Data"

# Bouw de specifieke paden op
tsv_path = DATA_ROOT / "CP_FM_dataset" / "data" / "participants.tsv"
base_results_dir = THESIS_ROOT / "FM_SVM_Riemannian" / "results" / "CP_FM_dataset" / "cp_fm_dataset"

# Voer het script uit
volledige_data, latex_df = validate_and_analyze_data(tsv_path, base_results_dir)