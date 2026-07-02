
# when automatically going over all bands amend to below


def run_msffs_all_bands():
    print("🚀 STARTING mSFFS FEATURE SELECTION FOR ALL BANDS")

    # Laad de trainingsdata 1 keer in voor efficiëntie
    train_path = PROCESSED_DATA_DIR / "final_dataset_train.csv"
    train_df = pd.read_csv(train_path)
    y_train = train_df['Target'].values
    groups_train = train_df['Subject'].values

    for band_name in BANDS.keys():
        band_name_lower = band_name.lower()
        print(f"\n{'='*60}\n📡 RUNNING mSFFS FOR BAND: {band_name.upper()}\n{'='*60}")

        # A. Controleer of Script 2 de Top 10 heeft gegenereerd voor deze specifieke band
        top_10_path = PROCESSED_DATA_DIR / f"top_10_roi_features_{band_name_lower}.csv"
        if not top_10_path.exists():
            print(f"⚠️ Skipping {band_name.upper()}: Could not find {top_10_path.name}. Run Script 2 first.")
            continue
            
        top_10_features = pd.read_csv(top_10_path)['Feature'].tolist()

        # B. Haal de overige ROI features op voor DEZE specifieke band
        meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
        X_train_full = train_df.drop(columns=[c for c in meta_cols if c in train_df.columns])

        all_roi_features = []
        for col in X_train_full.columns:
            if f'({band_name.upper()})' in col:  # <--- Dynamische band check
                pair = col.replace(f'({band_name.upper()})', '').split('-')
                if pair[0] in BEST_CHANNELS_EVALUATE and pair[1] in BEST_CHANNELS_EVALUATE:
                    all_roi_features.append(col)

        # C. Selecteer precies de pool van 20
        remaining_roi = [f for f in all_roi_features if f not in top_10_features]
        pool_of_20_features = top_10_features + remaining_roi[:10]
        X_train_roi = X_train_full[pool_of_20_features]

        # ... [Hier komt jouw bestaande mSFFS logica, scaling en K-fold] ...

        # D. Sla alles dynamisch op met de band_name_lower in de bestandsnaam
        plot_msffs_curve(f_counts, np.array(tr_scores), np.array(cv_scores), np.array(cv_stds), band_name_lower)

        stats_path = SVM_DATA_DIR / f"msffs_statistical_summary_{band_name_lower}.csv"
        stats_df.to_csv(stats_path, index=False)

        output_path = SVM_DATA_DIR / f"final_msffs_selected_features_{band_name_lower}.csv"
        output_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_msffs_all_bands()