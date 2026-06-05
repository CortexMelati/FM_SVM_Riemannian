
### Stap 1 t/m 3: Data Voorbereiding (De Fundering)

* **Stap 1:** Mappenstructuur voor `CP_FM_dataset`. Make sure to delete (not include) the maps deratives\cartool as we don't need it.
* **Stap 2:** `.mat` naar `.vhdr` conversie.
* **Stap 3:** TDBRAIN whitelisten (zodat ADHD/Depressie patiënten worden genegeerd).


### Stap 4: Riemannian Preprocessing (Het Schoonmaken)

* Filteren (0.5-44 Hz, 50 Hz Notch), in 1-seconde epochs knippen, en opslaan als brandschone `.fif` bestanden.
* Plotting function included


### Stap 5: De Feature Pipelines 

* **Pipeline A (Jouw Riemannian Aanpak):** Leest de 1s `.fif` epochs in **$\rightarrow$** Isoleert Gamma-band & 9 centrale elektroden **$\rightarrow$** Berekent 9x9 SCM's **$\rightarrow$** Projecteert naar Tangent Space **$\rightarrow$** Slaat op als `.csv` / `.npy`.
* **Pipeline B (De Li et al. Replicatie):** Leest de 1s `.fif` epochs in **$\rightarrow$** **Groepeert ze in blokken van 30 seconden** **$\rightarrow$** Berekent de Spectrale Coherentie tussen de 19 kanalen over 5 frequentiebanden **$\rightarrow$** Levert 855 features op **$\rightarrow$** Slaat op als `.csv` / `.npy`.


### Stap 6: De 10% Hold-out Set 

* 6 subjects for CP_FM_dataset

### Stap 7: Machine Learning & mSFFS 

* De resterende ~39 subjecten gaan de ML-pipeline in.
* We gebruiken inderdaad  **Group K-Fold Cross Validation** . Dit is cruciaal! Dit zorgt ervoor dat alle epochs/blokken van Patiënt X áltijd samen in de train- of test-vouw belanden, wat de gevreesde *data leakage* voorkomt.
* We passen mSFFS toe voor feature selection (om de ruis weg te snijden).
* We trainen de SVM met een RBF-kernel en zoeken de optimale parameters (zoals gamma op 5.1053) via Grid Search. 0.0001 - 30

### Stap 8: Visualisaties & SHAP
