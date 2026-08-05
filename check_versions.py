import importlib

# Alle externe packages uit jouw code
packages = {
    "MNE-Python": "mne",
    "MNE-Connectivity": "mne_connectivity",
    "PyRiemann": "pyriemann",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "NumPy": "numpy",
    "SciPy": "scipy",
    "statsmodels": "statsmodels",
    "XGBoost": "xgboost",
    "Matplotlib": "matplotlib",
    "Seaborn": "seaborn",
    "SHAP": "shap",
    "mlxtend": "mlxtend",
    "ADAPT": "adapt",
    "joblib": "joblib",
    "tqdm": "tqdm"
}

print(f"{'Package':<20} | {'Version'}")
print("-" * 35)

for display_name, module_name in packages.items():
    try:
        module = importlib.import_module(module_name)
        # Meestal zit de versie in __version__, maar we vangen uitzonderingen af
        version = getattr(module, '__version__', 'Geen __version__ attribuut')
        print(f"{display_name:<20} | {version}")
    except ImportError:
        print(f"{display_name:<20} | NIET GEÏNSTALLEERD!")