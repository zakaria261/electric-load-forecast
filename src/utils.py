"""Constantes du projet et fonctions utilitaires (seed, chemins)."""
import os
import random
from pathlib import Path

import numpy as np

SEED = 42
HORIZON = 24                  # prévision des 24 prochaines heures
LOOKBACKS = [24, 48, 168]     # historiques comparés : 24 h, 48 h, 7 jours
TRAIN_FRAC, VALID_FRAC = 0.70, 0.15
ANNUAL_VALID_YEARS = 1
ANNUAL_TEST_YEARS = 1
ROLLING_FOLD_MONTHS = 3
MIN_MINUTES = 45              # mesures valides minimales pour valider une heure
TRAIN_STRIDE = 1              # entraînement : une origine par heure (~24 000 fenêtres, tous les cycles)
EVAL_STRIDE = 6               # validation/test : origines à 0h, 6h, 12h, 18h (4 phases du cycle)

FEATURES = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
SUBMETERS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "household_power_consumption.txt"
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int = SEED) -> None:
    """Fixe les graines (random, numpy, torch si installé) pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):   # torch absent ou DLL non chargeable : baselines/SARIMA OK
        pass
