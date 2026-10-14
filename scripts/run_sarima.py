"""SARIMA(1,0,1)(1,0,1)_24 : validation (ajusté jusqu'à train_end)."""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from src.data_processing import build_dataset
from src.evaluate import score
from src.models import sarima_predict
from src.utils import RESULTS_DIR

ds = build_dataset()
t0 = time.time()
p = sarima_predict(ds.y, ds.time_index, ds.train_end, ds.valid_origins)
vt = ds.truth(ds.valid_origins)
np.save(RESULTS_DIR / "sarima_validation_pred.npy", p)
print("SARIMA validation:", score(vt, p), f"{time.time() - t0:.0f}s")
