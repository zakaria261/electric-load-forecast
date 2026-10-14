"""Évalue les baselines sur validation (choix) puis test (une seule fois, pour référence)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_processing import build_dataset
from src.evaluate import mae_by_horizon, peak_scores, score_table, season_scores
from src.models import BASELINES, baseline_predict
from src.utils import RESULTS_DIR, set_seed

set_seed()
FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ds = build_dataset()
y = ds.y

fb_val = float(np.nanmedian(y[:ds.train_end]))          # fallback = médiane du train
fb_test = float(np.nanmedian(y[:ds.valid_end]))         # fallback = médiane train+val
vt, tt = ds.truth(ds.valid_origins), ds.truth(ds.test_origins)
vp = {k: baseline_predict(k, ds.valid_origins, y, fb_val) for k in BASELINES}
tp = {k: baseline_predict(k, ds.test_origins, y, fb_test) for k in BASELINES}

val_tab, test_tab = score_table(vt, vp), score_table(tt, tp)
print("=== VALIDATION ==="); print(val_tab.round(4).to_string())
print("\n=== TEST (référence) ==="); print(test_tab.round(4).to_string())
val_tab.to_csv(RESULTS_DIR / "baselines_validation.csv")
test_tab.to_csv(RESULTS_DIR / "baselines_test.csv")

best = val_tab.index[0]
print("\nMeilleure baseline (validation, MAE) :", best)
thr = float(np.nanquantile(y[:ds.train_end], .95))
print(f"\nPics (cible >= {thr:.2f} kW), validation, {best}:")
print(peak_scores(vt, vp[best], thr).round(4).to_string())
print("\nPar saison (test):")
print(season_scores(tt, tp[best], ds.time_index[ds.test_origins]).round(4).to_string())

h = pd.DataFrame({k: mae_by_horizon(vt, v) for k, v in vp.items()})
print("\nMAE par horizon (validation), h=1,6,12,18,24:")
print(h.loc[[1, 6, 12, 18, 24]].round(3).to_string())
h.plot(marker="o", figsize=(10, 4.5), title="Baselines — MAE de validation par horizon")
plt.xlabel("Horizon (h)"); plt.ylabel("kW"); plt.xticks(range(1, 25)); plt.tight_layout()
plt.savefig(FIG / "08_baselines_mae_horizon.png", dpi=110)
