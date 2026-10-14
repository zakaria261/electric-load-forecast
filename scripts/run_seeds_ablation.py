"""4 configs x 3 graines x {avec, sans} calendrier. Sélection sur validation uniquement."""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_processing import build_dataset
from src.evaluate import score
from src.train import fit_net, predict_net
from src.utils import RESULTS_DIR

SEEDS = [0, 1, 2]
CONFIGS = [("LSTM", 24), ("LSTM", 168), ("TCN", 24), ("TCN", 168)]
ds = build_dataset()
xs, mean, std = ds.preprocess(ds.train_end)
vt = ds.truth(ds.valid_origins)

rows, preds = [], {}
out = RESULTS_DIR / "seeds_ablation_validation.csv"
npz = RESULTS_DIR / "seeds_ablation_preds.npz"
if out.exists() and npz.exists():          # reprise après interruption
    rows = pd.read_csv(out).to_dict("records")
    with np.load(npz) as z:
        preds = {k: z[k] for k in z.files}
done = {(r["modele"], bool(r["calendrier"]), int(r["seed"])) for r in rows}
for use_cal in [True, False]:
    for kind, lb in CONFIGS:
        for seed in SEEDS:
            if (f"{kind} | {lb} h", use_cal, seed) in done:
                continue
            t0 = time.time()
            m = fit_net(kind, lb, ds, ds.train_origins, xs, mean, std, use_calendar=use_cal,
                        val_origins=ds.valid_origins, seed=seed, verbose=False)
            p = predict_net(m, lb, ds, ds.valid_origins, xs, mean, std, use_calendar=use_cal)
            preds[f"{kind}_{lb}_{'cal' if use_cal else 'nocal'}_s{seed}"] = p
            rows.append({"modele": f"{kind} | {lb} h", "calendrier": use_cal, "seed": seed,
                         "best_epoch": m.best_epoch, **score(vt, p)})
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
            np.savez(RESULTS_DIR / "seeds_ablation_preds.npz", **preds)

df = pd.DataFrame(rows)
agg = df.groupby(["calendrier", "modele"])[["MAE_kW", "RMSE_kW", "MAPE_%"]].agg(["mean", "std"]).round(4)
print("\n=== MOYENNE ± ÉCART-TYPE (3 graines) ==="); print(agg.to_string())
agg.to_csv(RESULTS_DIR / "seeds_ablation_summary.csv")
