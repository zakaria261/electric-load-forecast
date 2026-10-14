"""ARCHIVE : protocole historique 70/15/15, scores non comparables au protocole annuel.

Benchmark LSTM / TCN x historique 24/48/168 h : entraînement sur train, choix sur validation."""
import json
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
from src.utils import LOOKBACKS, RESULTS_DIR, set_seed

set_seed()
(RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
ds = build_dataset(split_mode="legacy")
xs, mean, std = ds.preprocess(ds.train_end)          # ajustés sur le train seulement
vt = ds.truth(ds.valid_origins)

rows, preds = [], {}
for kind in ["LSTM", "TCN"]:
    for lb in LOOKBACKS:
        name = f"{kind} | {lb} h | calendrier"
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        model = fit_net(kind, lb, ds, ds.train_origins, xs, mean, std, use_calendar=True,
                        val_origins=ds.valid_origins)
        p = predict_net(model, lb, ds, ds.valid_origins, xs, mean, std)
        preds[name] = p
        rows.append({"modele": name, "best_epoch": model.best_epoch,
                     "temps_s": round(time.time() - t0), **score(vt, p)})
        print(f"-> {rows[-1]}", flush=True)
        import torch
        torch.save(model.state_dict(), RESULTS_DIR / "models" / f"{kind}_{lb}h.pt")
        # sauvegarde incrémentale
        pd.DataFrame(rows).set_index("modele").sort_values("MAE_kW").to_csv(RESULTS_DIR / "neural_validation.csv")
        np.savez(RESULTS_DIR / "neural_validation_preds.npz", **{k.replace(" ", "_"): v for k, v in preds.items()})

print("\n=== RÉSUMÉ VALIDATION ===")
print(pd.DataFrame(rows).set_index("modele").sort_values("MAE_kW").round(4).to_string())
json.dump({"mean": mean, "std": std}, open(RESULTS_DIR / "neural_scaler.json", "w"))
