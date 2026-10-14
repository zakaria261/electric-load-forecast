"""Test final : une seule évaluation des finalistes fixés sur la validation.

Finalistes (annoncés avant le test) : SARIMA (meilleure MAE de validation) et
TCN 168 h + calendrier (meilleur réseau). Référence : saisonnier hebdomadaire (J-7).
"""
import sys
import time
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
from src.evaluate import (coverage, empirical_interval, mae_by_horizon, peak_scores, score,
                          season_scores)
from src.models import baseline_predict, sarima_predict
from src.train import fit_net, predict_net
from src.utils import HORIZON, RESULTS_DIR, set_seed

set_seed()
FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]

ds = build_dataset()
y, ti = ds.y, ds.time_index
vt, tt = ds.truth(ds.valid_origins), ds.truth(ds.test_origins)
thr = float(np.nanquantile(y[:ds.train_end], .95))          # seuil de pic : P95 du train

# ---------- prédictions de validation (calibration des intervalles), modèles entraînés sur train
with np.load(RESULTS_DIR / "seeds_ablation_preds.npz") as z:
    val_tcn = np.mean([z[f"TCN_168_cal_s{s}"] for s in SEEDS], axis=0)
val_sarima = np.load(RESULTS_DIR / "sarima_validation_pred.npy")
val_j7 = baseline_predict("Saisonnier hebdomadaire", ds.valid_origins, y,
                          float(np.nanmedian(y[:ds.train_end])))
val = {"TCN 168h": val_tcn, "SARIMA": val_sarima, "J-7": val_j7}

# ---------- époques retenues sur validation (moyenne des meilleures époques, TCN 168 h + calendrier)
sv = pd.read_csv(RESULTS_DIR / "seeds_ablation_validation.csv")
epochs = int(round(sv[(sv.modele == "TCN | 168 h") & (sv.calendrier == True)].best_epoch.mean()))
print("Époques (validation) :", epochs, flush=True)

# ---------- test : TCN réentraîné sur train+val, 3 graines
xs, mean, std = ds.preprocess(ds.valid_end)
tcn_runs = []
for s in SEEDS:
    t0 = time.time()
    m = fit_net("TCN", 168, ds, ds.trainval_origins, xs, mean, std, use_calendar=True,
                epochs=epochs, seed=s, verbose=False)
    tcn_runs.append(predict_net(m, 168, ds, ds.test_origins, xs, mean, std))
    print(f"TCN seed {s}: MAE {score(tt, tcn_runs[-1])['MAE_kW']:.4f} ({time.time() - t0:.0f}s)", flush=True)
test_tcn = np.mean(tcn_runs, axis=0)                          # moyenne des 3 graines

# ---------- test : SARIMA (ajusté sur les 120 j avant la fin de la validation) et J-7
t0 = time.time()
test_sarima = sarima_predict(y, ti, ds.valid_end, ds.test_origins)
print(f"SARIMA test ({time.time() - t0:.0f}s)", flush=True)
test_j7 = baseline_predict("Saisonnier hebdomadaire", ds.test_origins, y, float(np.nanmedian(y[:ds.valid_end])))
test = {"TCN 168h": test_tcn, "SARIMA": test_sarima, "J-7": test_j7}
np.savez(RESULTS_DIR / "test_predictions.npz", truth=tt, origins=ds.test_origins, **{k.replace(" ", "_"): v for k, v in test.items()})

# ---------- tableaux
tab = pd.DataFrame({k: score(tt, v) for k, v in test.items()}).T
seed_tab = pd.DataFrame([score(tt, p) for p in tcn_runs], index=[f"seed {s}" for s in SEEDS])
print("\n=== TEST — scores globaux ==="); print(tab.round(4).to_string())
print("\n=== TCN 168h par graine ==="); print(seed_tab.round(4).to_string())
print(f"TCN seeds MAE: {seed_tab.MAE_kW.mean():.4f} ± {seed_tab.MAE_kW.std():.4f}")
tab.to_csv(RESULTS_DIR / "test_scores.csv")

hz = pd.DataFrame({k: mae_by_horizon(tt, v) for k, v in test.items()})
hz.to_csv(RESULTS_DIR / "test_mae_by_horizon.csv")
print("\n=== MAE par horizon (h=1,3,6,12,18,24) ==="); print(hz.loc[[1, 3, 6, 12, 18, 24]].round(3).to_string())

print(f"\n=== PICS (cible >= {thr:.2f} kW) ===")
peak_rows = {k: peak_scores(tt, v, thr).loc["Pics"] for k, v in test.items()}
print(pd.DataFrame(peak_rows).T.round(4).to_string())
print(f"part des valeurs en pic dans le test : {100 * np.mean(tt >= thr):.1f} %")

seas = {k: season_scores(tt, v, ti[ds.test_origins]) for k, v in test.items()}
print("\n=== MAE par saison (test) ===")
print(pd.DataFrame({k: v["MAE_kW"] for k, v in seas.items()}).round(4).to_string())

# erreur selon l'heure de la cible
tgt_hours = np.array([[ti[i + h].hour for h in range(1, HORIZON + 1)] for i in ds.test_origins])
by_hour = pd.DataFrame({k: pd.Series(np.abs(tt - v).ravel()).groupby(tgt_hours.ravel()).mean()
                        for k, v in test.items()})
print("\n=== MAE par heure de la cible (h=0,4,8,12,16,20) ==="); print(by_hour.loc[[0, 4, 8, 12, 16, 20]].round(3).to_string())

# ---------- intervalles 90 % (quantile des erreurs absolues de validation, par horizon)
print("\n=== Intervalles 90 % — couverture empirique sur le test ===")
cov = {}
for k in test:
    lo, hi = empirical_interval(vt, val[k], test[k], .90)
    cov[k] = {"couverture_%": 100 * coverage(tt, lo, hi), "largeur_moy_kW": float(np.mean(hi - lo))}
print(pd.DataFrame(cov).T.round(2).to_string())

# ---------- figures
palette = {"TCN 168h": "#1f77b4", "SARIMA": "#d95f02", "J-7": "#7f7f7f"}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for k in test:
    ax[0].plot(hz.index, hz[k], marker="o", label=k, color=palette[k])
    ax[1].plot(by_hour.index, by_hour[k], marker="o", label=k, color=palette[k])
ax[0].set(title="Test — MAE par horizon de prévision", xlabel="Horizon (h)", ylabel="MAE (kW)")
ax[1].set(title="Test — MAE par heure de la journée cible", xlabel="Heure", ylabel="MAE (kW)")
ax[0].legend(); plt.tight_layout(); plt.savefig(FIG / "09_test_horizon_heure.png", dpi=110); plt.close()

# exemples de prévisions avec intervalle : 4 origines (été/automne), modèle TCN et SARIMA
sel = np.linspace(0, len(ds.test_origins) - 1, 4).astype(int)
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
for r, k in enumerate(["SARIMA", "TCN 168h"]):
    lo, hi = empirical_interval(vt, val[k], test[k], .90)
    for c, j in enumerate(sel):
        i = int(ds.test_origins[j]); dates = ti[i + 1:i + HORIZON + 1]
        a = axes[r, c]
        a.plot(dates, tt[j], "k-o", ms=3, label="Observé")
        a.plot(dates, test[k][j], "-o", ms=3, color=palette[k], label=f"Prévu {k}")
        a.fill_between(dates, lo[j], hi[j], color=palette[k], alpha=.2, label="IC 90 %")
        a.set_title(f"{k} — origine {ti[i]:%d/%m %Hh}", fontsize=9)
        a.tick_params(axis="x", rotation=45, labelsize=7)
axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel("kW"); axes[1, 0].set_ylabel("kW")
plt.tight_layout(); plt.savefig(FIG / "10_test_previsions_ic.png", dpi=110); plt.close()

# pires origines (erreur MAE par journée de 24 h) pour le meilleur modèle en MAE
best = tab.MAE_kW.idxmin()
err = np.nanmean(np.abs(tt - test[best]), axis=1)
worst = np.argsort(err)[::-1][:5]
print(f"\n=== 5 pires origines ({best}) ===")
for j in worst:
    print(ti[ds.test_origins[j]], f"MAE {err[j]:.3f} kW", f"pic obs {np.nanmax(tt[j]):.2f} kW")

plt.figure(figsize=(6, 6))
for k in ["SARIMA", "TCN 168h"]:
    plt.scatter(tt.ravel(), test[k].ravel(), s=2, alpha=.2, label=k, color=palette[k])
mx = np.nanmax(tt); plt.plot([0, mx], [0, mx], "k--", lw=1)
plt.axvline(thr, color="red", lw=.8, ls=":"); plt.xlabel("Observé (kW)"); plt.ylabel("Prévu (kW)")
plt.title("Test — prévu vs observé (pointillé rouge : seuil de pic)"); plt.legend(markerscale=6)
plt.tight_layout(); plt.savefig(FIG / "11_test_scatter.png", dpi=110); plt.close()
print("\nFigures et tableaux enregistrés dans results/")
