"""Analyse exploratoire sur l'entraînement du protocole annuel. Figures -> results/figures/."""
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
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data_processing import calendar_features, describe_raw, load_raw, to_hourly, Dataset
from src.utils import FEATURES, RESULTS_DIR

FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=110)
    plt.close()


raw = load_raw()
hourly = to_hourly(raw)
ds = Dataset(hourly, split_mode="annual")
train = hourly["Global_active_power"].iloc[:ds.train_end]

# --- Tableau descriptif
desc = describe_raw(raw)
desc.to_csv(RESULTS_DIR / "dataset_description.csv")
print(f"Observations : {len(raw):,} | features : {len(FEATURES)} | fréquence : 1 min")
print(f"Période : {raw.index.min()} -> {raw.index.max()}")
print(f"Heures : {len(hourly):,} | cible manquante : {100*np.isnan(ds.y).mean():.2f} %")
print(desc.to_string())

# --- 1. Série brute
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
train.resample("D").mean().plot(ax=ax[0, 0], title="Moyenne journalière (kW)")
train.iloc[:24 * 14].plot(ax=ax[0, 1], title="Deux semaines horaires (kW)")
train.dropna().plot.hist(bins=60, ax=ax[1, 0], title="Distribution horaire (kW)")
train.groupby(train.index.hour).mean().plot(ax=ax[1, 1], marker="o", title="Profil moyen par heure")
save("01_serie_distribution_profil.png")

# --- 2. Cycles hebdo + heure x jour
wk = train.groupby([train.index.dayofweek, train.index.hour]).mean().unstack(0)
wk.columns = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
plt.figure(figsize=(9, 5))
sns.heatmap(wk, cmap="rocket_r", cbar_kws={"label": "kW"})
plt.title("Puissance moyenne : heure x jour de semaine")
save("02_heure_x_jour.png")

monthly = train.resample("MS").mean()
print("\nMoyenne par mois calendaire (kW):")
print(train.groupby(train.index.month).mean().round(3).to_string())

# --- 3. Décompositions
daily = train.resample("D").mean().interpolate(limit=2).dropna()
seasonal_decompose(daily, model="additive", period=7).plot()
plt.suptitle("Décomposition hebdomadaire (moyenne journalière)", y=1.0)
save("03_decomp_hebdo.png")
m = monthly.interpolate().dropna()
seasonal_decompose(m, model="additive", period=12).plot()
plt.suptitle("Décomposition annuelle (moyenne mensuelle)", y=1.0)
save("04_decomp_annuelle.png")

# --- 4. Corrélations + ACF
plt.figure(figsize=(8, 6.5))
sns.heatmap(hourly.iloc[:ds.train_end][FEATURES].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Corrélations contemporaines — entraînement")
save("05_correlations.png")

s = train.interpolate(limit=6).dropna()
fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(s, lags=24 * 8, ax=ax)
ax.set_title("Autocorrélation de la consommation horaire (pics à 24 h, 168 h)")
save("06_acf.png")
acf = pd.Series([s.autocorr(l) for l in (1, 12, 24, 48, 72, 168)], index=[1, 12, 24, 48, 72, 168])
print("\nACF aux retards (h):"); print(acf.round(3).to_string())

# --- 5. Effet calendrier
cal = calendar_features(train.index)
d = pd.DataFrame({"kw": train, "weekend": cal["weekend"], "ferie": cal["fr_holiday"]}).dropna()
print("\nMoyenne kW selon calendrier:")
print(d.groupby(["weekend", "ferie"])["kw"].agg(["mean", "count"]).round(3).to_string())
plt.figure(figsize=(7, 4))
sns.boxplot(data=d.assign(type=np.where(d.ferie == 1, "Férié", np.where(d.weekend == 1, "Week-end", "Semaine"))),
            x="type", y="kw", showfliers=False)
plt.title("Consommation horaire par type de jour")
save("07_type_jour.png")

# --- Part des trous
na = pd.Series(ds.y, index=ds.time_index).isna()
print(f"\nHeures NaN dans train : {na.iloc[:ds.train_end].sum()}, val : "
      f"{na.iloc[ds.train_end:ds.valid_end].sum()}, test : {na.iloc[ds.valid_end:].sum()}")
print(f"Seuil pic (P95 train) : {np.nanpercentile(train, 95):.3f} kW")
