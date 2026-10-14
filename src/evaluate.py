"""Métriques et analyses d'erreur (MAE, RMSE, MAPE ; par horizon, pics, saison)."""
import numpy as np
import pandas as pd


def score(actual, pred) -> dict:
    """MAE et RMSE en kW ; MAPE (%) calculée sur les cibles non nulles seulement."""
    a, p = np.asarray(actual).ravel(), np.asarray(pred).ravel()
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    nz = np.abs(a) > 1e-6
    return {"MAE_kW": float(np.mean(np.abs(a - p))),
            "RMSE_kW": float(np.sqrt(np.mean((a - p) ** 2))),
            "MAPE_%": float(100 * np.mean(np.abs((a[nz] - p[nz]) / a[nz]))) if nz.any() else np.nan,
            "n": int(len(a))}


def score_table(truth, preds: dict) -> pd.DataFrame:
    """Tableau comparatif trié par MAE."""
    return pd.DataFrame({k: score(truth, v) for k, v in preds.items()}).T.sort_values("MAE_kW")


def mae_by_horizon(truth, pred) -> pd.Series:
    """MAE pour chaque horizon 1..24 h."""
    return pd.Series(np.nanmean(np.abs(truth - pred), axis=0),
                     index=np.arange(1, truth.shape[1] + 1), name="MAE_kW")


def peak_scores(truth, pred, threshold: float) -> pd.DataFrame:
    """Scores globaux vs sur les heures de pic (cible >= seuil, ex. P95 du train)."""
    peaks = truth >= threshold
    return pd.DataFrame([{"segment": "Toutes", **score(truth, pred)},
                         {"segment": "Pics", **score(truth[peaks], pred[peaks])}]).set_index("segment")


def season_scores(truth, pred, origin_times) -> pd.DataFrame:
    """Scores par saison météorologique de l'origine (hiver DJF, printemps, été, automne)."""
    names = {12: "Hiver", 1: "Hiver", 2: "Hiver", 3: "Printemps", 4: "Printemps", 5: "Printemps",
             6: "Été", 7: "Été", 8: "Été", 9: "Automne", 10: "Automne", 11: "Automne"}
    seas = np.array([names[t.month] for t in origin_times])
    rows = {s: score(truth[seas == s], pred[seas == s]) for s in pd.unique(seas)}
    return pd.DataFrame(rows).T


def empirical_interval(val_truth, val_pred, test_pred, level: float = 0.90):
    """Intervalle = prévision ± quantile des erreurs absolues de validation, par horizon."""
    q = np.quantile(np.abs(val_truth - val_pred), level, axis=0)
    return np.maximum(0, test_pred - q), test_pred + q


def coverage(truth, lo, hi) -> float:
    ok = np.isfinite(truth)
    return float(np.mean((truth >= lo)[ok] & (truth <= hi)[ok]))
