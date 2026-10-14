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
    """Scores par saison de chaque heure cible (hiver DJF, printemps, été, automne).

    Une prévision émise le 31 août peut couvrir des heures de septembre : chaque
    couple origine-horizon est classé selon la date de sa cible, comme dans le notebook.
    """
    names = {12: "Hiver", 1: "Hiver", 2: "Hiver", 3: "Printemps", 4: "Printemps", 5: "Printemps",
             6: "Été", 7: "Été", 8: "Été", 9: "Automne", 10: "Automne", 11: "Automne"}
    truth, pred = np.asarray(truth), np.asarray(pred)
    if truth.shape != pred.shape or truth.ndim != 2 or len(origin_times) != len(truth):
        raise ValueError("Cibles, prévisions et origines de formes incompatibles.")
    targets = pd.DatetimeIndex(origin_times).to_numpy()[:, None] + np.arange(
        1, truth.shape[1] + 1)[None, :] * np.timedelta64(1, "h")
    seasons = np.array([names[month] for month in pd.DatetimeIndex(targets.ravel()).month])
    seasons = seasons.reshape(truth.shape)
    rows = {s: score(truth[seasons == s], pred[seasons == s])
            for s in ("Hiver", "Printemps", "Été", "Automne") if np.any(seasons == s)}
    return pd.DataFrame(rows).T


def empirical_interval(val_truth, val_pred, test_pred, level: float = 0.90):
    """Intervalle = prévision ± quantile des erreurs absolues de validation, par horizon."""
    q = np.quantile(np.abs(val_truth - val_pred), level, axis=0)
    return np.maximum(0, test_pred - q), test_pred + q


def coverage(truth, lo, hi) -> float:
    ok = np.isfinite(truth)
    return float(np.mean((truth >= lo)[ok] & (truth <= hi)[ok]))


def paired_weekly_mae_interval(truth, first_pred, second_pred, origin_times,
                               draws: int = 5000, seed: int = 42) -> dict:
    """Écart de MAE (premier - second) et intervalle descriptif par semaines.

    Les 24 horizons d'une origine restent ensemble et les semaines d'émission
    sont rééchantillonnées pour tenir compte du chevauchement des fenêtres.
    La sélection préalable du modèle n'est pas corrigée par cet intervalle.
    """
    truth, first_pred, second_pred = map(np.asarray, (truth, first_pred, second_pred))
    if (truth.shape != first_pred.shape or truth.shape != second_pred.shape
            or truth.ndim != 2 or len(origin_times) != len(truth)):
        raise ValueError("Formes incompatibles pour la comparaison appariée.")
    origin_error = (np.mean(np.abs(truth - first_pred), axis=1)
                    - np.mean(np.abs(truth - second_pred), axis=1))
    weeks = pd.DatetimeIndex(origin_times).to_period("W")
    grouped = pd.DataFrame({"week": weeks, "difference": origin_error}).groupby(
        "week")["difference"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    sampled = grouped[rng.integers(0, len(grouped), size=(draws, len(grouped)))]
    estimates = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    lower, upper = np.quantile(estimates, [.025, .975])
    return {"difference_MAE_kW": float(origin_error.mean()),
            "borne_2_5_kW": float(lower), "borne_97_5_kW": float(upper),
            "semaines": int(len(grouped))}
