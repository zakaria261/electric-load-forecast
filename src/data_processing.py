"""Chargement, agrégation horaire, features calendaires, split temporel et fenêtres."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .utils import (DATA_PATH, EVAL_STRIDE, FEATURES, HORIZON, LOOKBACKS, MIN_MINUTES,
                    SUBMETERS, TRAIN_FRAC, TRAIN_STRIDE, VALID_FRAC)

CAL_COLS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
            "year_sin", "year_cos", "weekend", "fr_holiday"]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Lit le fichier UCI (minute) ; '?' -> NaN ; index temporel trié sans doublon."""
    raw = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str),
                                      format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for c in FEATURES:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    return raw.loc[~raw.index.duplicated(keep="first"), FEATURES]


def describe_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Tableau descriptif : dimensions, période, fréquence, taux de valeurs manquantes."""
    return pd.DataFrame({
        "type": raw.dtypes.astype(str),
        "manquants": raw.isna().sum(),
        "manquants_%": (100 * raw.isna().mean()).round(3),
        "min": raw.min(), "moyenne": raw.mean().round(3), "max": raw.max(),
    })


def to_hourly(raw: pd.DataFrame, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """Agrège à l'heure. Ligne étiquetée 10:00 = intervalle [09:00, 10:00).

    Moyenne pour les grandeurs instantanées, somme pour les sous-compteurs (Wh).
    Une heure avec moins de `min_minutes` mesures valides devient NaN (jamais interpolée).
    """
    rs = raw.resample("h", closed="left", label="right")
    hourly, counts = rs.mean(), rs.count()
    for c in SUBMETERS:
        hourly[c] = raw[c].resample("h", closed="left", label="right").sum(min_count=1)
    hourly, counts = hourly.asfreq("h"), counts.asfreq("h")
    for c in FEATURES:
        hourly.loc[counts[c] < min_minutes, c] = np.nan
    return hourly


def calendar_features(idx) -> pd.DataFrame:
    """Heure, jour de semaine, saison (sin/cos), week-end, jour férié français."""
    idx = pd.DatetimeIndex(idx)
    h, wd, doy = idx.hour.to_numpy(), idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()
    days = np.where(idx.is_leap_year, 366., 365.)
    out = pd.DataFrame(index=idx)
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    out["weekday_sin"], out["weekday_cos"] = np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)
    out["year_sin"] = np.sin(2 * np.pi * (doy - 1) / days)
    out["year_cos"] = np.cos(2 * np.pi * (doy - 1) / days)
    out["weekend"] = (wd >= 5).astype(float)
    try:
        import holidays
        fr = holidays.country_holidays("FR", years=range(idx.min().year - 1, idx.max().year + 2))
        out["fr_holiday"] = [float(t.date() in fr) for t in idx]
    except ImportError:
        out["fr_holiday"] = 0.
    return out[CAL_COLS].astype(float)


class Dataset:
    """Série horaire + calendrier + origines de prévision train/val/test.

    Une origine i = dernière heure connue ; cibles = heures i+1 .. i+24.
    Une fenêtre appartient à la partition de ses 24 cibles (aucune cible ne traverse
    une borne) ; les fenêtres à cibles incomplètes sont écartées.
    """

    def __init__(self, hourly: pd.DataFrame):
        self.hourly = hourly
        self.time_index = hourly.index
        self.y = hourly["Global_active_power"].to_numpy(float)
        self.n = len(hourly)
        self.train_end = int(self.n * TRAIN_FRAC)
        self.valid_end = int(self.n * (TRAIN_FRAC + VALID_FRAC))
        self.cal_array = calendar_features(self.time_index).to_numpy(np.float32)

        first = max(LOOKBACKS) - 1
        # Entraînement : pas fin (1 h). Évaluation : pas de 6 h, aligné sur 0h/6h/12h/18h
        # pour que chaque horizon h couvre plusieurs heures du jour.
        a_tr = np.arange(first, self.n - HORIZON, TRAIN_STRIDE)
        start_ev = next(i for i in range(first, self.n) if self.time_index[i].hour % EVAL_STRIDE == 0)
        a_ev = np.arange(start_ev, self.n - HORIZON, EVAL_STRIDE)
        self.train_origins = self._complete(a_tr[a_tr + HORIZON < self.train_end])
        self.valid_origins = self._complete(
            a_ev[(a_ev + 1 >= self.train_end) & (a_ev + HORIZON < self.valid_end)])
        self.test_origins = self._complete(a_ev[a_ev + 1 >= self.valid_end])
        self.trainval_origins = self._complete(a_tr[a_tr + HORIZON < self.valid_end])

    def _complete(self, origins):
        return np.asarray([i for i in origins
                           if np.isfinite(self.y[i + 1:i + HORIZON + 1]).all()], dtype=int)

    def truth(self, origins) -> np.ndarray:
        return np.stack([self.y[i + 1:i + HORIZON + 1] for i in origins])

    def preprocess(self, fit_end: int):
        """Imputation médiane + standardisation ajustées sur [0, fit_end) seulement (pas de fuite)."""
        imputer = SimpleImputer(strategy="median").fit(self.hourly[FEATURES].iloc[:fit_end])
        filled = imputer.transform(self.hourly[FEATURES])
        scaler = StandardScaler().fit(filled[:fit_end])
        xs = scaler.transform(filled).astype(np.float32)
        mean = float(np.nanmean(self.y[:fit_end]))
        std = float(np.nanstd(self.y[:fit_end]))
        return xs, mean, (std if std > 0 else 1.)

    def windows(self, origins, lookback, xs, mean, std, use_calendar=True):
        """Retourne (historique, calendrier futur, cibles normalisées)."""
        xh, xf, yy = [], [], []
        for i in origins:
            hist = xs[i - lookback + 1:i + 1]
            if use_calendar:
                hist = np.column_stack([hist, self.cal_array[i - lookback + 1:i + 1]])
                future = self.cal_array[i + 1:i + HORIZON + 1]
            else:
                future = np.zeros((HORIZON, len(CAL_COLS)), np.float32)
            xh.append(hist); xf.append(future)
            yy.append((self.y[i + 1:i + HORIZON + 1] - mean) / std)
        return tuple(np.asarray(v, np.float32) for v in (xh, xf, yy))


def build_dataset(path=DATA_PATH) -> Dataset:
    return Dataset(to_hourly(load_raw(path)))
