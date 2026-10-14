"""Modèles de prévision à 24 h : baselines, SARIMA, LSTM, TCN (PyTorch)."""
import numpy as np
import pandas as pd

from .utils import HORIZON

try:
    import torch
    from torch import nn
except (ImportError, OSError):      # torch absent/cassé : baselines et SARIMA restent utilisables
    torch = nn = None

BASELINES = ["Persistance", "Moyenne mobile 24h", "Saisonnier quotidien", "Saisonnier hebdomadaire"]


def baseline_predict(name: str, origins, y: np.ndarray, fallback: float) -> np.ndarray:
    """Prévision (n_origines, 24) par baseline. N'utilise que y[<= origine] (aucune fuite).

    - Persistance : dernière valeur connue répétée.
    - Moyenne mobile 24h : moyenne des 24 dernières heures connues, constante.
    - Saisonnier quotidien / hebdomadaire : même heure à J-1 / J-7 (lag >= 24 donc
      l'indice i+h-lag reste <= i pour tout horizon h <= 24).
    Une valeur historique manquante est remplacée par `fallback` (médiane du train).
    """
    origins = np.asarray(origins)
    h = np.arange(1, HORIZON + 1)
    if name == "Persistance":
        pred = np.repeat(y[origins, None], HORIZON, axis=1)
    elif name == "Moyenne mobile 24h":
        ma = np.array([np.nanmean(y[i - 23:i + 1]) if np.isfinite(y[i - 23:i + 1]).any() else np.nan
                       for i in origins])
        pred = np.repeat(ma[:, None], HORIZON, axis=1)
    elif name in ("Saisonnier quotidien", "Saisonnier hebdomadaire"):
        lag = 24 if name == "Saisonnier quotidien" else 168
        ix = origins[:, None] + h[None, :] - lag
        assert (ix >= 0).all() and (ix <= origins[:, None]).all()
        pred = y[ix].astype(float)
    else:
        raise ValueError(f"baseline inconnue : {name}")
    pred = np.where(np.isfinite(pred), pred, fallback)
    return np.maximum(pred, 0.)


def sarima_predict(y, time_index, fit_end: int, origins, fit_days: int = 120) -> np.ndarray:
    """SARIMA(1,0,1)(1,0,1)_24 ajusté sur les `fit_days` derniers jours avant `fit_end`.

    Les paramètres restent fixes. À chaque origine (croissante), l'état est mis à jour avec
    les observations jusqu'à l'origine (sans refit), puis on prévoit 24 h. Les valeurs futures
    ne sont jamais ajoutées : valable pour des origines qui se chevauchent (pas de 6 h).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    start = max(0, fit_end - fit_days * 24)
    series = pd.Series(y[start:fit_end], index=time_index[start:fit_end])
    fitted = SARIMAX(series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=40)
    last, out = fit_end - 1, []
    for o in origins:
        if o > last:
            upd = pd.Series(y[last + 1:o + 1], index=time_index[last + 1:o + 1])
            fitted = fitted.append(upd, refit=False)
            last = o
        out.append(np.maximum(np.asarray(fitted.get_forecast(HORIZON).predicted_mean), 0))
    return np.stack(out)


if torch is not None:
    class LSTMForecaster(nn.Module):
        """LSTM sur l'historique ; la tête reçoit l'état final + calendrier des 24 h futures."""
        def __init__(self, d_in: int, d_future: int, hidden: int = 32):
            super().__init__()
            self.rnn = nn.LSTM(d_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + HORIZON * d_future, 64), nn.ReLU(),
                                      nn.Linear(64, HORIZON))

        def forward(self, x, f):
            z, _ = self.rnn(x)
            return self.head(torch.cat([z[:, -1], f.flatten(1)], 1))

    class _Block(nn.Module):
        """Bloc TCN causal : 2 convolutions dilatées + connexion résiduelle."""
        def __init__(self, c_in: int, c_out: int, dilation: int):
            super().__init__()
            pad = 2 * dilation      # padding puis troncature à droite => causalité stricte
            self.a = nn.Conv1d(c_in, c_out, 3, padding=pad, dilation=dilation)
            self.b = nn.Conv1d(c_out, c_out, 3, padding=pad, dilation=dilation)
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            t = x.shape[-1]
            z = torch.relu(self.a(x)[:, :, :t])
            z = torch.relu(self.b(z)[:, :, :t])
            return torch.relu(z + self.skip(x))

    class TCNForecaster(nn.Module):
        """3 blocs dilatés (1,2,4) : champ réceptif 29 h ; lecture du dernier pas temporel."""
        def __init__(self, d_in: int, d_future: int, channels: int = 32):
            super().__init__()
            self.conv = nn.Sequential(_Block(d_in, channels, 1), _Block(channels, channels, 2),
                                      _Block(channels, channels, 4))
            self.head = nn.Sequential(nn.Linear(channels + HORIZON * d_future, 64), nn.ReLU(),
                                      nn.Linear(64, HORIZON))

        def forward(self, x, f):
            z = self.conv(x.transpose(1, 2))[:, :, -1]
            return self.head(torch.cat([z, f.flatten(1)], 1))


def build_net(kind: str, d_in: int, d_future: int):
    if torch is None:
        raise RuntimeError("torch indisponible : réparer l'installation avant d'entraîner LSTM/TCN")
    return {"LSTM": LSTMForecaster, "TCN": TCNForecaster}[kind](d_in, d_future)
