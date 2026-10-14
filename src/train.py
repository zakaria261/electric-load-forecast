"""Entraînement et inférence des réseaux (LSTM / TCN)."""
import copy

import numpy as np

from .models import build_net, torch, nn
from .utils import SEED, set_seed


def _tensors(ds, origins, lookback, xs, mean, std, use_calendar):
    x, f, y = ds.windows(origins, lookback, xs, mean, std, use_calendar)
    return torch.tensor(x), torch.tensor(f), torch.tensor(y)


def fit_net(kind, lookback, ds, train_origins, xs, mean, std, use_calendar=True,
            val_origins=None, epochs=20, patience=4, lr=1e-3, batch_size=128, verbose=True,
            seed=SEED):
    """Entraîne un réseau (MSE sur cibles standardisées, Adam).

    Avec `val_origins` : early stopping sur la perte de validation (on garde le meilleur
    état) et `model.best_epoch` est renseigné. Sans validation : entraîne exactement `epochs`
    époques (utile pour le ré-entraînement train+val avec `best_epoch` connu).
    """
    set_seed(seed)
    x, f, y = _tensors(ds, train_origins, lookback, xs, mean, std, use_calendar)
    model = build_net(kind, x.shape[-1], f.shape[-1])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, f, y), batch_size=batch_size,
                                         shuffle=True, generator=torch.Generator().manual_seed(seed))
    if val_origins is not None:
        xv, fv, yv = _tensors(ds, val_origins, lookback, xs, mean, std, use_calendar)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad, model.best_epoch = np.inf, None, 0, epochs
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.
        for xb, fb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        msg = f"  [{kind} {lookback}h] epoch {ep:2d} train {tot / len(x):.4f}"
        if val_origins is not None:
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(xv, fv), yv).item()
            msg += f" val {vloss:.4f}"
            if vloss < best - 1e-4:
                best, best_state, bad, model.best_epoch = vloss, copy.deepcopy(model.state_dict()), 0, ep
            else:
                bad += 1
        if verbose:
            print(msg, flush=True)
        if val_origins is not None and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_net(model, lookback, ds, origins, xs, mean, std, use_calendar=True):
    """Prévisions en kW, shape (n_origines, 24), bornées à 0."""
    x, f, _ = _tensors(ds, origins, lookback, xs, mean, std, use_calendar)
    model.eval()
    with torch.no_grad():
        pred = model(x, f).numpy()
    return np.maximum(pred * std + mean, 0)
