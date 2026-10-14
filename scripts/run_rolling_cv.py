"""Validation croisée temporelle : quatre blocs trimestriels avant l'année de test.

Exemple complet : python scripts/run_rolling_cv.py
Exemple rapide : python scripts/run_rolling_cv.py --models baselines

Les résultats historiques 70/15/15 ne sont pas modifiés. Ce script écrit des
fichiers distincts dans results/ et n'évalue jamais la période de test.
"""
import argparse
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_processing import build_dataset
from src.evaluate import score
from src.models import BASELINES, baseline_predict, sarima_predict
from src.train import fit_net, predict_net
from src.utils import RESULTS_DIR, set_seed


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["baselines", "sarima", "tcn", "lstm"],
                        default=["baselines", "sarima", "tcn"],
                        help="Familles évaluées ; LSTM est optionnel car le calcul est long.")
    parser.add_argument("--lookback", type=int, choices=[24, 48, 168], default=24,
                        help="Historique du TCN/LSTM en heures (défaut : 24).")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Nombre fixe d'époques par pli ; aucun early stopping sur le pli évalué.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                        help="Graines indépendantes pour chaque réseau (défaut : 0 1).")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                        help="Dossier des nouveaux tableaux et de la figure.")
    args = parser.parse_args()
    if args.epochs < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("--epochs doit être positif et les graines doivent être non négatives.")
    return args


def main():
    args = arguments()
    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(split_mode="annual")
    folds = ds.folds
    print(f"Validation croisée : {len(folds)} blocs, test réservé à partir de "
          f"{ds.time_index[ds.valid_end]}", flush=True)
    fold_table = pd.DataFrame([{
        "bloc": fold["name"],
        "train_jusqu_au": ds.time_index[fold["fit_end"] - 1],
        "validation_du": ds.time_index[fold["start"]],
        "validation_au": ds.time_index[fold["stop"] - 1],
        "origines_train": len(fold["train_origins"]),
        "origines_validation": len(fold["valid_origins"]),
    } for fold in folds])
    print(fold_table.to_string(index=False), flush=True)

    rows = []
    truth_parts = []
    prediction_parts = defaultdict(list)
    selected = set(args.models)
    partial_path = args.output_dir / "rolling_cv_partial.csv"
    for fold in folds:
        origins = fold["valid_origins"]
        actual = ds.truth(origins)
        truth_parts.append(actual)
        fit_end = fold["fit_end"]
        fallback = float(np.nanmedian(ds.y[:fit_end]))
        specs = []
        if "baselines" in selected:
            for name in BASELINES:
                specs.append((name, lambda name=name: baseline_predict(
                    name, origins, ds.y, fallback)))
        if "sarima" in selected:
            specs.append(("SARIMA", lambda: sarima_predict(
                ds.y, ds.time_index, fit_end, origins)))
        for kind in ("TCN", "LSTM"):
            if kind.lower() not in selected:
                continue
            name = f"{kind} {args.lookback}h calendrier"

            def neural_prediction(kind=kind):
                xs, mean, std = ds.preprocess(fit_end)
                predictions = []
                for seed in args.seeds:
                    model = fit_net(kind, args.lookback, ds, fold["train_origins"],
                                    xs, mean, std, use_calendar=True,
                                    val_origins=None, epochs=args.epochs,
                                    seed=seed, verbose=False)
                    predictions.append(predict_net(
                        model, args.lookback, ds, origins, xs, mean, std))
                return np.mean(predictions, axis=0)

            specs.append((name, neural_prediction))

        for name, predict in specs:
            started = time.time()
            pred = predict()
            prediction_parts[name].append(pred)
            result = {
                "bloc": fold["name"], "modele": name,
                "train_jusqu_au": ds.time_index[fit_end - 1],
                "validation_du": ds.time_index[fold["start"]],
                "validation_au": ds.time_index[fold["stop"] - 1],
                "origines_train": len(fold["train_origins"]),
                "origines_validation": len(origins),
                "temps_s": round(time.time() - started, 1),
                **score(actual, pred),
            }
            rows.append(result)
            print(f"{fold['name']} | {name} | MAE {result['MAE_kW']:.4f} kW "
                  f"| {result['temps_s']} s", flush=True)
            # Écriture incrémentale : les plis terminés restent disponibles après interruption.
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "rolling_cv.csv", index=False)
    partial_path.unlink(missing_ok=True)
    pooled_truth = np.vstack(truth_parts)
    summary_rows = []
    for name, predictions in prediction_parts.items():
        fold_mae = results.loc[results["modele"] == name, "MAE_kW"]
        pooled = score(pooled_truth, np.vstack(predictions))
        summary_rows.append({
            "modele": name,
            "MAE_moyenne_plis_kW": float(fold_mae.mean()),
            "MAE_ecart_type_plis_kW": float(fold_mae.std(ddof=0)),
            "MAE_globale_kW": pooled["MAE_kW"],
            "RMSE_globale_kW": pooled["RMSE_kW"],
            "n_previsions": pooled["n"],
        })
    summary = pd.DataFrame(summary_rows).set_index("modele").sort_values("MAE_globale_kW")
    summary.to_csv(args.output_dir / "rolling_cv_summary.csv")
    print("\nRésumé des quatre blocs :")
    print(summary.round(4).to_string())

    ax = results.pivot(index="bloc", columns="modele", values="MAE_kW").plot(
        marker="o", figsize=(10, 5), title="Validation glissante — MAE par bloc")
    ax.set(xlabel="Bloc chronologique", ylabel="MAE (kW)")
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "12_rolling_cv_mae.png", dpi=120)
    plt.close(ax.figure)
    print("\nSorties : rolling_cv.csv, rolling_cv_summary.csv et figures/12_rolling_cv_mae.png")


if __name__ == "__main__":
    main()
