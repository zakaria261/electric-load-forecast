"""Validation croisée temporelle : quatre blocs trimestriels avant l'année de test.

Exemple complet : python scripts/run_rolling_cv.py
Exemple rapide : python scripts/run_rolling_cv.py --models baselines

Les résultats historiques 70/15/15 ne sont pas modifiés. Ce script écrit des
fichiers distincts dans results/ et n'évalue jamais la période de test.
"""
import argparse
import json
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
from src.evaluate import peak_scores, score, season_scores
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
    parser.add_argument("--reuse-classical", action="store_true",
                        help="Réutilise les prédictions déjà sauvées des baselines/SARIMA "
                             "si les origines et les cibles sont identiques.")
    parser.add_argument("--ablation", action="store_true",
                        help="Évalue aussi les réseaux sans variables calendaires.")
    parser.add_argument("--without-calendar", action="store_true",
                        help="Évalue uniquement les réseaux sans calendrier (dossier distinct conseillé).")
    args = parser.parse_args()
    if args.epochs < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("--epochs doit être positif et les graines doivent être non négatives.")
    if args.ablation and args.without_calendar:
        parser.error("Choisir --ablation ou --without-calendar, pas les deux.")
    return args


def main():
    args = arguments()
    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(split_mode="annual")
    folds = ds.folds
    cached = {}
    cache_path = args.output_dir / "rolling_cv_predictions.npz"
    if args.reuse_classical and cache_path.is_file():
        with np.load(cache_path) as saved:
            expected_origins = np.concatenate([fold["valid_origins"] for fold in folds])
            if (np.array_equal(saved["origins"], expected_origins)
                    and np.array_equal(saved["truth"], ds.truth(expected_origins))):
                cached = {str(name): pred for name, pred in
                          zip(saved["model_names"], saved["predictions"])}
                print("Prédictions classiques réutilisables :",
                      sorted(set(cached) & (set(BASELINES) | {"SARIMA"})), flush=True)
            else:
                raise ValueError("Cache incompatible avec les origines ou les cibles actuelles.")
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
    origin_parts = []
    prediction_parts = defaultdict(list)
    selected = set(args.models)
    partial_path = args.output_dir / "rolling_cv_partial.csv"
    offset = 0
    for fold in folds:
        origins = fold["valid_origins"]
        actual = ds.truth(origins)
        truth_parts.append(actual)
        origin_parts.append(origins)
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
            calendar_options = ([True, False] if args.ablation else
                                [False] if args.without_calendar else [True])
            for use_calendar in calendar_options:
                suffix = "calendrier" if use_calendar else "sans calendrier"
                name = f"{kind} {args.lookback}h {suffix}"

                def neural_prediction(kind=kind, use_calendar=use_calendar):
                    xs, mean, std = ds.preprocess(fit_end)
                    predictions = []
                    for seed in args.seeds:
                        model = fit_net(kind, args.lookback, ds, fold["train_origins"],
                                        xs, mean, std, use_calendar=use_calendar,
                                        val_origins=None, epochs=args.epochs,
                                        seed=seed, verbose=False)
                        predictions.append(predict_net(
                            model, args.lookback, ds, origins, xs, mean, std,
                            use_calendar=use_calendar))
                    return np.mean(predictions, axis=0)

                specs.append((name, neural_prediction))

        for name, predict in specs:
            started = time.time()
            reused = name in cached and name in (set(BASELINES) | {"SARIMA"})
            if reused:
                pred = cached[name][offset:offset + len(origins)]
            else:
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
                "cache": reused,
                **score(actual, pred),
            }
            rows.append(result)
            print(f"{fold['name']} | {name} | MAE {result['MAE_kW']:.4f} kW "
                  f"| {result['temps_s']} s", flush=True)
            # Écriture incrémentale : les plis terminés restent disponibles après interruption.
            pd.DataFrame(rows).to_csv(partial_path, index=False)
        offset += len(origins)

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "rolling_cv.csv", index=False)
    partial_path.unlink(missing_ok=True)
    pooled_truth = np.vstack(truth_parts)
    pooled_origins = np.concatenate(origin_parts)
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

    # Même ordre d'origines et de cibles pour chaque modèle : comparaison directe.
    model_names = list(prediction_parts)
    pooled_predictions = np.stack([np.vstack(prediction_parts[name]) for name in model_names])
    np.savez_compressed(args.output_dir / "rolling_cv_predictions.npz",
                        origins=pooled_origins, truth=pooled_truth,
                        model_names=np.asarray(model_names), predictions=pooled_predictions)
    (args.output_dir / "rolling_cv_config.json").write_text(json.dumps({
        "split_mode": "annual", "models": args.models, "lookback": args.lookback,
        "epochs": args.epochs, "seeds": args.seeds, "ablation": args.ablation,
        "without_calendar": args.without_calendar,
        "validation_start": str(ds.time_index[ds.train_end]),
        "test_start": str(ds.time_index[ds.valid_end]),
    }, ensure_ascii=False, indent=2) + "\n")
    details = []
    origin_times = ds.time_index[pooled_origins]
    peak_threshold = float(np.nanquantile(ds.y[:ds.train_end], .95))
    for name, pred in zip(model_names, pooled_predictions):
        for season, metrics in season_scores(pooled_truth, pred, origin_times).iterrows():
            details.append({"modele": name, "dimension": "saison cible",
                            "groupe": season, **metrics.to_dict()})
        for hour in sorted(origin_times.hour.unique()):
            mask = origin_times.hour == hour
            details.append({"modele": name, "dimension": "heure émission",
                            "groupe": str(hour), **score(pooled_truth[mask], pred[mask])})
        for horizon in range(1, pooled_truth.shape[1] + 1):
            details.append({"modele": name, "dimension": "horizon",
                            "groupe": str(horizon),
                            **score(pooled_truth[:, horizon - 1], pred[:, horizon - 1])})
        details.append({"modele": name, "dimension": "pics",
                        "groupe": f">= {peak_threshold:.3f} kW",
                        **peak_scores(pooled_truth, pred, peak_threshold).loc["Pics"].to_dict()})
    diagnostics = pd.DataFrame(details)
    diagnostics["n"] = diagnostics["n"].astype(int)
    diagnostics.to_csv(args.output_dir / "rolling_cv_diagnostics.csv", index=False)

    ax = results.pivot(index="bloc", columns="modele", values="MAE_kW").plot(
        marker="o", figsize=(11, 5), title="Validation glissante — MAE par bloc")
    ax.set(xlabel="Bloc chronologique", ylabel="MAE (kW)")
    ax.legend(title="Modèle", loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "12_rolling_cv_mae.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    print("\nSorties : rolling_cv.csv, rolling_cv_summary.csv, "
          "rolling_cv_diagnostics.csv, rolling_cv_predictions.npz, "
          "rolling_cv_config.json et "
          "figures/12_rolling_cv_mae.png")


if __name__ == "__main__":
    main()
