"""Évalue le gagnant de la validation glissante sur la dernière année.

Les prédictions de validation et les paramètres des réseaux proviennent de
scripts/run_rolling_cv.py. Le test historique ayant déjà été consulté dans ce
projet, ce résultat est un bilan descriptif, pas un test entièrement vierge.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_processing import build_dataset
from src.evaluate import (coverage, empirical_interval, paired_weekly_mae_interval,
                          peak_scores, score, season_scores)
from src.models import BASELINES, baseline_predict, sarima_predict
from src.train import fit_net, predict_net
from src.utils import HORIZON, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, nargs="+", default=[RESULTS_DIR],
                        help="Un ou plusieurs dossiers de validation sur les mêmes blocs.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(split_mode="annual")
    expected = np.concatenate([fold["valid_origins"] for fold in ds.folds])
    summaries = []
    validation_predictions = {}
    for directory in args.cv_dir:
        with np.load(directory / "rolling_cv_predictions.npz") as saved:
            if (not np.array_equal(saved["origins"], expected)
                    or not np.array_equal(saved["truth"], ds.truth(expected))):
                raise ValueError(f"Validation incompatible avec les données : {directory}")
            for name, pred in zip(saved["model_names"].astype(str), saved["predictions"]):
                if name in validation_predictions:
                    raise ValueError(f"Modèle présent dans plusieurs dossiers : {name}")
                validation_predictions[name] = pred
        part = pd.read_csv(directory / "rolling_cv_summary.csv")
        part["cv_dir"] = str(directory)
        summaries.append(part)
    summary = pd.concat(summaries, ignore_index=True).sort_values("MAE_globale_kW")
    if summary["modele"].duplicated().any():
        raise ValueError("Les dossiers de validation contiennent des noms de modèles en double.")
    validation_truth = ds.truth(expected)
    for row in summary.itertuples(index=False):
        if row.modele not in validation_predictions:
            raise ValueError(f"Prédictions absentes pour {row.modele}.")
        measured = score(validation_truth, validation_predictions[row.modele])["MAE_kW"]
        if not np.isclose(measured, row.MAE_globale_kW, atol=1e-8):
            raise ValueError(f"Tableau et prévisions de validation incohérents pour {row.modele}.")
    summary.to_csv(args.output_dir / "annual_model_comparison.csv", index=False)
    selected = str(summary.iloc[0]["modele"])
    selected_dir = Path(summary.iloc[0]["cv_dir"])
    if selected not in validation_predictions:
        raise ValueError("Le modèle retenu manque dans les prédictions de validation.")
    val_truth = validation_truth
    val_pred = validation_predictions[selected]
    if len(summary) >= 2:
        second = str(summary.iloc[1]["modele"])
        pairwise = paired_weekly_mae_interval(
            val_truth, val_pred, validation_predictions[second], ds.time_index[expected])
        pd.DataFrame([{"premier": selected, "second": second,
                       **pairwise}]).to_csv(args.output_dir / "annual_cv_pairwise.csv", index=False)

    test_origins = ds.test_origins
    test_truth = ds.truth(test_origins)
    fallback = float(np.nanmedian(ds.y[:ds.valid_end]))
    config_path = selected_dir / "rolling_cv_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if selected in BASELINES:
        test_pred = baseline_predict(selected, test_origins, ds.y, fallback)
    elif selected == "SARIMA":
        test_pred = sarima_predict(ds.y, ds.time_index, ds.valid_end, test_origins)
    elif selected.startswith(("TCN ", "LSTM ")):
        kind = selected.split()[0]
        use_calendar = not selected.endswith("sans calendrier")
        suffix = "calendrier" if use_calendar else "sans calendrier"
        if not config or selected != f"{kind} {config['lookback']}h {suffix}":
            raise ValueError("Configuration du réseau absente ou incompatible avec la validation.")
        xs, mean, std = ds.preprocess(ds.valid_end)
        runs = []
        for seed in config["seeds"]:
            model = fit_net(kind, config["lookback"], ds, ds.trainval_origins,
                            xs, mean, std, use_calendar=use_calendar, val_origins=None,
                            epochs=config["epochs"], seed=seed, verbose=False)
            runs.append(predict_net(model, config["lookback"], ds, test_origins,
                                    xs, mean, std, use_calendar=use_calendar))
        test_pred = np.mean(runs, axis=0)
    else:
        raise ValueError(f"Modèle non reconnu : {selected}")

    predictions = {selected: test_pred}
    reference = "Profil moyen 7 jours"
    if selected != reference:
        predictions[reference] = baseline_predict(reference, test_origins, ds.y, fallback)
    scores = pd.DataFrame({name: score(test_truth, pred)
                           for name, pred in predictions.items()}).T
    scores["n"] = scores["n"].astype(int)
    scores.index.name = "modele"
    scores.to_csv(args.output_dir / "annual_test_scores.csv")

    details = []
    target_times = ds.time_index[test_origins]
    peak_threshold = float(np.nanquantile(ds.y[:ds.train_end], .95))
    for name, pred in predictions.items():
        for season, metrics in season_scores(test_truth, pred, target_times).iterrows():
            details.append({"modele": name, "dimension": "saison cible",
                            "groupe": season, **metrics.to_dict()})
        for hour in sorted(target_times.hour.unique()):
            mask = target_times.hour == hour
            details.append({"modele": name, "dimension": "heure émission",
                            "groupe": str(hour), **score(test_truth[mask], pred[mask])})
        for horizon in range(1, HORIZON + 1):
            details.append({"modele": name, "dimension": "horizon",
                            "groupe": str(horizon),
                            **score(test_truth[:, horizon - 1], pred[:, horizon - 1])})
        details.append({"modele": name, "dimension": "pics",
                        "groupe": f">= {peak_threshold:.3f} kW",
                        **peak_scores(test_truth, pred, peak_threshold).loc["Pics"].to_dict()})
    diagnostics = pd.DataFrame(details)
    diagnostics["n"] = diagnostics["n"].astype(int)
    diagnostics.to_csv(args.output_dir / "annual_test_diagnostics.csv", index=False)

    lower, upper = empirical_interval(val_truth, val_pred, test_pred, level=.90)
    interval_rows = [{
        "modele": selected, "segment": "Toutes",
        "couverture_pct": 100 * coverage(test_truth, lower, upper),
        "largeur_moyenne_kW": float(np.mean(upper - lower)),
        "n": int(test_truth.size),
    }]
    targets = target_times.to_numpy()[:, None] + np.arange(1, HORIZON + 1)[None, :] * np.timedelta64(1, "h")
    months = pd.DatetimeIndex(targets.ravel()).month.to_numpy().reshape(test_truth.shape)
    for season, season_months in [("Hiver", [12, 1, 2]), ("Printemps", [3, 4, 5]),
                                  ("Été", [6, 7, 8]), ("Automne", [9, 10, 11])]:
        mask = np.isin(months, season_months)
        interval_rows.append({
            "modele": selected, "segment": season,
            "couverture_pct": 100 * coverage(test_truth[mask], lower[mask], upper[mask]),
            "largeur_moyenne_kW": float(np.mean((upper - lower)[mask])),
            "n": int(mask.sum()),
        })
    interval_stats = pd.DataFrame(interval_rows).set_index(["modele", "segment"])
    interval_stats.to_csv(args.output_dir / "annual_test_intervals.csv")
    np.savez_compressed(args.output_dir / "annual_test_predictions.npz",
                        origins=test_origins, truth=test_truth, prediction=test_pred,
                        lower=lower, upper=upper, selected=np.asarray(selected))

    # Quatre origines réparties dans l'année pour montrer le changement de saison.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, position in zip(axes.flat, np.linspace(0, len(test_origins) - 1, 4).astype(int)):
        origin = int(test_origins[position])
        dates = ds.time_index[origin + 1:origin + HORIZON + 1]
        ax.plot(dates, test_truth[position], "k-o", ms=2, label="Observé")
        ax.plot(dates, test_pred[position], "C0-o", ms=2, label="Prévu")
        ax.fill_between(dates, lower[position], upper[position], alpha=.2,
                        color="C0", label="Intervalle empirique 90 %")
        ax.set_title(f"{selected} — émission {ds.time_index[origin]:%d/%m/%Y %Hh}")
        ax.tick_params(axis="x", rotation=40)
        ax.set_ylabel("kW")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "13_annual_test_predictions.png", dpi=120)
    plt.close(fig)
    print(f"Choix validation : {selected}")
    if len(summary) >= 2:
        print("Écart apparié de MAE sur validation (intervalle descriptif par semaines) :")
        print(pd.read_csv(args.output_dir / "annual_cv_pairwise.csv").round(4).to_string(index=False))
    print(scores.round(4).to_string())
    print("\nIntervalles :")
    print(interval_stats.round(2).to_string())
    print("Résultats : annual_test_*.csv/npz et figures/13_annual_test_predictions.png")


if __name__ == "__main__":
    main()
