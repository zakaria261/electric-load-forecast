# Prévision de consommation électrique à 24 h

Le projet prévoit les 24 prochaines puissances horaires (kW) du foyer UCI. Le
[notebook guide](notebooks/01_TP_prevision_electrique_guide.ipynb) explique les données,
les baselines, les modèles et l'évaluation.

## Validation croisée temporelle

Le script `scripts/run_rolling_cv.py` reprend le calendrier du notebook : environ
deux ans d'entraînement initial, quatre blocs successifs de trois mois pour la
validation, puis la dernière année réservée au test. Avant chaque bloc, le modèle
est réentraîné sur les données antérieures. Il utilise une origine d'entraînement
par heure et évalue une prévision de 24 h à 0 h, 6 h, 12 h et 18 h. Les fenêtres
évaluées se chevauchent ; la MAE porte sur les couples origine–horizon.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_rolling_cv.py --models baselines
```

Pour comparer aussi SARIMA et TCN (calcul plus long) :

```bash
python scripts/run_rolling_cv.py --models baselines sarima tcn
```

Le script accepte `--models lstm`, `--lookback 24`, `--epochs 4` et
`--seeds 0 1`. Les époques sont fixées avant les blocs évalués : aucun arrêt
anticipé n'est réglé sur les cibles du bloc. Les sorties sont
`results/rolling_cv.csv`, `results/rolling_cv_summary.csv` et
`results/figures/12_rolling_cv_mae.png`. Elles sont distinctes des résultats
historiques de `results/`, calculés avec le découpage 70/15/15 ; ces anciens
scores ne doivent pas être comparés directement aux scores de validation
glissante. Le script ne consulte pas le test final. Le test historique a déjà
été consulté lors du travail initial : cette validation glissante vérifie la
stabilité saisonnière, sans rétablir un test final entièrement vierge.
