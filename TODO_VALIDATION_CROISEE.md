# État de la validation croisée temporelle

Le protocole du [notebook guide](notebooks/01_TP_prevision_electrique_guide.ipynb) et celui de `src/` sont maintenant alignés : entraînement initial jusqu'au 25 novembre 2008, quatre blocs trimestriels de validation jusqu'au 25 novembre 2009, puis une année de test. Les origines d'entraînement avancent d'une heure ; celles d'évaluation de six heures. Les 24 cibles d'une origine restent dans un seul bloc.

## Réalisé

- [x] Cinq baselines identiques dans le notebook et `src/` ; écart maximal de leurs prévisions vérifié à 0 kW.
- [x] SARIMA, LSTM et TCN évalués sur les mêmes 33 888 couples origine–horizon, avec deux graines et quatre époques fixes pour les réseaux.
- [x] Fenêtres 24 h, 48 h et 168 h comparées ; champ réceptif du TCN corrigé pour chaque fenêtre.
- [x] Ablation du calendrier sur le TCN 24 h : MAE 0,520 sans et 0,479 kW avec calendrier.
- [x] Scores par bloc, saison de l'heure cible, heure d'émission, horizon et pic ; figures et prédictions sauvegardées dans `results/`.
- [x] Sélection par MAE de validation, réentraînement sur entraînement + validation et bilan annuel dans `scripts/run_annual_final.py`.
- [x] Notebook léger `notebooks/demo.ipynb`, instructions d'installation et document [COMPARAISON.md](COMPARAISON.md) pour la présentation.

## Résultats à retenir

Le meilleur TCN utilise 24 h d'historique et le calendrier : 0,479 kW de MAE en validation glissante. Le profil moyen 7 jours suit à 0,480 kW ; l'écart est minuscule et son intervalle descriptif par semaines inclut zéro. Le TCN 48 h atteint 0,481 kW et le TCN 168 h 0,484 kW. Sur l'année de test, le TCN 24 h fait 0,463 kW, contre 0,466 kW pour le profil 7 jours. Sur les pics, leurs MAE sont 1,606 et 1,844 kW. L'intervalle empirique à 90 % couvre 90,95 % des cibles du test, mais 87,06 % de celles de l'hiver.

## Limites et suites possibles

- [ ] Appairer une météo horaire et les vacances scolaires, absentes du jeu de données utilisé, avant d'attribuer un effet à la température ou aux vacances.
- [ ] Si une nouvelle évaluation indépendante est nécessaire, obtenir d'autres années ou d'autres foyers : le test historique a déjà été consulté avant cette révision.
- [ ] Pour isoler strictement l'effet de la longueur d'historique du TCN, fixer sa profondeur et refaire une comparaison avec entrées masquées à 24/48/168 h. Ici la profondeur change afin de couvrir chaque fenêtre.
- [ ] Pour optimiser les époques et hyperparamètres, utiliser une validation interne au passé de chaque bloc ; ne pas ajuster l'arrêt anticipé sur le bloc évalué.
- [ ] Étudier des intervalles calibrés séparément par saison ou régime de charge : la couverture hivernale actuelle est inférieure à 90 %.

Les anciens résultats `results/baselines_test.csv`, `results/test_scores.csv` et `results/seeds_ablation_validation.csv` restent des archives du découpage 70/15/15. Leurs scores ne doivent pas entrer dans le tableau du protocole annuel ; les anciens scores TCN 168 h utilisaient aussi le champ réceptif insuffisant de l'architecture à trois blocs.
