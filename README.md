# Prévision de consommation électrique à 24 heures

Ce projet prévoit, à partir des mesures connues à une heure d'émission, les **24 puissances horaires suivantes** d'un foyer du jeu UCI. Une sortie est donc un vecteur de 24 valeurs en kW, pas une seule valeur pour une heure fixe. Le [notebook guide](notebooks/01_TP_prevision_electrique_guide.ipynb) explique les données et chaque étape ; le [notebook de démonstration](notebooks/demo.ipynb) montre rapidement les résultats sauvegardés pour la présentation. `src/` porte les fonctions réutilisables.

La [présentation HTML des baselines](presentation_baselines.html) s'ouvre directement dans un navigateur, sans Jupyter. Cliquer sur **Mode présentateur** ouvre une seconde fenêtre avec les notes orales et les commandes ; la fenêtre initiale reste destinée au public. Utiliser les écrans en **affichage étendu**, déplacer les notes sur l'écran privé, puis mettre la fenêtre publique sur le projecteur. Les flèches changent de diapositive dans les deux fenêtres ; **F** active le plein écran dans la fenêtre publique, **N** ouvre les notes et **P** ouvre l'impression pour exporter en PDF.

## Données et protocole commun

Le [fichier UCI Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption) contient 2 075 259 lignes à la minute, de décembre 2006 à novembre 2010. Les sept variables électriques sont agrégées à l'heure : moyenne des grandeurs instantanées, somme des sous-compteurs. Une heure avec moins de 45 mesures valides reste manquante ; une prévision n'est évaluée que si ses 24 cibles sont observées. Le fichier téléchargé reste dans `data/`, ignoré par Git.

Le découpage principal est chronologique : entraînement initial jusqu'au 25 novembre 2008, validation du 26 novembre 2008 au 25 novembre 2009 en **quatre blocs de trois mois**, puis test du 26 novembre 2009 à novembre 2010. Le modèle est réajusté avec tout le passé disponible avant chaque bloc. Il y a une origine d'entraînement par heure et des origines évaluées à 0 h, 6 h, 12 h et 18 h ; chacune produit 24 cibles. Les fenêtres évaluées se chevauchent : la MAE représente l'erreur moyenne par couple origine–horizon, pas par heure civile unique.

L'imputation, la standardisation et le seuil de pic sont ajustés avec le passé d'entraînement seulement. Les mesures météo et les vacances scolaires ne figurent pas dans ce jeu ; les variables calendaires futures contiennent les cycles horaires, hebdomadaires et annuels, le week-end et, si `holidays` est installé, les jours fériés français.

La **MAE en kW** est la métrique de choix ; la RMSE signale davantage les fortes erreurs, notamment aux pics. La MAPE est aussi calculée sur les cibles non nulles, mais elle devient très sensible lorsque la consommation est proche de zéro.

## Installation

Les versions de `requirements.txt` correspondent à l'environnement ayant produit les tableaux (Python 3.14 sur macOS arm64). Une autre plateforme peut nécessiter des paquets compatibles.

```bash
./setup_env.sh
```

Ou manuellement :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_data.py
```

Ouvrir ensuite `notebooks/01_TP_prevision_electrique_guide.ipynb` dans Jupyter. Le notebook peut aussi fonctionner seul dans Colab : il télécharge le fichier UCI si nécessaire. Les réseaux et SARIMA y sont désactivés par défaut pour que les étapes données et baselines restent rapides ; les interrupteurs `RUN_NEURAL` et `RUN_SARIMA` les activent.

## Comparaison sur validation glissante

```bash
.venv/bin/python scripts/run_rolling_cv.py --models baselines
.venv/bin/python scripts/run_rolling_cv.py --models baselines sarima tcn lstm --lookback 24 --epochs 4 --seeds 0 1 --reuse-classical
```

Le second appel peut réutiliser les prédictions classiques du premier si les origines et cibles sont identiques. Pour garder les comparaisons des fenêtres séparées :

```bash
.venv/bin/python scripts/run_rolling_cv.py --models tcn lstm --lookback 48 --epochs 4 --seeds 0 1 --output-dir results/cv_48
.venv/bin/python scripts/run_rolling_cv.py --models tcn lstm --lookback 168 --epochs 4 --seeds 0 1 --output-dir results/cv_168
.venv/bin/python scripts/run_rolling_cv.py --models tcn --lookback 24 --epochs 4 --seeds 0 1 --without-calendar --output-dir results/cv_24_nocal
```

Les réseaux utilisent un nombre fixe d'époques, choisi avant les blocs évalués ; aucun arrêt anticipé ne consulte le bloc en cours. Le TCN a un champ réceptif qui couvre réellement la fenêtre de 24, 48 ou 168 h choisie. Sa profondeur change avec la fenêtre : cette comparaison porte donc sur la configuration complète.

Les fichiers `rolling_cv.csv`, `rolling_cv_summary.csv`, `rolling_cv_diagnostics.csv` et `rolling_cv_predictions.npz` contiennent les scores par bloc, globaux, par saison de **l'heure cible**, par heure d'émission et par horizon. `rolling_cv_config.json` conserve les paramètres du benchmark. La figure `results/figures/12_rolling_cv_mae.png` montre la variation entre blocs.

Les premiers résultats mesurés sur **33 888 couples origine–horizon** sont :

| Modèle | MAE globale validation (kW) |
|---|---:|
| TCN 24 h + calendrier, 2 graines | 0,479 |
| Profil moyen des 7 jours précédents | 0,480 |
| TCN 48 h + calendrier, 2 graines | 0,481 |
| TCN 168 h + calendrier, 2 graines | 0,484 |
| SARIMA journalier | 0,487 |
| LSTM 24 h + calendrier, 2 graines | 0,487 |
| LSTM 168 h + calendrier, 2 graines | 0,491 |
| LSTM 48 h + calendrier, 2 graines | 0,491 |
| TCN 24 h sans calendrier, 2 graines | 0,520 |
| Saisonnier hebdomadaire (J-7) | 0,571 |
| Saisonnier quotidien (J-1) | 0,609 |
| Moyenne mobile des 24 h passées | 0,656 |
| Persistance | 0,767 |

Le TCN 24 h dépasse la règle simple de seulement 0,002 kW en MAE globale : cet écart minime ne suffit pas à prétendre à un gain robuste. Le calendrier réduit la MAE du TCN de 0,520 à 0,479 kW, soit environ 7,9 % par rapport à la version sans calendrier. Sur le profil 7 jours, la MAE vaut 0,617 kW dans le bloc d'hiver et 0,356 kW dans le bloc d'été. La saison est celle de chaque heure prévue. Le [détail des différences et des décisions](COMPARAISON.md) relie le notebook au travail d'Amine.

Après avoir fixé les candidats et produit la validation glissante, le bilan de la dernière année peut être lancé avec :

```bash
.venv/bin/python scripts/run_annual_final.py --cv-dir results results/cv_48 results/cv_168 results/cv_24_nocal
```

Ce script sélectionne le gagnant selon la MAE de validation, le réentraîne sur entraînement + validation si nécessaire, et enregistre les métriques de test ainsi que des intervalles empiriques calibrés sur les résidus de validation. Le test historique avait déjà été consulté lors du travail initial : il faut présenter ce bilan comme **descriptif**, pas comme une nouvelle mesure entièrement indépendante.

Sur l'année de test (33 456 couples origine–horizon), le TCN 24 h avec calendrier obtient une **MAE de 0,463 kW**, contre **0,466 kW** pour le profil moyen 7 jours. Pour les cibles de pic (seuil 3,050 kW fixé sur l'entraînement), leurs MAE sont respectivement **1,606 et 1,844 kW**. Les intervalles empiriques à 90 % couvrent 90,95 % de toutes les cibles, mais seulement 87,06 % en hiver. Le fichier `annual_cv_pairwise.csv` indique que le petit écart global de validation TCN–baseline est compatible avec zéro dans un rééchantillonnage descriptif par semaines.

## Résultats historiques et fichiers

Les scripts `run_baselines.py`, `run_neural.py`, `run_sarima.py`, `run_seeds_ablation.py` et `run_test.py` gardent explicitement le découpage historique **70/15/15**. Les anciens fichiers `results/baselines_test.csv`, `results/test_scores.csv` et les figures numérotées 08 à 11 viennent de ce protocole. Leur test couvre surtout avril à novembre 2010, sans hiver complet ; leurs MAE ne se comparent pas directement aux scores du tableau ci-dessus. Les anciens scores TCN 168 h utilisent l'architecture à trois blocs ; relancer les scripts avec le TCN corrigé peut modifier ces scores. `run_eda.py` utilise désormais l'entraînement du protocole annuel pour les analyses exploratoires.

`results/` contient des fichiers de résultats de taille modeste ; les données brutes et les modèles sauvegardés sont exclus du dépôt. Pour une démonstration, montrer la table descriptive, les quatre blocs, la comparaison MAE, les erreurs d'hiver/pics et quelques prévisions à 24 h avec leur intervalle.
