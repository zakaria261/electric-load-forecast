# Ce qui a été retenu des deux travaux

Il n'y a qu'un notebook dans le dépôt, `notebooks/01_TP_prevision_electrique_guide.ipynb`. Le travail d'Amine est principalement dans `src/`, les scripts et les premiers fichiers `results/`. Les deux réalisations visent la même prévision de 24 puissances horaires d'un foyer UCI, avec une origine d'entraînement par heure et une origine évaluée toutes les six heures.

| Sujet | Notebook initial | Pipeline d'Amine initial | Version réunie |
|---|---|---|---|
| Découpage | Environ 2 ans train, 4 blocs trimestriels de validation, dernière année test | 70/15/15, test surtout avril–novembre 2010 | Protocole annuel et glissant par défaut dans `src/` ; anciens scripts explicitement `legacy` |
| Saisons | Couverture de chaque saison et erreur selon l'heure **cible** | Test sans hiver complet ; erreur selon la saison de l'**origine** | Test annuel avec hiver ; erreur par saison de chaque cible, identique dans notebook et `src/` |
| Baselines | Persistance, J-1, J-7, profil moyen sur 7 jours | Persistance, moyenne mobile 24 h, J-1, J-7 | Les cinq règles sur les mêmes 24 cibles et les mêmes quatre blocs |
| Modèles avancés | LSTM, TCN et SARIMA optionnels, une graine | SARIMA, LSTM, TCN, plusieurs graines et ablation du calendrier | Réutilisation des architectures de `src/`, plusieurs graines possibles, SARIMA et réseaux évalués sur les mêmes blocs ; époques fixées avant le bloc évalué |
| Champ réceptif du TCN | Trois blocs pour toute fenêtre | Trois blocs pour toute fenêtre | Nombre de blocs adapté à 24, 48 ou 168 h ; une entrée 168 h est réellement accessible au dernier état |
| Restitution | Explications pédagogiques, MAE par bloc/heure et prévision 24 h | Figures EDA, pics, horizons, intervalles empiriques, fichiers de résultats | Notebook pédagogique + CSV/NPZ et figures reproductibles pour la présentation |

## Ce que montrent les données

- 2 075 259 mesures à la minute, sept variables électriques, 34 589 heures sur la grille ; 34 151 heures ont une cible valide. Le taux de cible manquante est de 1,27 %.
- Heures cibles manquantes : 69 dans l'entraînement initial, 74 en validation et 295 dans l'année de test. Exiger 24 cibles complètes retire aussi les origines proches de ces trous ; les scores décrivent donc les fenêtres évaluables.
- La moyenne de consommation de l'entraînement est de 1,10 kW ; l'hiver est plus élevé (1,49 kW) que l'été (0,72 kW). La validation et le test contiennent chacun les quatre saisons.
- Quatre blocs glissants couvrent la validation du 26 novembre 2008 au 25 novembre 2009 ; le test commence le 26 novembre 2009. Les 1 412 origines valides de validation donnent 33 888 couples origine–horizon évalués.

## Premiers résultats comparables

| Modèle | MAE globale validation (kW) | Hiver (kW) | Été (kW) |
|---|---:|---:|---:|
| TCN 24 h + calendrier, 2 graines | 0,479 | 0,637 | 0,356 |
| Profil moyen 7 jours | 0,480 | 0,617 | 0,356 |
| TCN 48 h + calendrier, 2 graines | 0,481 | — | — |
| TCN 168 h + calendrier, 2 graines | 0,484 | — | — |
| SARIMA | 0,487 | 0,595 | 0,363 |
| LSTM 24 h + calendrier, 2 graines | 0,487 | — | — |
| LSTM 168 h + calendrier, 2 graines | 0,491 | — | — |
| LSTM 48 h + calendrier, 2 graines | 0,491 | — | — |
| TCN 24 h sans calendrier, 2 graines | 0,520 | — | — |
| J-7 | 0,571 | — | — |
| J-1 | 0,609 | — | — |
| Moyenne mobile 24 h | 0,656 | — | — |
| Persistance | 0,767 | — | — |

Le TCN 24 h devance la règle des sept jours de seulement 0,002 kW en MAE globale. La règle simple est meilleure en hiver et en été ; SARIMA fait mieux pendant l'hiver. Les versions 48 h et 168 h des deux réseaux font moins bien que les versions 24 h sur cette validation, malgré un champ réceptif couvrant vraiment la semaine pour le TCN 168 h. Le classement global ne suffit donc pas à démontrer une supériorité nette du réseau. Les cinq prévisions du notebook et de `src/` coïncident exactement sur les mêmes origines (écart maximal mesuré : 0 kW).

En rééchantillonnant les **53 semaines d'émission** de validation, l'écart de MAE apparié TCN 24 h moins profil 7 jours vaut −0,0018 kW, avec un intervalle descriptif à 95 % d'environ [−0,013 ; +0,011] kW. Il inclut zéro. Ce calcul regroupe les prévisions chevauchantes d'une semaine, mais ne corrige pas la sélection préalable du gagnant sur cette même validation.

L'ablation du calendrier garde la même architecture TCN 24 h, les mêmes deux graines et les mêmes origines : sa MAE passe de 0,479 à 0,520 kW sans les variables calendaires, soit un gain d'environ 7,9 % en faveur du calendrier. Cette ablation regroupe heure, jour de semaine, saison annuelle, week-end et jours fériés ; elle ne sépare pas leur contribution individuelle.
Le gain se concentre au printemps (0,501 → 0,464 kW), en été (0,436 → 0,356 kW) et en automne (0,509 → 0,459 kW) ; en hiver, les deux versions sont pratiquement à égalité (0,634 sans, 0,637 avec).

Pour le TCN, la profondeur augmente avec la fenêtre afin que toute l'entrée soit accessible au dernier état. La comparaison 24/48/168 h compare donc des **configurations complètes** (historique et capacité du réseau), pas le seul effet causal de la longueur d'historique. Le LSTM garde la même architecture entre fenêtres.

Les réseaux sont évalués avec deux graines et quatre époques fixes par bloc. Ce choix évite d'utiliser le bloc évalué pour l'arrêt anticipé, mais n'optimise pas séparément la durée d'entraînement de chaque architecture. Une recherche d'hyperparamètres imbriquée dans le passé serait nécessaire pour conclure plus fortement sur le classement.

## Bilan de la dernière année

Le protocole complet retient le TCN 24 h avec calendrier d'après la validation, puis l'entraîne sur entraînement + validation. Sur 33 456 couples origine–horizon de l'année de test, il obtient **0,463 kW de MAE** et **0,627 kW de RMSE** ; le profil 7 jours obtient **0,466 kW** et **0,655 kW**. La différence de MAE globale reste faible.

| Segment du test | TCN 24 h (MAE kW) | Profil 7 jours (MAE kW) |
|---|---:|---:|
| Hiver | 0,551 | 0,546 |
| Printemps | 0,459 | 0,460 |
| Été | 0,367 | 0,373 |
| Automne | 0,474 | 0,482 |
| Pics ≥ 3,050 kW | **1,606** | 1,844 |

Le réseau réduit l'erreur sur les pics d'environ 13 % par rapport au profil 7 jours, même si la baseline simple reste légèrement meilleure en hiver. Son intervalle empirique à 90 % couvre **90,95 %** des cibles du test et **87,06 %** de celles de l'hiver : la couverture hivernale est insuffisante pour promettre 90 % toute l'année. Sur le test, l'horizon 1 h a une MAE de 0,368 kW et l'horizon 20 h de 0,530 kW ; l'heure d'émission doit être examinée séparément de l'horizon.

## Comment présenter l'évaluation

1. Montrer la série et les distributions saisonnières ; expliquer que **kW** est une puissance moyenne horaire et que chaque émission fournit 24 horizons.
2. Montrer les quatre blocs et la séparation stricte des cibles. L'entraînement s'allonge avant chaque bloc ; le test annuel vient ensuite.
3. Donner la MAE en kW avec un exemple : 0,480 kW signifie un écart absolu moyen de 0,480 kW par couple origine–horizon, pas une erreur de 0,480 kWh sur la journée. Comparer par saison, horizon, heure d'émission et pics.
4. Expliquer que des origines toutes les six heures couvrent 0 h, 6 h, 12 h et 18 h, au lieu de mesurer toujours une prévision émise à 17 h.
5. Présenter les intervalles comme **empiriques**, calibrés avec les résidus de validation. Les fenêtres se chevauchent ; la couverture observée est descriptive.

Les anciens scores de `results/test_scores.csv` et `results/baselines_test.csv` viennent du 70/15/15 : il ne faut pas les mettre dans le même classement. Le test historique a déjà été consulté, donc la nouvelle évaluation finale ne peut plus être qualifiée de test indépendant vierge. Les données météorologiques et les vacances scolaires ne sont pas disponibles dans ce projet ; leur effet reste une piste future.
