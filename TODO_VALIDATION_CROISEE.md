# Reste à faire — validation croisée glissante et livrables

État au dernier push : pipeline complet (données, EDA, baselines, LSTM/TCN, SARIMA, ablation du
calendrier, test final). Il manque la validation glissante et les livrables du sujet.

## Résultats actuels (à ne pas perdre)

| Modèle | MAE val. (kW) | MAE test (kW) | RMSE test (kW) |
|---|---|---|---|
| Baseline J-7 | 0,602 | 0,556 | 0,799 |
| SARIMA(1,0,1)(1,0,1)24 | **0,488** | 0,448 | 0,616 |
| TCN 168 h + calendrier (3 graines) | 0,503 | **0,421** | **0,578** |
| LSTM 168 h + calendrier (3 graines) | 0,509 | — | — |

- Le classement SARIMA/TCN s'inverse entre validation et test : c'est la raison d'être de la
  validation glissante.
- Le calendrier apporte −5,5 % de MAE sur le TCN (hors bruit), peu sur le LSTM 24 h.
- Pics (≥ 2,95 kW, 1,6 % des valeurs) : MAE 1,8–2,2 kW, soit 4–5 fois la moyenne. Point faible.
- Intervalles 90 % : couverture empirique ≈ 93 % sur le test.
- Limite : le test (avril–novembre 2010) ne contient pas d'hiver.

## 1. Validation croisée glissante (fenêtre croissante)

Objectif : vérifier que le classement des modèles est robuste selon la saison.

- [ ] Créer `scripts/run_rolling_cv.py` et une fonction de plis dans `src/data_processing.py`
      (`rolling_folds(...)`).
- [ ] 6 plis trimestriels, de juin 2008 à avril 2010 (avant la borne de validation actuelle) ;
      au moins 18 mois d'entraînement pour le premier pli. Le test final reste inchangé.
- [ ] Modèles : J-7, SARIMA, TCN 24 h avec calendrier (TCN 168 h et LSTM écartés : coût).
- [ ] Réseaux : 2 graines par pli, 4 époques fixes (meilleures époques observées : 2 à 7).
      Standardisation et imputation ajustées sur le train de chaque pli seulement.
- [ ] SARIMA : réajusté sur les 120 jours précédant chaque pli.
- [ ] Sorties : `results/rolling_cv.csv` (pli × modèle), moyenne ± écart-type par modèle,
      figure « MAE par pli ».
- [ ] Analyser : le TCN bat-il SARIMA sur tous les plis ou seulement l'été/automne ?
      Les premiers plis ont moins de données d'entraînement : à signaler.
- [ ] Coût estimé : ≈ 50 min de calcul, ≈ 1 Go de RAM. Fermer le navigateur avant de lancer
      (les tâches ont déjà été arrêtées par manque de mémoire).
- [ ] Honnêteté méthodologique : le test a déjà été consulté, donc la validation glissante sert à
      vérifier la robustesse, pas à refaire une sélection non biaisée. À écrire dans le compte rendu.

## 2. Livrables du sujet

- [ ] `notebooks/demo.ipynb` : notebook léger qui importe `src/` et charge `results/` (pas de
      réentraînement en direct) : tableau descriptif, EDA, baselines, comparaison, prévisions
      commentées avec intervalles, validation glissante.
- [ ] `README.md` : description, installation, exécution, résultats, limites.
- [ ] `setup_env.sh` : création du venv + `pip install -r requirements.txt`.
- [ ] `requirements.txt` : fixer les versions réellement utilisées (numpy 1.26.4, pandas 3.0.2,
      scipy 1.15.3, statsmodels 0.15.0, torch 2.14.1 CPU, scikit-learn 1.7.2, matplotlib 3.10.6,
      seaborn 0.12.2, holidays 0.106, requests 2.31.0).
- [ ] Supprimer `_inspect.py` (fichier temporaire).
- [ ] Anciennes sorties du notebook guide à nettoyer, ou le retirer au profit de `demo.ipynb`.

## 3. Pistes d'analyse du sujet encore à couvrir

- [ ] Horizon de prédiction optimal : l'erreur est plate de 1 h à 24 h pour les deux modèles ;
      l'écrire clairement, en séparant l'effet de l'heure de la journée.
- [ ] Impact des vacances scolaires (calendrier externe) : non fait, seulement les jours fériés.
- [ ] Météo : absente du jeu UCI (limite à citer ; piste : apparier une source horaire).
- [ ] Optionnel : LSTNet / N-BEATS (publications de référence), pertes asymétriques pour les pics.

## 4. Présentation

- [ ] Point mi-parcours : EDA, baseline, premiers LSTM, axes d'amélioration (déjà couverts).
- [ ] Présentation finale (15 min + démo) : plan, messages clés, limites.

## Notes techniques

- Origines : entraînement au pas de 1 h, validation/test au pas de 6 h (0h, 6h, 12h, 18h). Le pas de
  24 h du notebook initial faisait toutes les prévisions à 17h, ce qui mélangeait horizon et
  heure de la journée.
- Environnement : l'Anaconda local avait des DLL C++ (14.27) qui empêchaient `import torch`. Elles
  ont été remplacées (sauvegarde dans `anac3\_vc_backup_14.27`). Un venv propre évite ce problème.
- Le hook claude-mem bloque l'outil de lecture d'images (Git Bash absent) : vérifier les figures
  à l'œil.
