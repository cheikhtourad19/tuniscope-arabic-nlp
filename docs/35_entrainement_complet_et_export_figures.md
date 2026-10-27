# Entraînement complet et export intégré des résultats

**Date :** 27 octobre 2026. **Contributions partielles :** L04, L07, L11 et L12. Ce document met à jour le parcours du notebook 02 après l'ajout du lecteur autonome décrit dans le document 34.

## Diagnostic du mode affiché

La copie de travail sauvegardée avant modification utilisait déjà `full` par défaut. Ses sorties contenaient un contrôle `smoke`, puis un entraînement complet : TSAC 10 053 train / 1 774 validation et MADAR 6 388 train / 800 validation. Le résumé imprimait une liste JSON de deux runs, avec le smoke en premier. Ce premier champ `mode: smoke` ne décrivait donc pas l'ensemble de l'exécution.

L'exécution personnelle sauvegardée référençait `outputs/baselines/notebooks/2d4f8e32d39c4fc7b757b5b395d4231d_characters_full`. Avant réécriture des cellules, le notebook et ses sorties ont été préservés dans `outputs/notebook_snapshots/2026-10-27/02_avant_integration_678271d21510.ipynb`, SHA-256 `678271d21510f096187903960fa461991bacae5206e542bcd04479f9215623b8`. Les anciens runs ne sont pas modifiés.

## Parcours intégré

Le notebook `02_baselines_classiques.ipynb` garde les fonctions réutilisables dans les scripts et orchestre :

1. Configuration, contrôle d'environnement et disponibilité de Matplotlib dans le noyau.
2. Smoke de 100 lignes par partition et tâche, vérifié avant le full.
3. Apprentissage sur l'intégralité du train et prédiction sur l'intégralité du dev, si le mode demandé est `full`.
4. Sélection explicite du run correspondant au mode demandé. Si un full est absent, aucun retour silencieux aux résultats smoke.
5. Tableau final lisible des six résultats du run sélectionné, avec effectifs, accuracy et macro-F1. Le tableau JSON smoke/full n'est plus imprimé.
6. Export et affichage de six PNG produits depuis ce même run. Dossier des modèles/JSON et dossier des images affichés séparément, puis message final avec le mode réellement utilisé.

Le mode par défaut reste `full`, représentation `characters`. Les variables `TUNISCOPE_BASELINE_MODE` et `TUNISCOPE_BASELINE_FAMILY` peuvent explicitement imposer d'autres réglages ; le mode effectif et l'origine de cette surcharge sont affichés. `off` conserve la vérification fictive sans corpus pour les contrôles logiciels ; `smoke` reste un petit contrôle clairement identifié.

Un full ne signifie pas apprendre et évaluer sur les mêmes lignes. Il utilise **tout le train pour apprendre et tout le dev pour mesurer**. Le test officiel reste réservé. Ni partitions, ni hyperparamètres, ni données, ni règles d'évaluation ne changent par cette intégration.

## Rôles des fichiers

| Fichier | Responsabilité |
| --- | --- |
| `scripts/run_baselines.py` | Apprendre et évaluer, écrire les modèles, prédictions et métriques. Pas de changement de la méthode. |
| `scripts/verify_baseline_run.py` | Recharger et vérifier les prédictions et métriques. |
| `scripts/notebook_support.py` | Orchestrer smoke/full, sélectionner la bonne étape, formater le tableau et déclencher l'export. |
| `scripts/plot_validation_results.py` | Recalculer les agrégats et produire les figures, sans apprentissage. |
| `notebooks/02_baselines_classiques.ipynb` | Paramétrer, lancer, afficher les résultats et les images. |

Les sorties détaillées des sous-processus sont capturées dans ce parcours concis. Une erreur reste visible : stdout/stderr sont imprimés avant propagation de l'exception, et la suite s'arrête. La variable `VIRTUAL_ENV` du noyau est retirée uniquement de l'environnement du sous-processus d'entraînement, pour cibler `.venv` sans avertissement ambigu ; l'environnement du noyau et celui de l'utilisateur restent inchangés.

## Produits et destinations

Chaque full écrit six modèles `.joblib`, six fichiers `.predictions.json` et `results.json`, comme auparavant. Le nouveau sous-dossier `figures/` contient six PNG et `summary.json`. Une réexécution de la cellule d'export ne remplace pas ce sous-dossier : un suffixe unique est ajouté s'il existe déjà.

Les figures ne sont jamais obtenues par sélection du « dernier dossier » ou depuis une référence historique fixée. Elles suivent le run retourné et vérifié par cet entraînement. Les résultats JSON restent disponibles, sans texte de corpus dans les PNG ou le résumé agrégé.

Matplotlib reste une dépendance optionnelle du noyau privé, définie dans `requirements-visualization.txt`. Le verrouillage d'entraînement `uv.lock`, `pyproject.toml` et `requirements.txt` restent inchangés. Si Matplotlib manque, le notebook échoue avant l'apprentissage avec une instruction d'installation, sans téléchargement automatique.

## Exécution complète vérifiée

Le notebook 02 modifié a été exécuté dans le noyau Jupyter local `tuniscope-notebook` avec son réglage par défaut `full`, quatre cellules de code, puis une assertion finale de contrôle. Un smoke vérifié a précédé le full :

`outputs/baselines/notebooks/eb2659131f30409faee36203e0ab9fac_characters_full`

Les six PNG sont dans le sous-dossier `figures/` de ce run. Six messages d'affichage d'images ont été reçus par le client Jupyter. Le mode du run sélectionné et celui de l'export valent tous deux `full`. Le notebook source reste propre, sans anciennes sorties smoke affichées.

Une seconde exportation contrôlée est conservée dans `figures_808ec6e70019/`, sans écraser la première : le pied de figure indique les scores de validation de l'expérience source, sans la formulation ambiguë « aucun entraînement » qui décrivait seulement l'étape de tracé. Le modèle a bien été entraîné avant la génération des figures.

| Caractères bruts | Train | Dev | Accuracy dev | Macro-F1 dev |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 10 053 | 1 774 | 0,9289740699 | 0,9288980670 |
| MADAR | 6 388 | 800 | 0,9512500000 | 0,9492441926 |

Les empreintes des six fichiers de prédictions sont identiques à celles de la référence historique v2, y compris les variantes normalisées. Les empreintes du résultat source et des six PNG concordent avec `figures/summary.json`. Les durées représentent ce nouveau run, pas l'ancien du 17 octobre.

Six nouveaux tests fictifs couvrent sélection du full, absence de fallback, mode smoke explicite, tableau lisible, environnement du sous-processus, visibilité des échecs et destination de l'export. La suite générale compte 119 tests : 117 réussites, deux erreurs des contrôles de propreté car les notebooks 01 et 03 contiennent des sorties personnelles sauvegardées. Celles-ci sont préservées ; aucune annonce de suite générale entièrement verte. Le notebook 02 passe sa validation structurelle et son exécution fictive en `off`.

## Portée et limites

Cette vérification locale couvre effectivement l'apprentissage complet et les figures dans le noyau Jupyter, mais pas l'interface VS Code de l'utilisateur, une relance Kaggle ou une installation vierge. Les modes mots et combinés conservent la même interface ; aucun nouveau full de ces deux familles n'est lancé par cette étape. Les scores restent ceux du dev, sans test officiel, trois graines finales ou bootstrap. Les courbes par époque des futurs modèles neuronaux ne sont pas inventées à partir de ces résultats.
