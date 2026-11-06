# Contrôle initial CAMeLBERT sur Kaggle CUDA

**5 novembre 2026 — contribution à L05 et L12 ; livrables encore partiels.**

## Périmètre et provenance

Nous exécutons les mêmes scripts et la même configuration que sur Mac dans un environnement encodeur isolé Python 3.12.15/PyTorch 2.15.0. Le diagnostic préalable identifie deux Tesla T4 ; le programme utilise un seul GPU CUDA. Le noyau Kaggle reste distinct de cet environnement. Aucun parallélisme multi-GPU ni changement de précision n’est appliqué.

Le commit exécuté est `5e7768e25d6804953efcbf9851ed3230081a523e`, avec arbre suivi déclaré propre. La configuration est `configs/camelbert_finetuning_v1.json`, représentation raw-v0 et manifestes v2. Nous vérifions localement les empreintes du code, des verrous, du protocole et des manifestes. Les résumés de partitions, notamment les empreintes des identifiants, correspondent exactement à ceux du smoke MPS avec journal corrigé.

Le smoke utilise 100 exemples de train et 100 de validation par tâche, deux époques, graine 42, avec deux régimes : tête CLS seule apprise ou adaptation de tous les paramètres. Il ne sélectionne aucune ligne de test. Ces scores ne remplacent pas les résultats FULL du Mac.

## Résultats du contrôle

| Corpus | Régime | Macro-F1 | Accuracy | Dernière époque train MPS (s) | CUDA (s) | Ratio MPS/CUDA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | frozen | 0.328859 | 0.490000 | 0.510952 | 0.375312 | 1.36 |
| TSAC | finetuned | 0.612403 | 0.620000 | 2.116272 | 1.143187 | 1.85 |
| MADAR | frozen | 0.168543 | 0.230000 | 0.349882 | 0.215694 | 1.62 |
| MADAR | finetuned | 0.825919 | 0.830000 | 1.607703 | 0.894860 | 1.80 |

Durée totale du run CUDA : **43.743977 s**, contre **22.061002 s** pour ce smoke MPS. Ces totaux incluent chargements, empreintes, sauvegardes, apprentissage et validation ; ils excluent le rejeu ultérieur. Les ratios par époque isolent mieux la boucle d’apprentissage, mais un seul petit essai ne constitue pas un benchmark répété ni une estimation garantie du débit FULL.

## Preuves reçues et vérifications locales

L’archive transmise contient uniquement `results.json`, `verification.json` et `summary.txt`. Nous contrôlons son intégrité, le lien SHA-256 entre résultats et vérification, la cohérence des quatre configurations de modèles, les identifiants des exemples et les métriques globales et par classe recalculées depuis les matrices de confusion.

Le vérificateur exécuté sur Kaggle rapporte quatre modèles rechargés, 400 prédictions de validation reproduites avec une différence maximale de probabilité de zéro, et huit époques dont les métriques et prédictions archivées ont été contrôlées. L’archive reçue ne contient pas les poids, prédictions individuelles ou journaux d’époques : nous ne revendiquons donc pas un nouveau rejeu local ni un recalcul indépendant des huit époques.

SHA-256 de l’archive : `fcfee3bceb63cd38c81910440acb9cc226e1d33a2642d766e01e59d03a8568bc`. SHA-256 des résultats : `2c3a599d16c2c52b5b9b46c26ea0fdbac46910fad7e0b253fe27e0bab07b07a8`. Les pièces et le relevé de contrôles sont conservés sous `outputs/cloud_checks/2026-11-05/kaggle_t4_paired_smoke/`, exclu de Git.

## Décision et limites

Le contrôle établit la compatibilité initiale du profil verrouillé et du code d’apprentissage CAMeLBERT avec le CUDA fourni par cette session Kaggle. Nous conservons le Mac pour le développement et l’analyse ; Kaggle devient une possibilité vérifiée pour des expériences ultérieures plus longues. Le second T4, le débit sur toutes les données, les runs finaux, la relance indépendante et le LLM/LoRA restent à vérifier séparément. Aucune modification des partitions, aucun remplacement des modèles applicatifs et aucune évaluation du test officiel ne résultent de ce contrôle.

La prochaine étape scientifique proposée reste l’analyse des erreurs du fine-tuning complet existant, avant une confirmation de configuration finale. Aucun nouvel entraînement complet n’est nécessaire pour importer ces preuves.
