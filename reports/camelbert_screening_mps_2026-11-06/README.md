# Criblage CAMeLBERT complet sur Mac

Campagne locale lancée par nous depuis notebook07, graine42, trois taux, donnéesv2brutes. Six modèles adaptés, rechargements du pipeline et journaux des18époques vérifiés ; sixfigures liées aux résultats par empreinte. Aucun test officiel évalué.

| Taux | TSAC Macro-F1 | MADAR Macro-F1 | Durée FULL Mac | Durée FULL T4, journal arrondi |
| ---: | ---: | ---: | ---: | ---: |
| 1e-05 | 0.933102 | 0.950401 | 937.877 s | 533.2 s |
| 2e-05 | 0.928898 | 0.956013 | 925.903 s | 539.6 s |
| 5e-05 | 0.936585 | 0.948276 | 934.468 s | 537.2 s |

Somme FULL : **2798.248 s sur Mac**, contre **1610.0 s sur un T4**, soit un ratio de durée **1.738**. Hors smokes, rechargements et exports ultérieurs ; aucune mesure répétée ni accélération garantie sur d’autres tâches.

Somme des smokes Mac : **66.281 s**. Ces temps ne représentent pas la durée totale du notebook, qui inclut aussi vérifications, figures et travail du noyau.

**Taux proposés par la campagne Mac : TSAC5e-5, MADAR2e-5.** Le journal Kaggle proposait2e-5 pour les deux tâches. Nous conservons ces deux observations séparément et ne mélangeons pas les scores pour choisir une plateforme ou une meilleure graine. Des graines égales ne garantissent pas des trajectoires numériques identiques entre appareils ; le mécanisme précis de cet écart n’est pas établi par cette comparaison. Confirmation sur trois graines encore nécessaire.

La campagne locale et ses artefacts sont conservés sous `outputs/encoder_finetuning/screening/notebooks/708fce3a508b4bb6b1a7d2fbce54739c_full`. L’agent a contrôlé empreintes, alignements et métriques ; les rechargements sont ceux du pipeline utilisateur, sans nouvel entraînement ni nouvelle inférence ajoutés par cette lecture.

L05/L07/L12 restent partiels : criblage local complet conservé, confirmation finale sur trois graines et autres exigences d’évaluation distinctes. Les fichiers Kaggle d’origine restent à rechercher séparément.
