# CAMeLBERT gelé — résultats de validation

Rapport généré depuis un run FULL explicite ; aucun score du test officiel. Contributions partielles à L05/L07/L12.

| Corpus | Graine | Accuracy | Macro-F1 | Époque retenue | Époques exécutées |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 13 | 0.841601 | 0.841001 | 16 | 21 |
| TSAC | 42 | 0.848365 | 0.847584 | 25 | 30 |
| TSAC | 2026 | 0.846110 | 0.845241 | 23 | 28 |
| MADAR | 13 | 0.923750 | 0.919112 | 3 | 8 |
| MADAR | 42 | 0.925000 | 0.920714 | 7 | 12 |
| MADAR | 2026 | 0.930000 | 0.925954 | 14 | 19 |

| Corpus | TF-IDF caractères | BiGRU, moyenne 3 graines | CAMeLBERT gelé, moyenne ± écart-type |
| --- | ---: | ---: | ---: |
| TSAC | 0.928898 | 0.853201 | 0.844608 ± 0.003337 |
| MADAR | 0.949244 | 0.899806 | 0.921926 ± 0.003579 |

Durée réelle : **86.0 s**, encodeur **mps**, têtes CPU.
Backbone chargé : **108 490 752** paramètres gelés ; têtes apprises : **1 538 / 2 307** paramètres.
Même validation v2 ; TF-IDF et BiGRU sont des références déjà fixées, pas une sélection de la meilleure graine.
Les têtes et métriques sont rejouées intégralement ; dix textes de validation par tâche sont réencodés depuis les poids originaux.
Les empreintes du backbone avant/après sont identiques. Le cache de vecteurs et les poids restent privés.

Les poids généraux gelés ne dépassent pas TF-IDF caractères. Ces résultats ne mesurent pas le futur fine-tuning, ni une qualité universelle du modèle.
Le run est local MPS ; aucun résultat CUDA ni budget mémoire de fine-tuning n’est établi.

Provenance du run : `outputs/frozen_encoder/2026-11-04_camelbert_frozen_mps_full`.
Les PNG du run privé montrent les pertes, le macro-F1 et les matrices, séparément pour chaque graine.
