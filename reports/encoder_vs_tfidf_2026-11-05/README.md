# Comparaison des erreurs CAMeLBERT / TF-IDF

Validation FULL v2, texte brut, trois graines conservées. Aucun nouvel entraînement ni évaluation du test.

| Corpus | Graine | Erreurs TF-IDF | Erreurs encodeur | Corrigées | Introduites | Communes | Correctes pour les deux |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TSAC | 13 | 126 | 109 | 87 | 70 | 39 | 1578 |
| TSAC | 42 | 126 | 122 | 85 | 81 | 41 | 1567 |
| TSAC | 2026 | 126 | 109 | 81 | 64 | 45 | 1584 |
| MADAR | 13 | 39 | 35 | 26 | 22 | 13 | 739 |
| MADAR | 42 | 39 | 38 | 23 | 22 | 16 | 739 |
| MADAR | 2026 | 39 | 40 | 27 | 28 | 12 | 733 |

« Corrigées » : TF-IDF se trompe et CAMeLBERT retrouve le label de référence. « Introduites » : TF-IDF retrouve le label de référence et CAMeLBERT se trompe. « Communes » : les deux se trompent, éventuellement avec des prédictions différentes.

Les comptes suivent les annotations du corpus ; ils ne constituent pas une validation humaine des labels. Les quatre catégories couvrent tous les exemples de validation. Leurs effectifs ne sont pas des catégories linguistiques.

Les identifiants, groupes, labels, empreintes des fichiers et métriques recalculées sont contrôlés. Le rapport réutilise la preuve antérieure de rechargement de l’encodeur, sans refaire l’inférence. La revue sémantique d’au moins 30 erreurs distinctes par tâche reste à effectuer. L07/L10 restent partiels.

Reproduction depuis les runs locaux autorisés :

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/compare_encoder_errors.py --output reports/encoder_vs_tfidf_2026-11-05
```
