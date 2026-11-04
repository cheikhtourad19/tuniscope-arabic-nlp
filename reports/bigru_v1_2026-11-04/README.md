# BiGRU — résultats de validation v2

Run `2026-11-04_bigru_v1_cpu_full`, texte raw-v0, appareil `cpu`. Six modèles, trois graines par tâche.

| Corpus | Graine | Époque retenue | Macro-F1 | Accuracy | Paramètres |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 13 | 12 | 0.847140 | 0.847238 | 479106 |
| TSAC | 42 | 9 | 0.855657 | 0.855693 | 479106 |
| TSAC | 2026 | 6 | 0.856806 | 0.856821 | 479106 |
| MADAR | 13 | 9 | 0.904381 | 0.911250 | 307331 |
| MADAR | 42 | 9 | 0.895672 | 0.901250 | 307331 |
| MADAR | 2026 | 6 | 0.899366 | 0.906250 | 307331 |

## Moyenne et variabilité

- TSAC : macro-F1 moyen **0.853201 ± 0.005281**, écart-type d'échantillon sur trois graines. Référence TF-IDF caractères brut sur les mêmes exemples : 0.928898.
- MADAR : macro-F1 moyen **0.899806 ± 0.004371**, écart-type d'échantillon sur trois graines. Référence TF-IDF caractères brut sur les mêmes exemples : 0.949244.

Sélection d'époque sur validation, égalités au premier checkpoint. Aucun choix de meilleure graine. Les scores ne sont pas ceux du test officiel ; ils ne prouvent pas une généralisation hors domaine. La comparaison des systèmes change à la fois représentation et classifieur.

Vocabulaire et audit reconstruits depuis le train ; six checkpoints rechargés, prédictions rejouées et métriques recalculées. Les courbes et artefacts individuels restent dans le run privé. L04/L07/L12 restent partiels.
