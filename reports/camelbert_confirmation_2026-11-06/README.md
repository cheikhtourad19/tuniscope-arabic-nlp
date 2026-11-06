# Confirmation de CAMeLBERT sur trois graines

Résultats de **validation v2**, générés depuis une campagne FULL contrôlée. Test officiel non évalué.

| Corpus | Taux | Graine | Macro-F1 | Accuracy | Époque retenue / exécutées |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 5e-05 | 13 | 0.933368 | 0.933484 | 3 / 3 |
| TSAC | 5e-05 | 42 | 0.930285 | 0.930665 | 3 / 3 |
| TSAC | 5e-05 | 2026 | 0.928850 | 0.928974 | 3 / 3 |
| MADAR | 2e-05 | 13 | 0.954004 | 0.956250 | 3 / 3 |
| MADAR | 2e-05 | 42 | 0.954370 | 0.957500 | 3 / 3 |
| MADAR | 2e-05 | 2026 | 0.952071 | 0.953750 | 3 / 3 |

| Corpus | Macro-F1 moyen ± écart-type échantillonnal | TF-IDF caractères brut | Différence |
| --- | ---: | ---: | ---: |
| TSAC | 0.930834 ± 0.002308 | 0.928898 | +0.001936 |
| MADAR | 0.953482 ± 0.001235 | 0.949244 | +0.004238 |

Les mêmes identifiants, groupes et labels de validation sont contrôlés pour toutes les graines et TF-IDF.
Les trois scores sont conservés, sans sélection de meilleure graine. Aucune significativité ni généralisation au test établie.
Les taux ont été sélectionnés sur cette validation ; ces moyennes ne sont pas une évaluation finale indépendante.
Stages FULL cumulés : 2777.361288 s. Campagne avant contrôles finaux : 2884.380476 s (smokes, rejeux et figures inclus).
Six checkpoints et six PNG conservés ; cartes provisoires vérifiées. Les données, prédictions individuelles et poids restent privés.
Campagne : `outputs/encoder_finetuning/confirmation/notebooks/177d9f5ad78b4ef4907e97d97abe96f8_full`.
L05 : confirmation de validation terminée ; L07/L08/L12 restent partiels (test, compléments d’évaluation, intégration et reproduction indépendante).
