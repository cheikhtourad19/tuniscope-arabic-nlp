# TuniScope — Expériences Gold sur la validation

Validation v2. Les résultats de classification sur le test officiel restent à produire.

## Quantité de données et performance de TF-IDF caractères

Même validation TSAC (1 774 avis) et mêmes hyperparamètres à chaque essai. Le vocabulaire TF-IDF et la régression logistique sont ajustés à nouveau sur chaque sous-ensemble. Les graines fixent l’ordre des groupes. Les tailles sont imbriquées dans chaque graine. Les groupes restent entiers, ce qui explique les petits écarts d’effectifs.

| Budget cible | Avis d’entraînement min–max | Macro-F1 moyen | Écart-type |
| ---: | ---: | ---: | ---: |
| 10 % | 1007–1008 | 85,02 % | 0,82 % |
| 25 % | 2513–2514 | 88,62 % | 0,66 % |
| 50 % | 5024–5028 | 91,43 % | 0,15 % |
| 100 % | 10053–10053 | 92,89 % | 0,00 % |

L’écart-type est calculé entre trois graines. À 100 %, les trois essais reprennent les mêmes données dans le même ordre et reproduisent toutes les prédictions de la référence. Le score à 100 % est donc un point de référence déterministe.

## Abstention exploratoire de CAMeLBERT

Le modèle accepte un texte si son score le plus élevé atteint le seuil. La couverture est la part des textes acceptés. Le taux d’erreur est calculé uniquement parmi ces textes. Nous retenons pour chaque tâche le seuil de la grille le plus haut qui accepte au moins 90 % des textes pour chacune des trois graines de validation.

| Corpus | Seuil | Couverture | Erreurs parmi les textes acceptés |
| --- | ---: | ---: | ---: |
| TSAC | 0.00 | 100,00 % | 6,90 % (367/5322) |
| TSAC | 0.50 | 100,00 % | 6,90 % (367/5322) |
| TSAC | 0.60 | 98,93 % | 6,46 % (340/5265) |
| TSAC | 0.70 | 98,03 % | 5,98 % (312/5217) |
| TSAC | 0.80 | 97,14 % | 5,80 % (300/5170) |
| TSAC | 0.85 | 96,28 % | 5,52 % (283/5124) |
| TSAC | 0.90 (retenu) | 95,00 % | 5,16 % (261/5056) |
| TSAC | 0.95 | 92,30 % | 4,50 % (221/4912) |
| TSAC | 0.99 | 79,95 % | 2,75 % (117/4255) |
| MADAR | 0.00 | 100,00 % | 4,42 % (106/2400) |
| MADAR | 0.50 | 99,88 % | 4,34 % (104/2397) |
| MADAR | 0.60 | 99,17 % | 4,03 % (96/2380) |
| MADAR | 0.70 | 98,83 % | 3,88 % (92/2372) |
| MADAR | 0.80 | 98,29 % | 3,77 % (89/2359) |
| MADAR | 0.85 | 97,96 % | 3,70 % (87/2351) |
| MADAR | 0.90 | 96,96 % | 3,31 % (77/2327) |
| MADAR | 0.95 | 96,00 % | 2,91 % (67/2304) |
| MADAR | 0.99 (retenu) | 92,67 % | 1,93 % (43/2224) |

Les comptes regroupent trois séries de prédictions sur les mêmes textes. Le seuil et les performances sont calculés sur la même validation. Les scores sont les probabilités brutes du modèle et leur calibration reste à étudier.

## Comparaison appariée des erreurs

| Corpus | Graine | Erreurs TF-IDF | Erreurs CAMeLBERT | Corrigées par CAMeLBERT | Introduites par CAMeLBERT | Communes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSAC | 13 | 126 | 118 | 81 | 73 | 45 |
| TSAC | 42 | 126 | 123 | 89 | 86 | 37 |
| TSAC | 2026 | 126 | 126 | 81 | 81 | 45 |
| MADAR | 13 | 39 | 35 | 26 | 22 | 13 |
| MADAR | 42 | 39 | 34 | 26 | 21 | 13 |
| MADAR | 2026 | 39 | 37 | 25 | 23 | 14 |

Une erreur corrigée est un texte que TF-IDF classe mal et que CAMeLBERT classe correctement. Une erreur introduite correspond à la situation inverse. Ces comptes utilisent les labels de référence et les prédictions sur les mêmes textes.
