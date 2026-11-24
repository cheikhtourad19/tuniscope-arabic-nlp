# Évaluation Gold sur la validation

**22 novembre 2026.** Cette étape contribue à **L07 — Évaluation**, **L10 — Rapport** et **L12 — Reproductibilité**. Elle couvre l’expérience E04 sur la quantité de données, un choix exploratoire de seuils d’abstention et une comparaison chiffrée des erreurs sur la campagne de confirmation de CAMeLBERT.

## Quantité de données et sentiment tunisien

Nous reprenons le TF-IDF caractères sur texte brut et la régression logistique. Chaque graine définit un ordre pour les groupes d’exemples. Nous prenons les premiers groupes de cet ordre par strate de labels. Ainsi, le lot à 10 % est inclus dans celui à 25 %. Les groupes restent entiers, d’où de petits écarts autour des tailles cibles.

Nous réajustons le vocabulaire TF-IDF et le classifieur à chaque taille. Les **1 774 mêmes avis de validation** servent à mesurer les douze ajustements. À 100 %, les trois ajustements utilisent les mêmes **10 053 avis** dans le même ordre et reproduisent exactement les prédictions historiques.

Le macro-F1 moyen atteint **85,02 %**, **88,62 %**, **91,43 %** et **92,89 %** aux quatre tailles. Passer de la moitié à la totalité du train apporte **1,46 point** de macro-F1. L’expérience décrit cette configuration de TF-IDF sur TSAC. Les barres du graphique donnent l’écart-type entre les trois graines.

## Abstention de CAMeLBERT

Nous reprenons les scores par classe sauvegardés des six modèles CAMeLBERT confirmés. Le modèle répond si son score maximal atteint le seuil. La *couverture* est la part des textes acceptés. Le taux d’erreur porte sur ces textes. Notre règle exploratoire retient le seuil le plus élevé de la grille qui accepte au moins 90 % des textes **pour chacune des trois graines** de la tâche.

| Corpus | Seuil retenu | Couverture cumulée | Erreur en répondant à tout | Erreur parmi les réponses acceptées |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 0,90 | 95,00 % | 6,90 % | 5,16 % |
| MADAR | 0,99 | 92,67 % | 4,42 % | 1,93 % |

La couverture cumulée agrège trois séries de prédictions sur les mêmes exemples. Sur TSAC, le seuil 0,95 aurait une couverture cumulée de 92,30 %, mais une des graines ne couvrirait que 85,51 % des textes. Le seuil 0,90 satisfait la règle pour les trois graines. Les seuils et les taux proviennent de la même validation. Il reste à mesurer la calibration des scores et à évaluer les seuils sur des données finales distinctes.

## Comprendre les changements d’erreurs

Sur TSAC, pour la graine 42 de la **campagne de confirmation**, CAMeLBERT corrige **89** erreurs de TF-IDF et en introduit **86**. Les deux systèmes se trompent ensemble sur **37** avis. CAMeLBERT commet ainsi trois erreurs de moins pour cette graine, tout en changeant de nombreuses décisions. Ici, « corriger » signifie retrouver le label de référence là où TF-IDF se trompe.

Les revues antérieures fournissent des hypothèses linguistiques. Sur 30 erreurs TSAC revues pour TF-IDF, les catégories « avis mixte / ambigu » et « orthographe / Arabizi » contiennent respectivement 9 et 5 cas. Sur 30 erreurs TSAC revues pour la **campagne initiale** de CAMeLBERT, elles en contiennent 7 et 7. Dans les deux revues MADAR, on retrouve des formulations communes à plusieurs variétés. Chaque lot a sa propre sélection. Ces comptes décrivent les lots revus et ne permettent pas d’estimer la fréquence des difficultés sur tout le corpus ni d’attribuer une cause aux erreurs de la campagne de confirmation.

Nous avons saisi les avis sur les labels du lot TF-IDF. Une partie des catégories et les hypothèses ont été complétées avec l’IA. Les avis et catégories du lot CAMeLBERT ont été proposés avec l’IA et restent à confirmer. Voir les [revues TF-IDF](27_analyse_erreurs_classiques_assistee.md) et [CAMeLBERT](51_analyse_erreurs_encodeur_assistee.md).

## Preuves et reproduction

- [Tableaux générés](../reports/gold_validation_2026-11-22/README.md) et [détail par graine en JSON](../reports/gold_validation_2026-11-22/summary.json).
- [Courbe du budget](../reports/gold_validation_2026-11-22/figures/budget_tfidf_tsac.png) et [graphique d’abstention](../reports/gold_validation_2026-11-22/figures/abstention_camelbert.png).
- Le calcul a pris **4,54 s**, dont **3,50 s** pour les ajustements TF-IDF. Le rejeu confirme les sélections, les prédictions de contrôle à 100 %, les résultats chiffrés et les tableaux. Dix tests ciblés passent.

Depuis la racine, avec les sources autorisées et les prédictions locales conservées, choisir un dossier de sortie neuf.

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/run_gold_validation.py --output outputs/gold_reproduction
sh scripts/encoder_uv.sh run --locked --offline python scripts/plot_gold_validation.py outputs/gold_reproduction/summary.json --output outputs/gold_reproduction/figures
```

L’analyse d’erreurs reste partielle pour les autres composants appris. L’évaluation sur les jeux de défi et le test officiel de classification, le bootstrap de la comparaison principale et le rapport complet restent à réaliser.
