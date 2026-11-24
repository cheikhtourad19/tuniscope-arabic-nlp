# Évaluation technique du test documentaire

Le protocole est gelé avant calcul des scores. Les références de pertinence restent proposées par IA, avec double revue humaine en attente. Quatre questions figuraient déjà dans l’échantillon de revue publié.

| Méthode | Recall@3 | MRR@3 | Réponses pertinentes émises | Réponses non pertinentes | Refus attendus |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.8438 | 0.7760 | 19/32 | 9/28 | 4/8 |
| E5 | 0.9375 | 0.8854 | 13/32 | 3/16 | 6/8 |

Les passages émis sont des extraits littéraux. Une citation existante peut être non pertinente pour la question. Les refus sur questions documentées sont au nombre de huit pour BM25 et 18 pour E5.

[Protocole gelé](../../configs/rag_documentary_frozen_v1.json) · [Résultats détaillés](results.json) · [Vérification](verification.json) · [Archive des sources](source_archive.json)

[Recherche](figures/recherche.png) · [Réponses](figures/reponses.png) · [Toutes les décisions](figures/decisions.png)

Cette expérience concerne les 40 questions documentaires. Le test officiel de classification reste réservé. Les limites de revue empêchent de déclarer Silver entièrement conforme à ce stade.
