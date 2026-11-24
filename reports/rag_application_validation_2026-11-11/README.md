# Validation de la recherche documentaire applicative

Comparaison sur les mêmes 20 questions de validation, dont 16 avec réponse et 4 sans réponse dans les fiches. Le classement porte sur trois fiches dans la langue de la question. Les critères applicatifs sont conservés sans nouveau réglage.

| Méthode | Recall@3 | MRR@3 | Réponses avec citation pertinente | Sans réponse correctement rejetées |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.875 | 0.875 | 13/16 | 4/4 |
| E5 | 1.000 | 1.000 | 8/16 | 4/4 |

Recall@3 et MRR@3 mesurent le classement avant abstention. Les réponses mesurent les passages effectivement affichés après application des critères. Les 40 réponses applicatives ont été rejouées et comparées aux décisions évaluées.

Les annotations de pertinence restent des propositions IA. Les questions de validation ont déjà servi à la mise au point. Leurs scores ne constituent pas une évaluation indépendante. La double revue humaine est en attente. Les tests documentaire et de classification restent sans nouvelle évaluation.

[Résultats détaillés et empreintes](results.json) · [Vérification](verification.json)
