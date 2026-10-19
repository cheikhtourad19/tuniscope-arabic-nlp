# TuniScope — Baselines lexicales et portée locale de négation

**Livrables :** L04, L07, L10 et L12, contributions partielles. Validation v2 uniquement, paramètres fixés avant ce run.

## Résultats sur tous les exemples

Les métriques incluent le recours majoritaire quand le vote n'est pas décisif. La couverture est la fraction de votes décisifs, pas la proportion de textes auxquels le système retourne un label.

| Corpus | Système | Macro-F1 global | Accuracy | Couverture décisive | Recours majoritaire |
| --- | --- | ---: | ---: | ---: | ---: |
| tsac | majority | 0.343206 | 0.522548 | 0.00 % | 1774/1774 |
| tsac | lexicon | 0.720134 | 0.744645 | 54.85 % | 801/1774 |
| tsac | lexicon_negation | 0.718898 | 0.743517 | 54.85 % | 801/1774 |
| madar | majority | 0.222222 | 0.500000 | 0.00 % | 800/800 |
| madar | lexicon | 0.882974 | 0.891250 | 79.38 % | 165/800 |

## Effet apparié de la négation sur TSAC

Même lexique et mêmes textes : 4 prédictions changent. La variante avec négation corrige 1 erreur du lexique seul et introduit 3 nouvelles erreurs. Les labels source restent inchangés.

## Comparaison descriptive avec les caractères

| Corpus | Règles | Erreurs caractères corrigées | Erreurs nouvelles face aux caractères |
| --- | --- | ---: | ---: |
| tsac | lexicon | 76 | 403 |
| tsac | lexicon_negation | 76 | 405 |
| madar | lexicon | 13 | 61 |

## Méthode et limites

Les lexiques sont statistiquement induits du train : fréquence documentaire lissée par classe, seuils fixes et au plus 200 termes par classe. Ce ne sont pas des lexiques experts ni des preuves qu'un terme possède intrinsèquement un sentiment ou un dialecte. Noms propres et effets de domaine sont possibles.

Le vote lexical est déterministe. La variante de négation inverse les indices dans une fenêtre de trois mots, arrêtée aux frontières et adversatifs reconnus. Les marqueurs/ruptures/exceptions ne sont activés que s'ils sont attestés sur train. La parité des négations est une heuristique qui ne modélise pas la concordance négative, les cibles multiples ou l'ironie.

Émojis ignorés par ce lexique, graphies non reconnues, absence d'indices et égalités aboutissent parfois au recours majoritaire. Le vote n'est pas une probabilité calibrée. Les cas non couverts ne sont jamais supprimés du macro-F1 principal. Les petites slices restent signalées dans les métriques.

Le contraste avec les caractères aligne les identifiants, groupes et labels, mais compare des représentations et traitements différents. Aucun gain significatif ou causal, aucune généralisation au test officiel et aucun résultat d'intégration lexique+TF-IDF ne sont revendiqués.

## Vérification et ressources

Smoke test préalable sur 100 lignes par partition/tâche, puis run complet CPU. Les lexiques JSON sont reconstruits intégralement depuis le train, les prédictions et indicateurs rejoués, les matrices de confusion et macro-F1 recalculés indépendamment. Le checkpoint caractères de référence est également vérifié. Les lexiques et sorties par exemple restent privés.

Les durées et pics RSS cumulatifs au processus figurent dans summary.json. Aucun GPU ou modèle neuronal utilisé. Les sources et manifestes restent inchangés. La relecture des analyses linguistiques, l'éventuelle intégration contrôlée au meilleur classifieur, les notebooks et l'évaluation finale restent ouverts.
