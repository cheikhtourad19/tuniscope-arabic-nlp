# TuniScope — Comparaison TF-IDF mots et caractères

**Livrables :** L04, L07 et L12, contributions partielles.

Manifestes v2, mêmes identifiants/groupes/labels de validation. Prédictions et checkpoints vérifiés, métriques recalculées. Aucun test officiel chargé.

| Corpus | Variante | Macro-F1 caractères | Macro-F1 mots | Écart mots − caractères | Erreurs caractères | Erreurs mots |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tsac | raw-v0 | 0.928898 | 0.872023 | -0.056875 | 126 | 227 |
| tsac | norm-nfc-space-v1 | 0.928893 | 0.872023 | -0.056870 | 126 | 227 |
| madar | raw-v0 | 0.949244 | 0.917075 | -0.032169 | 39 | 62 |
| madar | norm-nfc-space-v1 | 0.949244 | 0.917075 | -0.032169 | 39 | 62 |

## Complémentarité des erreurs

| Corpus | Variante | Caractères seuls corrects | Mots seuls corrects | Deux systèmes en erreur | Vecteurs mots nuls en validation |
| --- | --- | ---: | ---: | ---: | ---: |
| tsac | raw-v0 | 142 | 41 | 85 | 172 |
| tsac | norm-nfc-space-v1 | 142 | 41 | 85 | 172 |
| madar | raw-v0 | 38 | 15 | 24 | 11 |
| madar | norm-nfc-space-v1 | 38 | 15 | 24 | 11 |

## Interprétation et limites

Les éléments comparés diffèrent par les unités, les étendues de n-grammes et le traitement des symboles. Les autres paramètres communs sont contrôlés. Ces scores de validation ne prouvent pas une supériorité générale ni un écart significatif sur le test final.

La baseline mots ignore ponctuation et émojis et ne réalise pas de segmentation morphologique arabe. Les lignes sans feature après segmentation/min_df restent dans les métriques. Les vocabulaires sont ajustés sur train seulement.

Les cas corrigés par un seul système décrivent une complémentarité observée. Leur somme ne constitue pas la performance d'une combinaison réelle : il faudrait savoir à l'avance quel système choisir. Une combinaison doit être entraînée et évaluée séparément.

Les comptes et empreintes sont dans `comparison.json`. Les prédictions ligne par ligne et vocabulaires restent privés. Les analyses d'erreurs humaines, règles linguistiques, combinaison et évaluation finale restent à réaliser.
