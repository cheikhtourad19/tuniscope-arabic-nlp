# TuniScope — Mots, caractères et combinaison

**Livrables :** L04, L07 et L12, contributions partielles.

Validation v2 commune : mêmes IDs, groupes, labels et configurations de classifieur. Les trois runs et leurs checkpoints sont vérifiés. Les vocabulaires et IDF des deux branches combinées sont identiques à ceux des références seules. Aucun test officiel chargé.

| Corpus | Variante | Macro-F1 mots | Macro-F1 caractères | Macro-F1 combinaison | Erreurs combinaison |
| --- | --- | ---: | ---: | ---: | ---: |
| tsac | raw-v0 | 0.872023 | 0.928898 | 0.923817 | 135 |
| tsac | norm-nfc-space-v1 | 0.872023 | 0.928893 | 0.923817 | 135 |
| madar | raw-v0 | 0.917075 | 0.949244 | 0.938384 | 46 |
| madar | norm-nfc-space-v1 | 0.917075 | 0.949244 | 0.938384 | 46 |

## Écarts appariés

| Corpus | Variante | Référence | Δ macro-F1 combinaison − référence | Référence seule correcte | Combinaison seule correcte | Deux systèmes en erreur |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| tsac | raw-v0 | char | -0.005082 | 30 | 21 | 105 |
| tsac | raw-v0 | word | +0.051794 | 23 | 115 | 112 |
| tsac | norm-nfc-space-v1 | char | -0.005076 | 30 | 21 | 105 |
| tsac | norm-nfc-space-v1 | word | +0.051794 | 23 | 115 | 112 |
| madar | raw-v0 | char | -0.010861 | 19 | 12 | 27 |
| madar | raw-v0 | word | +0.021308 | 3 | 19 | 43 |
| madar | norm-nfc-space-v1 | char | -0.010861 | 19 | 12 | 27 |
| madar | norm-nfc-space-v1 | word | +0.021308 | 3 | 19 | 43 |

## Portée

La combinaison est une nouvelle régression logistique sur les colonnes juxtaposées, pas un vote ou une sélection a posteriori. Blocs de poids 1:1 puis normalisation L2 globale, C=1. L'échelle de chaque bloc et la régularisation effective changent par rapport aux références seules ; ce protocole compare des systèmes, pas un effet isolé de feature.

Les lignes sans caractéristique restent évaluées. La recherche de familles est heuristique et les sous-groupes très petits restent fragiles. La validation n'établit ni significativité ni supériorité générale ; aucune recherche de poids ou de C n'a suivi les scores.

Les scores sont recalculés depuis les prédictions. `comparison.json` contient les comptes et empreintes, sans commentaires ni identifiants privés. Les règles, analyses linguistiques et humaines d'erreurs, autres baselines et évaluation finale restent à réaliser.
