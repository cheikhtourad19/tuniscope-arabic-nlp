# TuniScope — Caractères et indices lexicaux

**Livrables :** L04, L07, L10 et L12, contributions partielles. Validation v2, texte brut et configurations fixées avant ce run.

Les contrôles reproduisent exactement les prédictions historiques. Tous les exemples sont inclus ; aucune correction de label, sélection de meilleure graine ou ouverture du test officiel.

| Corpus | Système | Macro-F1 | Accuracy | Erreurs | Colonnes ajoutées |
| --- | --- | ---: | ---: | ---: | ---: |
| tsac | char_control | 0.928898 | 0.928974 | 126 | 0 |
| tsac | char_lexicon | 0.925016 | 0.925028 | 133 | 3 |
| tsac | char_lexicon_negation | 0.925580 | 0.925592 | 132 | 3 |
| madar | char_control | 0.949244 | 0.951250 | 39 | 0 |
| madar | char_lexicon | 0.905184 | 0.911250 | 71 | 4 |

## Comparaisons appariées avec le contrôle

| Corpus / ajout | Écart macro-F1 | Erreurs corrigées | Erreurs introduites | Prédictions différentes |
| --- | ---: | ---: | ---: | ---: |
| tsac:char_lexicon | -0.003882 | 43 | 50 | 93 |
| tsac:char_lexicon_negation | -0.003318 | 42 | 48 | 90 |
| madar:char_lexicon | -0.044060 | 13 | 45 | 59 |

## Négation dans le bloc lexical TSAC

Comparée à l'ajout lexical sans négation : 13 prédictions différentes, erreurs corrigées : 7, erreurs introduites : 6.

## Interprétation et limites

Le bloc ajouté contient les fractions de voix par classe et un indicateur de présence d'indice : trois colonnes TSAC, quatre MADAR. Ces valeurs ne sont pas des probabilités calibrées. Le poids du bloc est fixé à 1 ; aucune normalisation globale ne modifie les colonnes caractères.

Les lexiques restent ajustés uniquement sur train et identiques aux ressources de la baseline à règles. Le classifieur apprend à pondérer les nouvelles informations ; aucune règle n'écrase directement sa prédiction. Les associations lexicales sont supervisées et calculées sur le train qui sert aussi au classifieur : elles peuvent surajuster celui-ci. Les graphies rares, noms propres, effets de domaine et limites de portée restent présents.

Les nouveaux modèles partagent les mêmes vocabulaires/IDF caractères, exemples, paramètres du classifieur et budget de recherche. L'ajout change néanmoins la dimension et la géométrie de l'espace. Les écarts de validation ne prouvent ni significativité, ni gain sur le test final. Une amélioration sur quelques exemples ne justifie pas une généralisation à toutes les variétés ou sources.

## Vérification et suite

Tests fictifs, smoke test avant run complet, checkpoints rechargés et toutes les prédictions rejouées. Les blocs sont vérifiés numériquement, les lexiques reconstruits depuis le train, les références historiques revérifiées, et le macro-F1 recalculé depuis les matrices de confusion. Les coûts et nombres de blocs nuls figurent dans summary.json ; la mémoire maximale est cumulative au processus. Aucune dépendance ou poids neuronal ajouté.

Les notebooks, autres baselines requises, relecture humaine des études linguistiques et évaluation finale restent à réaliser. Aucun lexique, vocabulaire, checkpoint ou texte de corpus n'est publié dans ce rapport.
