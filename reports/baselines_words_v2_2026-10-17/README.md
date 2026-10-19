# TuniScope — Premières baselines : résultats de validation

**Livrables concernés :** L04 et L07, contributions partielles ; L12 pour la traçabilité.

Ce rapport est généré par `scripts/render_baseline_report.py` depuis les prédictions vérifiées. Il porte uniquement sur la validation des manifestes v2. Le test officiel n'a pas été chargé.

Configuration `baseline-words-v2`, Python 3.12.15, scikit-learn 1.9.1, Darwin arm64, CPU, un thread.

Commit de départ : `384f8a5aba14650b75ebd7a47a558433b56553ce` ; fichiers non commités lors du run : `true`. Les empreintes du code, du verrouillage et des manifestes figurent dans `metrics.json`.

## Résultats

| Corpus | Système | Texte | Train | Validation | Macro-F1 | Accuracy | Balanced accuracy | Ajustement (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tsac | majority | raw-v0 | 10053 | 1774 | 0.343206 | 0.522548 | 0.500000 | 0.002 |
| tsac | tfidf_word | raw-v0 | 10053 | 1774 | 0.872023 | 0.872041 | 0.874353 | 0.109 |
| tsac | tfidf_word | norm-nfc-space-v1 | 10053 | 1774 | 0.872023 | 0.872041 | 0.874353 | 0.101 |
| madar | majority | raw-v0 | 6388 | 800 | 0.222222 | 0.500000 | 0.333333 | 0.001 |
| madar | tfidf_word | raw-v0 | 6388 | 800 | 0.917075 | 0.922500 | 0.903333 | 0.120 |
| madar | tfidf_word | norm-nfc-space-v1 | 6388 | 800 | 0.917075 | 0.922500 | 0.903333 | 0.119 |

## Contraste normalisé − brut

Ces écarts concernent la même baseline et les mêmes exemples. Ils ne constituent pas un résultat final sur trois graines ni une preuve de significativité.

- tsac : Δ macro-F1 = +0.000000 (+0.0000 point(s) de pourcentage), 0 prédiction(s) différente(s), 227 erreur(s) en brut et 227 en normalisé.
- madar : Δ macro-F1 = +0.000000 (+0.0000 point(s) de pourcentage), 0 prédiction(s) différente(s), 62 erreur(s) en brut et 62 en normalisé.

L'analyseur mots utilise la segmentation Unicode déclarée dans la configuration. Les espaces et la ponctuation servent déjà de séparateurs ; les émojis ne sont pas représentés. Les lignes sans caractéristique retenue restent évaluées. Le contraste brut/normalisé porte sur NFC/espaces.

## Résultats par classe

| Corpus | Système | Texte | Classe | Précision | Rappel | F1 | Support validation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| tsac | majority | raw-v0 | NEG | 0.000000 | 0.000000 | 0.000000 | 847 |
| tsac | majority | raw-v0 | POS | 0.522548 | 1.000000 | 0.686412 | 927 |
| tsac | tfidf_word | raw-v0 | NEG | 0.827004 | 0.925620 | 0.873538 | 847 |
| tsac | tfidf_word | raw-v0 | POS | 0.923729 | 0.823085 | 0.870508 | 927 |
| tsac | tfidf_word | norm-nfc-space-v1 | NEG | 0.827004 | 0.925620 | 0.873538 | 847 |
| tsac | tfidf_word | norm-nfc-space-v1 | POS | 0.923729 | 0.823085 | 0.870508 | 927 |
| madar | majority | raw-v0 | MSA | 0.000000 | 0.000000 | 0.000000 | 200 |
| madar | majority | raw-v0 | TN | 0.500000 | 1.000000 | 0.666667 | 400 |
| madar | majority | raw-v0 | EG | 0.000000 | 0.000000 | 0.000000 | 200 |
| madar | tfidf_word | raw-v0 | MSA | 0.937500 | 0.900000 | 0.918367 | 200 |
| madar | tfidf_word | raw-v0 | TN | 0.894977 | 0.980000 | 0.935561 | 400 |
| madar | tfidf_word | raw-v0 | EG | 0.976471 | 0.830000 | 0.897297 | 200 |
| madar | tfidf_word | norm-nfc-space-v1 | MSA | 0.937500 | 0.900000 | 0.918367 | 200 |
| madar | tfidf_word | norm-nfc-space-v1 | TN | 0.894977 | 0.980000 | 0.935561 | 400 |
| madar | tfidf_word | norm-nfc-space-v1 | EG | 0.976471 | 0.830000 | 0.897297 | 200 |

## Traçabilité et limites

Les matrices de confusion, scores par écriture, tailles des vocabulaires, itérations, temps, empreintes et tailles des checkpoints sont dans `metrics.json`. Les classes sans prédiction reçoivent une précision/rappel/F1 nul selon `zero_division=0`. Les sous-groupes trop petits ou privés d'une classe sont signalés et ne doivent pas servir à une conclusion globale.

La mémoire maximale enregistrée est le pic du processus depuis son démarrage, donc cumulé entre systèmes ; elle n'est pas une mesure isolée du coût de chaque modèle. Le temps d'ajustement comprend le vectoriseur et le classifieur. Le modèle majoritaire utilise les fréquences du train uniquement. Les modèles TF-IDF ajustent vocabulaire et pondérations sur train uniquement ; les checkpoints ont été rechargés et leurs prédictions vérifiées.

Les prédictions ligne par ligne et les checkpoints contenant des vocabulaires restent sous `outputs/baselines/`, ignoré par Git. Le présent rapport ne contient aucun texte de corpus. Les comparaisons classiques complètes, règles, analyses d'erreurs, modèles neuronaux et résultats finaux restent à réaliser.

La v2 intègre les 12 décisions humaines de même sens et conserve ensemble toutes les familles candidates de la règle de répétitions, y compris celles qui ne traversaient pas les partitions v1. Cette précaution de partitionnement ne transforme pas les candidats non relus en équivalences sémantiques validées. Les variantes raw/norm d'un même run restent comparables, mais les scores TSAC v1 et v2 n'utilisent pas les mêmes exemples de validation : leur différence n'isole pas l'effet de la correction. Les résultats restent provisoires, sans garantie d'absence de toute similarité résiduelle ni comparaison directe à des benchmarks utilisant d'autres partitions.

## Lignes sans caractéristique retenue

Ces lignes sont conservées dans les prédictions et métriques, sans filtrage.

| Corpus | Système | Texte | Train | Validation |
| --- | --- | --- | ---: | ---: |
| tsac | tfidf_word | raw-v0 | 744 | 172 |
| tsac | tfidf_word | norm-nfc-space-v1 | 744 | 172 |
| madar | tfidf_word | raw-v0 | 46 | 11 |
| madar | tfidf_word | norm-nfc-space-v1 | 46 | 11 |
