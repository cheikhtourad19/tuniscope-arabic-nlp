# TuniScope — Premières baselines : résultats de validation

**Livrables concernés :** L04 et L07, contributions partielles ; L12 pour la traçabilité.

Ce rapport est généré par `scripts/render_baseline_report.py` depuis les prédictions vérifiées. Il porte uniquement sur la validation des manifestes v2. Le test officiel n'a pas été chargé.

Configuration `baseline-grouped-v2`, Python 3.12.15, scikit-learn 1.9.1, Darwin arm64, CPU, un thread.

Commit de départ : `384f8a5aba14650b75ebd7a47a558433b56553ce` ; fichiers non commités lors du run : `true`. Les empreintes du code, du verrouillage et des manifestes figurent dans `metrics.json`.

## Résultats

| Corpus | Système | Texte | Train | Validation | Macro-F1 | Accuracy | Balanced accuracy | Ajustement (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tsac | majority | raw-v0 | 10053 | 1774 | 0.343206 | 0.522548 | 0.500000 | 0.003 |
| tsac | tfidf_char | raw-v0 | 10053 | 1774 | 0.928898 | 0.928974 | 0.929390 | 0.676 |
| tsac | tfidf_char | norm-nfc-space-v1 | 10053 | 1774 | 0.928893 | 0.928974 | 0.929339 | 0.706 |
| madar | majority | raw-v0 | 6388 | 800 | 0.222222 | 0.500000 | 0.333333 | 0.001 |
| madar | tfidf_char | raw-v0 | 6388 | 800 | 0.949244 | 0.951250 | 0.938333 | 0.830 |
| madar | tfidf_char | norm-nfc-space-v1 | 6388 | 800 | 0.949244 | 0.951250 | 0.938333 | 0.848 |

## Contraste normalisé − brut

Ces écarts concernent la même baseline et les mêmes exemples. Ils ne constituent pas un résultat final sur trois graines ni une preuve de significativité.

- tsac : Δ macro-F1 = -0.000005 (-0.0005 point(s) de pourcentage), 2 prédiction(s) différente(s), 126 erreur(s) par variante.
- madar : Δ macro-F1 = +0.000000 (+0.0000 point(s) de pourcentage), 0 prédiction(s) différente(s), 39 erreur(s) par variante.

L'analyseur de caractères de scikit-learn harmonise déjà les suites d'espaces. Le contraste mesure l'apport supplémentaire de NFC et du retrait des espaces périphériques.

## Résultats par classe

| Corpus | Système | Texte | Classe | Précision | Rappel | F1 | Support validation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| tsac | majority | raw-v0 | NEG | 0.000000 | 0.000000 | 0.000000 | 847 |
| tsac | majority | raw-v0 | POS | 0.522548 | 1.000000 | 0.686412 | 927 |
| tsac | tfidf_char | raw-v0 | NEG | 0.914845 | 0.938607 | 0.926573 | 847 |
| tsac | tfidf_char | raw-v0 | POS | 0.942541 | 0.920173 | 0.931223 | 927 |
| tsac | tfidf_char | norm-nfc-space-v1 | NEG | 0.915802 | 0.937426 | 0.926488 | 847 |
| tsac | tfidf_char | norm-nfc-space-v1 | POS | 0.941566 | 0.921251 | 0.931298 | 927 |
| madar | majority | raw-v0 | MSA | 0.000000 | 0.000000 | 0.000000 | 200 |
| madar | majority | raw-v0 | TN | 0.500000 | 1.000000 | 0.666667 | 400 |
| madar | majority | raw-v0 | EG | 0.000000 | 0.000000 | 0.000000 | 200 |
| madar | tfidf_char | raw-v0 | MSA | 0.984615 | 0.960000 | 0.972152 | 200 |
| madar | tfidf_char | raw-v0 | TN | 0.923077 | 0.990000 | 0.955368 | 400 |
| madar | tfidf_char | raw-v0 | EG | 0.982955 | 0.865000 | 0.920213 | 200 |
| madar | tfidf_char | norm-nfc-space-v1 | MSA | 0.984615 | 0.960000 | 0.972152 | 200 |
| madar | tfidf_char | norm-nfc-space-v1 | TN | 0.923077 | 0.990000 | 0.955368 | 400 |
| madar | tfidf_char | norm-nfc-space-v1 | EG | 0.982955 | 0.865000 | 0.920213 | 200 |

## Traçabilité et limites

Les matrices de confusion, scores par écriture, tailles des vocabulaires, itérations, temps, empreintes et tailles des checkpoints sont dans `metrics.json`. Les classes sans prédiction reçoivent une précision/rappel/F1 nul selon `zero_division=0`. Les sous-groupes trop petits ou privés d'une classe sont signalés et ne doivent pas servir à une conclusion globale.

La mémoire maximale enregistrée est le pic du processus depuis son démarrage, donc cumulé entre systèmes ; elle n'est pas une mesure isolée du coût de chaque modèle. Le temps d'ajustement comprend le vectoriseur et le classifieur. Le modèle majoritaire utilise les fréquences du train uniquement. Les modèles TF-IDF ajustent vocabulaire et pondérations sur train uniquement ; les checkpoints ont été rechargés et leurs prédictions vérifiées.

Les prédictions ligne par ligne et les checkpoints contenant des vocabulaires restent sous `outputs/baselines/`, ignoré par Git. Le présent rapport ne contient aucun texte de corpus. Les baselines mots/règles, analyses d'erreurs, modèles neuronaux et résultats finaux restent à réaliser.

La v2 intègre les 12 décisions humaines de même sens et conserve ensemble toutes les familles candidates de la règle de répétitions, y compris celles qui ne traversaient pas les partitions v1. Cette précaution de partitionnement ne transforme pas les candidats non relus en équivalences sémantiques validées. Les variantes raw/norm d'un même run restent comparables, mais les scores TSAC v1 et v2 n'utilisent pas les mêmes exemples de validation : leur différence n'isole pas l'effet de la correction. Les résultats restent provisoires, sans garantie d'absence de toute similarité résiduelle ni comparaison directe à des benchmarks utilisant d'autres partitions.
