# Bilan des preuves techniques Silver

Scores de classification sur la validation v2. Les moyennes et écarts-types sont recalculés sur les trois graines pour les campagnes neuronales. Les références déterministes ou à génération fixée sont présentées sans écart-type.

| Système | TSAC macro-F1 | MADAR macro-F1 |
| --- | ---: | ---: |
| Classe majoritaire / brut | 0.343206 | 0.222222 |
| TF-IDF caractères / brut | 0.928898 | 0.949244 |
| TF-IDF caractères / NFC et espaces | 0.928893 | 0.949244 |
| TF-IDF mots / brut | 0.872023 | 0.917075 |
| TF-IDF mots / NFC et espaces | 0.872023 | 0.917075 |
| TF-IDF mots + caractères / brut | 0.923817 | 0.938384 |
| TF-IDF mots + caractères / NFC et espaces | 0.923817 | 0.938384 |
| Vote lexical | 0.720134 | 0.882974 |
| Vote lexical avec négation | 0.718898 | — |
| BiGRU | 0.853201 ± 0.005281 | 0.899806 ± 0.004371 |
| CAMeLBERT gelé, moyenne des vecteurs | 0.844608 ± 0.003337 | 0.921926 ± 0.003579 |
| CAMeLBERT confirmé | 0.930834 ± 0.002308 | 0.953482 ± 0.001235 |
| Qwen base / zero-shot | 0.344134 | 0.000000 |
| Qwen base / few-shot | 0.343206 | 0.183622 |
| Qwen LoRA / zero-shot | 0.706522 ± 0.013757 | 0.732088 ± 0.098624 |
| Qwen LoRA / few-shot | 0.626637 ± 0.064122 | 0.197409 ± 0.036520 |

La référence statistique de modèle de langue est évaluée séparément par perplexité.

| Corpus | Ordre du modèle caractères | Perplexité validation |
| --- | ---: | ---: |
| TSAC | 1 | 41.790861 |
| TSAC | 3 | 11.128590 |
| MADAR | 1 | 23.802216 |
| MADAR | 3 | 9.222734 |

La comparaison native CLS gelé/adapté de la campagne initiale est conservée séparément de la confirmation finale à taux sélectionnés. La référence gelée avec moyenne des vecteurs constitue une autre représentation.

Les matrices LoRA permettent de recalculer les scores et leur dispersion. Les fichiers de prédictions individuelles CUDA et le résumé original complet de campagne sont absents des archives reçues. Leur absence reste déclarée.

Le RAG dispose de résultats de validation et de test documentaire. Ses références de pertinence restent proposées par IA. L’exigence de double revue humaine et l’acceptation d’une adaptation au travail individuel sont ouvertes.

Les preuves d’entraînement, les artefacts locaux et les preuves de rechargement enregistrées sont contrôlés. Cette consolidation n’exécute aucune nouvelle inférence. Une sauvegarde sur support indépendant et une relance par un autre opérateur restent à confirmer. Le test officiel de classification reste réservé.

[Scores, ablations et empreintes](summary.json)
