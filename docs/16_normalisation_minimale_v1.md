# TuniScope — Normalisation minimale `norm-nfc-space-v1`

**Livrables concernés :** L02 (protocole et contrôle des données), contribution partielle ; L03 (préparation reproductible), contribution partielle. **Phase :** début de S3, traitement de la langue. **Date :** 16 octobre 2026.

## Choix de la règle

Nous avons défini une première normalisation volontairement minimale, appliquée **en mémoire** aux lignes train et validation déjà sélectionnées par les manifestes v1. Elle effectue seulement deux opérations, dans cet ordre : (1) Unicode **NFC**, qui harmonise les représentations canoniquement équivalentes, puis (2) remplacement de chaque suite d'espaces Unicode par un seul espace ordinaire, avec retrait des espaces en début et fin. Le code est dans `scripts/normalize_text.py` ; l'audit agrégé est dans `scripts/audit_normalization.py`.

Nous conservons toujours `text_raw` inchangé. `text_norm` reçoit le résultat et `normalization_version` vaut `norm-nfc-space-v1`. Cette traçabilité rend la transformation réversible **au niveau de l'enregistrement** : pour retrouver le texte initial, nous lisons `text_raw`, sans prétendre inverser mathématiquement un effacement d'espaces. Le programme crée de nouveaux dictionnaires en RAM et ne modifie ni les lignes originales ni les manifestes.

Nous ne supprimons pas les diacritiques, le tatweel, les répétitions, la ponctuation, les émojis, les chiffres Arabizi, les mots français/anglais ou les marqueurs de négation. Nous ne remplaçons pas encore URL et mentions par des marqueurs : cette décision demande un protocole distinct, notamment pour préserver la structure du sentiment. NFC n'est pas NFKC ; la clé NFKC/espaces de l'audit des doublons n'est pas utilisée comme normalisation de modèle.

## Résultats vérifiés sans ouvrir le test

| Corpus et partition | Lignes | `text_norm` différent de `text_raw` | Changement NFC | Changement d'espaces |
| --- | ---: | ---: | ---: | ---: |
| TSAC train | 10 053 | 874 | 1 | 873 |
| TSAC validation | 1 774 | 187 | 2 | 186 |
| MADAR train | 6 388 | 18 | 1 | 17 |
| MADAR validation | 800 | 3 | 0 | 3 |

Les colonnes « NFC » et « espaces » peuvent concerner une même ligne ; elles ne s'additionnent donc pas nécessairement au total changé. Les **11 tests locaux** passent. L'audit vérifie que `text_raw` est intact et qu'aucune forme `text_norm` exactement identique ne traverse **train et validation** dans chacun des deux corpus. TSAC n'a aucun groupe de doublons exacts restant au sein de ces deux partitions après cette règle. MADAR présente **82 groupes de formes normalisées répétées à l'intérieur d'une partition**, dont **13 groupes comportent plusieurs labels de variété**. Cela peut venir de traductions communes à plusieurs villes et ne prouve pas une erreur d'étiquetage ; nous ne les supprimons ni ne les réétiquetons automatiquement.

## Portée et suite

Ce contrôle n'a **pas ouvert le test officiel**. Il ne certifie donc pas encore l'absence de collision supplémentaire avec le test après `norm-nfc-space-v1`. Cette vérification devra être effectuée de manière préspécifiée et traçable avant toute évaluation finale, sans choisir la règle en fonction des performances du test. L'absence de doublon exact ne garantit pas l'absence de paraphrases ou de relations par auteur/conversation.

La prochaine décision est de déterminer, sur train/validation et exemples synthétiques, si nous ajoutons des remplacements réversibles pour URL et mentions, ou si cette version minimale suffit pour la première baseline. Nous comparerons ensuite `raw-v0` et `norm-nfc-space-v1` à protocole d'évaluation constant. **L02, L03 et le traitement linguistique S3 restent incomplets.**
