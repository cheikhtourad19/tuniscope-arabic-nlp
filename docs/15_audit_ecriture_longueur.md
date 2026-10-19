# TuniScope — Audit de l'écriture et de la longueur des textes retenus

**Livrables concernés :** L02 (composition des données et limites), contribution partielle ; L03 (audit reproductible), contribution partielle. **Date :** 16 octobre 2026.

## Méthode et périmètre

Nous avons chargé en mémoire uniquement les lignes `train` et `validation` des [manifestes v1](11_manifestes_donnees_v1.md), sans ouvrir le test officiel. Le script `python3 scripts/audit_text_composition.py` ne publie que des effectifs et des statistiques agrégées, jamais les textes ni les identifiants de lignes. Sept tests locaux, dont trois sur des exemples synthétiques pour la règle d'écriture, ont réussi.

La règle examine les **lettres Unicode** présentes dans le texte brut : lettres arabes seules → `AR`, lettres latines seules → `LATIN`, les deux → `MIXED`, aucune des deux → `UNKNOWN`. Chiffres, ponctuation et émojis ne suffisent pas à attribuer une écriture. C'est une **détection d'écriture**, pas une identification de langue : un texte en lettres latines peut être de l'Arabizi, du français, de l'anglais ou un mélange. `UNKNOWN` ne signifie pas « inutile » ; les émojis peuvent exprimer un sentiment.

La longueur en caractères correspond au nombre de points de code Unicode (`len` en Python), non au nombre de graphèmes visibles. Les « unités séparées par espaces » sont obtenues par `split()` ; ce ne sont **pas** des sous-mots de tokenizer. Les percentiles P50/P95 indiquent les longueurs au-dessous desquelles se trouvent respectivement au moins 50 %/95 % des lignes, avec la méthode du rang supérieur.

## Effectifs par écriture

| Corpus et partition | Étiquette | `AR` | `LATIN` | `MIXED` | `UNKNOWN` | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TSAC train | POS | 2 006 | 3 109 | 122 | 18 | 5 255 |
| TSAC train | NEG | 2 434 | 2 221 | 140 | 3 | 4 798 |
| TSAC validation | POS | 376 | 525 | 24 | 2 | 927 |
| TSAC validation | NEG | 395 | 432 | 19 | 1 | 847 |
| MADAR train | MSA | 1 596 | 0 | 1 | 0 | 1 597 |
| MADAR train | TN | 3 194 | 0 | 0 | 0 | 3 194 |
| MADAR train | EG | 1 597 | 0 | 0 | 0 | 1 597 |
| MADAR validation | MSA / TN / EG | 800 | 0 | 0 | 0 | 800 |

Dans le train TSAC, nous comptons au total **4 440 `AR`, 5 330 `LATIN`, 262 `MIXED` et 21 `UNKNOWN`**. La validation TSAC compte **771, 957, 43 et 3** lignes dans le même ordre. La proportion d'écriture latine varie aussi selon l'étiquette source ; nous devrons donc présenter les résultats par écriture et ne pas laisser le modèle exploiter cette différence comme unique raccourci. MADAR train est presque entièrement `AR` : une ligne MSA comporte à la fois des lettres arabes et latines selon cette règle ; nous ne déduisons pas sa langue à partir de ce seul comptage.

## Longueur des textes

| Corpus, partition et étiquette | Points de code P50 | Points de code P95 | Maximum | Unités espacées P50 | Unités espacées P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC train POS | 25 | 92 | 427 | 4 | 17 |
| TSAC train NEG | 28 | 139 | 1 024 | 5 | 25 |
| TSAC validation POS | 25 | 90 | 513 | 4 | 16 |
| TSAC validation NEG | 27 | 138 | 728 | 5 | 25 |
| MADAR train MSA | 32 | 70 | 193 | 7 | 15 |
| MADAR train TN | 25 | 58 | 160 | 5 | 11 |
| MADAR train EG | 27 | 61 | 169 | 5 | 12 |
| MADAR validation MSA | 33 | 68 | 164 | 7 | 14 |
| MADAR validation TN | 26 | 57 | 142 | 5 | 11 |
| MADAR validation EG | 27 | 59 | 136 | 6 | 12 |

Les commentaires TSAC négatifs présentent une queue de longueur plus importante que les positifs ; cette observation descriptive ne démontre pas la cause. Elle invite à contrôler la troncature quand nous auditerons les tokenizers. Les nombres de caractères ne déterminent pas directement la longueur maximale en sous-mots d'un modèle.

## Décisions soutenues et limites

Nous conservons pour l'instant toutes les lignes retenues dans les manifestes, y compris les catégories `UNKNOWN`, sans suppression automatique. La future normalisation doit préserver les émojis, les chiffres de l'Arabizi et les mots latins ; une comparaison `raw`/`normalisé` sera nécessaire. La composition par **langue** n'est pas connue : elle exigerait une méthode distincte et une vérification humaine sur échantillon. Le test officiel n'a pas été utilisé pour choisir une règle ou une longueur de modèle. Cet audit complète partiellement les cartes de données et prépare la phase de normalisation ; **L02 et L03 restent incomplets.**
