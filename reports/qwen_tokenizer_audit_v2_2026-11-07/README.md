# Audit du tokenizer Qwen3

Train et validation v2, texte brut.

| Corpus | Partition | Lignes | Texte seul p99 / max | Zero-shot p99 / max | Few-shot p99 / max |
| --- | --- | ---: | ---: | ---: | ---: |
| TSAC | train | 10053 | 92 / 376 | 142 / 426 | 187 / 471 |
| TSAC | validation | 1774 | 95 / 255 | 145 / 305 | 190 / 350 |
| MADAR | train | 6388 | 45 / 76 | 104 / 136 | 200 / 232 |
| MADAR | validation | 800 | 42 / 69 | 101 / 128 | 197 / 224 |

| Corpus | Partition | Écriture | Lignes | Tokens par champ | Au moins 100 lignes |
| --- | --- | --- | ---: | ---: | --- |
| TSAC | train | AR | 4426 | 2.40 | Oui |
| TSAC | train | LATIN | 5355 | 2.75 | Oui |
| TSAC | train | MIXED | 259 | 2.63 | Oui |
| TSAC | train | UNKNOWN | 13 | 3.50 | Non |
| TSAC | validation | AR | 785 | 2.37 | Oui |
| TSAC | validation | LATIN | 932 | 2.76 | Oui |
| TSAC | validation | MIXED | 46 | 2.82 | Non |
| TSAC | validation | UNKNOWN | 11 | 4.91 | Non |
| MADAR | train | AR | 6387 | 2.27 | Oui |
| MADAR | train | MIXED | 1 | 1.88 | Non |
| MADAR | validation | AR | 800 | 2.27 | Oui |

Budget retenu sur le train **512** tokens au total, avec **480** tokens d’entrée et **32** en réserve.
Les dépassements par budget, les longueurs par écriture et les sondes inventées sont consignés dans `summary.json`.
