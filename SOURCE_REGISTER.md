# TuniScope — Registre des sources

Ce registre consigne l'accès aux corpus, leurs conditions d'utilisation et les vérifications effectuées. Les données brutes restent hors du dépôt Git ; seules la provenance et les décisions sont documentées ici.

| Source | Usage | URL officielle | Accès vérifié | Licence et restrictions | Fichiers et étiquettes vérifiés | Décision |
| --- | --- | --- | --- | --- | --- | --- |
| MADAR Corpus-26 | Variété : MSA / TN / EG | https://camel.abudhabi.nyu.edu/madar-parallel-corpus/ | Oui — confirmation d'inscription reçue le 02/10/2026. | Usage interne de recherche et d'évaluation ; pas de droit de sous-licence, redistribution, cession ou modification du corpus. Archive brute privée et exclue de Git. | Oui — archive contrôlée ; les fichiers TSV MSA, Tunis, Sfax et Le Caire comportent `sentID.BTEC`, `split`, `lang` et `sent`. | Acquisition privée et audit autorisés. |
| TSAC | Sentiment tunisien : POS / NEG | https://github.com/fbougares/TSAC | Oui — dépôt public et archive locale vérifiés le 02/10/2026. | Le dépôt affiche LGPL-3.0. Citer l'article source et confirmer les obligations avant toute publication de données dérivées ; corpus brut exclu de Git. | Oui — `train_pos.txt`, `train_neg.txt`, `test_pos.txt` et `test_neg.txt` sont présents ; archive ZIP intègre. | Acquisition privée et audit autorisés. |

## Acquisition et vérification de MADAR

- Archive officielle : `MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip`.
- URL de téléchargement : <https://camel.abudhabi.nyu.edu/madar-parallel-corpus/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip>.
- Stockage local exclu de Git : `data/raw/madar/`.
- Copie extraite sans modification, exclue de Git : `data/raw/madar/original/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/`.
- SHA-256 de l'archive : `fc9c34638a9c8f8d5266d6431966b6ed933ad07b1e987514fa75f0b5eaf50a6a`.
- Contenu et intégrité vérifiés le 02/10/2026 : fichiers `MADAR.corpus.{MSA,Tunis,Sfax,Cairo}.tsv` présents ; aucune erreur dans le ZIP.
- Nombre de lignes, en-tête compris : MSA 12 001 ; Tunis 12 001 ; Sfax 2 001 ; Le Caire 12 001.
- Règle de partitionnement : utiliser `sentID.BTEC` pour maintenir ensemble les traductions parallèles. Les 2 000 phrases de Sfax définissent le sous-ensemble commun aux quatre variétés ; les lignes supplémentaires de Corpus-6 sont hors de cette première comparaison.

## Décision à l'issue de S1

- Variété : MADAR Corpus-26 ; accès, conditions d'usage, intégrité et fichiers requis vérifiés.
- Sentiment : TSAC ; accès, licence affichée par le dépôt, intégrité et fichiers étiquetés vérifiés.
- Les corpus bruts restent hors de Git ; les manifestes sans texte protégé, empreintes, scripts et documents peuvent être versionnés.

## Acquisition et vérification de TSAC

- Archive du dépôt officiel : `TSAC-master.zip` ; dépôt : <https://github.com/fbougares/TSAC>.
- Stockage local exclu de Git : `data/raw/tsac/`.
- Copie extraite sans modification, exclue de Git : `data/raw/tsac/original/TSAC-master/`.
- SHA-256 de l'archive : `757f2873b516574accb0e5c21d7be7ef86db414f110d5fc109b59988a1e69268`.
- Intégrité vérifiée le 02/10/2026 ; fichiers présents : `LICENSE`, `README.md`, `train_pos.txt`, `train_neg.txt`, `test_pos.txt` et `test_neg.txt`.
- Nombre de lignes : entraînement POS 7 154, NEG 6 515 ; test POS 1 700, NEG 1 700.
- Les fichiers de test officiels seront conservés sans modification ; la validation sera dérivée uniquement des fichiers d'entraînement.
