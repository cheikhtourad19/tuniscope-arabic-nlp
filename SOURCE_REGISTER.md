# TuniScope — Registre des sources

Ce registre consigne l'accès aux corpus, leurs conditions d'utilisation et les vérifications effectuées. Les données brutes restent hors du dépôt Git ; seules la provenance et les décisions sont documentées ici.

| Source | Usage | URL officielle | Accès vérifié | Licence et restrictions | Fichiers et étiquettes vérifiés | Décision |
| --- | --- | --- | --- | --- | --- | --- |
| MADAR Corpus-26 | Variété : MSA / TN / EG | https://camel.abudhabi.nyu.edu/madar-parallel-corpus/ | Oui — nous avons confirmé l'inscription et l'acceptation des conditions le 02/10/2026. | Usage interne de recherche et d'évaluation ; pas de droit de sous-licence, redistribution, cession ou modification du corpus. Archive brute privée et exclue de Git. | Oui — archive contrôlée ; les fichiers TSV MSA, Tunis, Sfax et Le Caire comportent `sentID.BTEC`, `split`, `lang` et `sent`. | Acquisition privée et audit autorisés. |
| TSAC | Sentiment tunisien : POS / NEG | https://github.com/fbougares/TSAC | Oui — dépôt public et archive locale vérifiés le 02/10/2026. | Le dépôt affiche LGPL-3.0. Citer l'article source et confirmer les obligations avant toute publication de données dérivées ; corpus brut exclu de Git. | Oui — `train_pos.txt`, `train_neg.txt`, `test_pos.txt` et `test_neg.txt` sont présents ; archive ZIP intègre. | Acquisition privée et audit autorisés. |
| CAMeLBERT Mix général | Audit WordPiece et référence encodeur gelé | https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix | Oui — fiche, tokenizer et poids officiels vérifiés le 04/11/2026. | Apache-2.0 déclaré dans la fiche du modèle ; code des auteurs sous MIT distinctement. Aucun poids redistribué. | Révision `9be352797bdf28a9ae21e2ae582aaaca7abdb22d`, vocabulaire 30 000 ; checkpoint général `pytorch_model.bin`, 438 873 130 octets, SHA-256 officiel vérifié dans cache ignoré distinct. | Audit et référence gelée MPS vérifiés, longueur128, backbone108490752 paramètres. Seules les têtes sont apprises ; fine-tuning complet restant. Voir docs44/45. |

## Acquisition et vérification de MADAR

- Archive officielle : `MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip`.
- Accès : inscription et acceptation des conditions via la [page officielle](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/) ; le lien reçu par courriel n'est pas publié dans le dépôt.
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

## Inspection de CAMeLBERT Mix — 4 novembre 2026

- Source officielle et révision : [dépôt modèle figé](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix/tree/9be352797bdf28a9ae21e2ae582aaaca7abdb22d).
- Nous avons acquis uniquement `README.md`, `config.json`, `tokenizer_config.json`, `special_tokens_map.json` et `vocab.txt`, soit 316 515 octets. Les URL publiques et empreintes sont conservées dans le manifeste de cache local ; aucun lien privé de corpus n'est ajouté.
- La fiche déclare Apache-2.0 pour le modèle. Le [dépôt de code CAMeLBERT](https://github.com/CAMeL-Lab/CAMeLBERT) affiche MIT séparément. Ces déclarations sont enregistrées à la source, sans annoncer une redistribution des poids.
- Le nombre de paramètres est calculé depuis la configuration : backbone avec pooler 109 081 344 ; variantes de classification 109 082 882/109 083 651 pour deux/trois labels. Ce sont des comptes théoriques ; la vérification d'un checkpoint chargé reste à réaliser.
- L'audit vérifié utilise uniquement train/validation v2 et les fichiers locaux, sans poids, entraînement ou consultation du test pour sélectionner la longueur. Voir `docs/42_protocole_audit_tokenizer_encodeur.md`, `docs/43_resultats_audit_tokenizer_encodeur.md` et le rapport agrégé généré.

Après cet audit, la phase gelée a été lancée : téléchargement explicite des poids de la même révision, SHA-256 LFS `d234e4fe0193e74036b1db72ba9b41a2e4921d8cf0c3c421d1f6a6d8816af729`. Les cinq fichiers du cache tokenizer n'ont pas changé. Le checkpoint est chargé localement sans code distant et avec `weights_only=True` ; le backbone reste gelé. Preuves et limites dans docs44/45 ; aucun poids n'est ajouté au dépôt Git.

## Référence bibliographique TSAC — 5 novembre 2026

Le [papier original WANLP2017](https://aclanthology.org/W17-1307.pdf), §5/tableaux5–6, est retenu pour préparer une reproduction Doc2vec+MLP. Le [dépôt des auteurs](https://github.com/fbougares/TSAC) cite ce travail. Référence consultée et limites de comparabilité consignées dans `docs/48_cadrage_reproduction_bibliographique_tsac.md` ; aucune reproduction ni comparaison directe à nos scores n'est revendiquée.
