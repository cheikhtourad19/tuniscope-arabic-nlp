# TuniScope — Audit initial des sources : intégrité et structure

**Livrables concernés :** L02 (preuves de qualité des données) et L03 (script reproductible), contributions partielles.

**Date :** 4 octobre 2026. **Périmètre :** étapes 1 et 2 de l'audit, sans nettoyage ni création de nouvelles partitions.

## Méthode

La commande `python3 scripts/audit_structure.py` utilise uniquement la bibliothèque standard de Python. Elle calcule les empreintes SHA-256, teste l'intégrité des ZIP, compare chaque fichier extrait à son entrée d'archive, décode les huit fichiers utiles en UTF-8 strict et compte les lignes et champs. Elle n'écrit ni ne modifie les corpus ; sa sortie contient uniquement des noms de fichiers, des empreintes, des valeurs de métadonnées et des effectifs, jamais les phrases ou commentaires.

Contrôles indépendants : `shasum -a 256` pour les deux archives, `unzip -tqq` pour leur intégrité, `wc -l` pour les huit fichiers utiles et `awk` pour les lignes vides de TSAC et le nombre de colonnes MADAR. Les résultats concordent avec le script.

## 1. Intégrité des archives et des copies extraites

| Source | SHA-256 de l'archive conforme au registre | Test ZIP | Entrées de fichiers identiques après extraction | Fichiers manquants ou supplémentaires |
| --- | --- | --- | ---: | ---: |
| MADAR v1.1 | Oui | Réussi | 34 | 0 |
| TSAC | Oui | Réussi | 6 | 0 |

Deux entrées MADAR correspondent à des fichiers macOS `Icon` vides : leur nom se termine par un retour chariot dans le ZIP, caractère supprimé lors de l'extraction. Le script reconnaît uniquement ce cas précis et confirme que les fichiers extraits sont également vides. Les fichiers de corpus sont, eux, identiques octet par octet à leurs entrées dans l'archive. Les copies originales restent dans `data/raw/`, exclu de Git.

## 2. Structure de MADAR

Les quatre TSV sélectionnés possèdent exactement l'en-tête `sentID.BTEC`, `split`, `lang`, `sent`. Le décodage UTF-8 est valide pour toutes les lignes ; aucune ligne de données n'est vide, mal formée ou privée d'identifiant, de partition, de langue ou de phrase.

| Ville | Étiquette du projet | Code `lang` observé | Lignes de données | Lignes avec quatre colonnes | Anomalies structurelles |
| --- | --- | --- | ---: | ---: | ---: |
| MSA | MSA | `MSA` | 12 000 | 12 000 | 0 |
| Tunis | TN | `TUN` | 12 000 | 12 000 | 0 |
| Sfax | TN | `SFX` | 2 000 | 2 000 | 0 |
| Le Caire | EG | `CAI` | 12 000 | 12 000 | 0 |

Pour chacune des quatre variétés, les partitions officielles de Corpus-26 contiennent 1 600 lignes `train`, 200 `dev` et 200 `test`. MSA, Tunis et Le Caire ont en outre chacun 9 000 lignes `corpus-6-train` et 1 000 lignes `corpus-6-dev`, hors du sous-ensemble initial retenu. Ces effectifs sont des lignes brutes, non des exemples définitivement acceptés. La vérification de l'unicité des identifiants et du regroupement des traductions dans une même partition fait partie de la prochaine passe.

## 3. Structure de TSAC

Les quatre fichiers se décodent entièrement en UTF-8. Chaque ligne non vide constitue un commentaire candidat ; l'étiquette provient du nom du fichier, non d'une colonne interne.

| Fichier | Partition officielle | Étiquette | Lignes physiques | Lignes vides | Lignes non vides |
| --- | --- | --- | ---: | ---: | ---: |
| `train_pos.txt` | train | POS | 7 154 | 2 | 7 152 |
| `train_neg.txt` | train | NEG | 6 515 | 2 | 6 513 |
| `test_pos.txt` | test | POS | 1 700 | 0 | 1 700 |
| `test_neg.txt` | test | NEG | 1 700 | 0 | 1 700 |

Le train officiel contient donc 13 665 lignes non vides et le test officiel 3 400. Les quatre lignes vides se trouvent uniquement dans le train. Elles sont signalées, mais aucun fichier original n'a été modifié et aucune suppression n'a encore été décidée. Les doublons, conflits d'étiquettes et recouvrements train–test restent à mesurer avant de calculer l'effectif exploitable.

## 4. Empreintes des huit fichiers utiles

| Fichier | SHA-256 |
| --- | --- |
| `MADAR.corpus.MSA.tsv` | `2b09710ece69651a0bee40f84c26993d017527386d3049a1061ed1158353d283` |
| `MADAR.corpus.Tunis.tsv` | `4567b4622a49d1a8c0a661ac890d2fefa7c6546bf351ded6410b2505f3e0f8a6` |
| `MADAR.corpus.Sfax.tsv` | `d6ea0d9ad1715fcac42f9ef59b20142715764f294ae0056dc285c2ee9f36c21b` |
| `MADAR.corpus.Cairo.tsv` | `288bd22455ff0d5421348bf3a5c9e713b95c2f860bb8ed2eb1908a138c660429` |
| `train_pos.txt` | `1ddedc6fce42e9d3e09a5892088700a1c9065dcd2f1739983babdf964fc21556` |
| `train_neg.txt` | `4babaf50959d556ea879a471acc3f5c0b794c6e606f31da8002633ab8b199668` |
| `test_pos.txt` | `36e64d31103fb770988337b0a3a71340ac2187c568d52e9eec139df4571b7eb1` |
| `test_neg.txt` | `41e8e829d1b74196ba82c03171e793ac641dd3c936de6a662d9fa399d438a3c6` |

## État et suite

Les étapes d'intégrité et de structure sont vérifiées. Elles ne prouvent **pas** encore l'absence de doublons ni de fuite entre partitions. La prochaine passe contrôlera les identifiants MADAR, les doublons exacts et proches de TSAC, les conflits de labels et les recouvrements train–test. Les décisions de nettoyage, les manifestes et les cartes de données remplies viendront ensuite. L02 et L03 restent incomplets.
