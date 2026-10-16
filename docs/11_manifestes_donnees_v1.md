# TuniScope — Manifestes de données v1

**Livrables concernés :** L02 (manifestes, traçabilité, partitions), contribution importante mais incomplète ; L03 (préparation reproductible), contribution partielle.

**Date :** 16 octobre 2026. **État :** manifestes de références créés et contrôlés ; aucune copie textuelle préparée pour l'entraînement ni carte de données finale. Les premières cartes [TSAC](12_carte_donnees_tsac.md) et [MADAR](13_carte_donnees_madar.md) ont depuis été rédigées.

## Ce qu'est un manifeste

Un manifeste est comparable à une table d'index dans une base de données : chaque ligne indique **où retrouver** une observation dans le corpus source, son étiquette d'origine, son groupe et sa partition, sans contenir le commentaire ou la phrase. Nous conservons les textes exclusivement dans les copies locales autorisées. Les quatre fichiers CSV se trouvent dans `data/manifests/v1/` : `tsac.csv`, `tsac_exclusions.csv`, `madar.csv` et `madar_exclusions.csv`. Le fichier `file_hashes.csv` enregistre leurs empreintes SHA-256 et `summary.json` les paramètres principaux.

Les CSV retenus sont des **manifestes de provenance et de partition**, pas encore des tables au schéma complet du modèle : `text_raw`, `text_norm`, `script` et `languages` seront produits dans une future copie de travail. Les listes d'exclusion enregistrent des références et motifs, jamais des textes.

## Règles appliquées

- **TSAC :** nous appliquons le relevé privé v2 des 1 842 exclusions train, puis la règle de groupes `tsac-candidate-components-v1` et la partition `tsac-grouped-validation-v1` (validation 15 %, graine 42). Le test officiel conserve ses 3 400 lignes et ses étiquettes. Aucun fichier source n'est réécrit.
- **MADAR Corpus-26 :** nous conservons les quatre variétés MSA, Tunis, Sfax et Le Caire avec les partitions officielles. Les trois identifiants de phrase source concernés par les quatre collisions textuelles inter-partitions sont écartés du **train entier**, soit 12 lignes ; validation et test officiels restent intacts.

| Corpus et état | Train | Validation | Test | Exclus du train |
| --- | ---: | ---: | ---: | ---: |
| TSAC, lignes | 10 053 | 1 774 | 3 400 | 1 842 |
| MADAR, lignes | 6 388 | 800 | 800 | 12 |

Dans TSAC, le train contient 5 255 POS et 4 798 NEG ; la validation, 927 POS et 847 NEG ; le test officiel, 1 700 de chaque classe. Les 1 842 exclusions concernent 972 lignes POS et 870 NEG, avec plusieurs motifs possibles par ligne. Dans MADAR, le train contient 1 597 MSA, 3 194 TN et 1 597 EG ; chaque partition validation/test contient 200 MSA, 400 TN et 200 EG. Les 12 exclusions correspondent à 3 MSA, 6 TN et 3 EG.

## Contrôles exécutés

`python3 scripts/build_data_manifests.py` produit la version v1 depuis les sources locales et refuse d'écraser un dossier de manifestes existant. `python3 scripts/verify_data_manifests.py` a confirmé les empreintes des quatre CSV, le rattachement de chaque ligne source retenue ou exclue, les effectifs par classe, l'absence de groupe partagé entre partitions, l'absence de recouvrement TSAC exact après la clé Unicode/espaces entre train de travail et test, ainsi que zéro chaîne MADAR exacte traversant les partitions retenues. Pour TSAC, les groupes candidats restent entiers entre train et validation. Les empreintes des fichiers source sont contrôlées avant la construction.

La vérification a réussi : **15 227 références TSAC retenues**, **1 842 exclues**, **7 988 références MADAR retenues** et **12 exclues**. Aucune ligne du test officiel n'est dans les listes d'exclusion. Les fichiers CSV ne comportent pas de colonne de texte. Le regroupement par similarité étant heuristique, cela ne prouve pas l'absence de toute fuite sémantique ; l'indépendance par auteur/conversation TSAC reste invérifiable.

## Ce qui manque encore

Il faut compléter les cartes de données propres à MADAR et TSAC après analyse par écriture/langue, décider et exécuter la normalisation versionnée dans une copie de travail, documenter la limitation de l'annotation individuelle et préparer les commandes complètes d'acquisition/préparation. Les tests de quasi-doublons ne sont pas une garantie absolue. **L02 et L03 ne sont pas encore terminés.**
