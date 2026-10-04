# TuniScope — Protocole des données

**Livrable :** L02, protocole préalable à l'audit et à l'apprentissage.

**Version :** 0.1, 3 octobre 2026.

**État :** l'intégrité et la structure des fichiers ont été vérifiées dans l'[audit initial](03_data_audit_structure.md) ; les contrôles de doublons, les cartes de données remplies et les manifestes restent à produire.

## 1. Objectif et sources

Le protocole décrit la provenance, les étiquettes, le schéma commun et le partitionnement des deux tâches. Il fixe les règles de contrôle avant tout entraînement et distingue les effectifs bruts vérifiés des effectifs exploitables, encore inconnus.

| Source | Tâche | Unité et étiquettes retenues | Usage prévu |
| --- | --- | --- | --- |
| MADAR v1.1, partie Corpus-26 | Identification de variété | Une traduction ; MSA → `MSA`, Tunis ou Sfax → `TN`, Le Caire → `EG`. | Sous-ensemble parallèle commun aux quatre variétés. |
| TSAC | Sentiment tunisien | Un commentaire ; fichiers `*_pos.txt` → `POS`, fichiers `*_neg.txt` → `NEG`. | Entraînement sur les fichiers `train_*` ; test officiel sur les fichiers `test_*`. |

Une observation n'obtient que l'étiquette justifiée par sa source. Les commentaires TSAC ne sont pas automatiquement ajoutés à l'entraînement de la tâche dialectale. Les archives originales et leurs copies extraites sont conservées hors du dépôt. Le registre [SOURCE_REGISTER.md](../SOURCE_REGISTER.md) documente les versions, empreintes SHA-256 et conditions d'accès.

## 2. Schéma commun des observations

| Champ | Définition |
| --- | --- |
| `id` | Identifiant unique, stable et dérivé de la source et de sa position. |
| `text_raw` | Texte source en UTF-8, conservé sans modification. |
| `text_norm` | Texte issu d'un prétraitement versionné ; identique à `text_raw` avant traitement. |
| `task` | `dialect` ou `sentiment`. |
| `label` | `MSA`/`TN`/`EG` ou `POS`/`NEG`, selon la tâche ; aucune étiquette manquante n'est inventée. |
| `script` | `AR`, `LATIN`, `MIXED` ou `UNKNOWN`, selon une règle de détection documentée. |
| `languages` | Langues identifiées avec justification ; valeur inconnue admise. L'écriture latine n'implique pas une langue donnée. |
| `source` | Nom et version du corpus. |
| `source_id` | MADAR : `sentID.BTEC` ; TSAC : nom du fichier et numéro de ligne d'origine. |
| `group_id` | Famille à maintenir dans une même partition : phrase source MADAR ou groupe de doublons/variantes identifiable dans TSAC. |
| `split` | `train`, `validation`, `test` ou `challenge`. |
| `annotation_status` | Provenance de l'étiquette : `original`, `double_annotated`, `adjudicated`, `ambiguous` ou `synthetic`. |
| `license` | Référence aux conditions d'utilisation de la source. |
| `normalization_version` | Identifiant du prétraitement ; `raw-v0` tant qu'aucune transformation n'est appliquée. |

La ville MADAR et le nom de la partition d'origine seront conservés en colonnes supplémentaires. Aucun texte brut protégé ne sera intégré aux manifestes diffusables.

## 3. Partitionnement de MADAR

L'audit structurel des fichiers locaux a confirmé que les 2 000 identifiants `sentID.BTEC` de Sfax existent également dans les fichiers MSA, Tunis et Le Caire. Pour ces identifiants, les quatre fichiers portent la même étiquette de partition officielle. Le sous-ensemble initial comporte donc 8 000 lignes avant nettoyage : 2 000 `MSA`, 4 000 `TN` et 2 000 `EG`.

| Étiquette officielle de la partie Corpus-26 | Phrases sources | Partition du projet | Lignes des quatre variétés avant nettoyage |
| --- | ---: | --- | ---: |
| `corpus-6-test-corpus-26-train` | 1 600 | `train` | 6 400 |
| `corpus-6-test-corpus-26-dev` | 200 | `validation` | 800 |
| `corpus-6-test-corpus-26-test` | 200 | `test` | 800 |

La partition officielle est retenue parce que les traductions d'une même phrase y sont déjà regroupées. Les 10 000 lignes supplémentaires de Corpus-6 présentes pour MSA, Tunis et Le Caire sont exclues de cette première comparaison. Leur ajout demanderait une expérience séparée et un nouveau contrôle des groupes. Le regroupement de Tunis et Sfax crée une classe `TN` deux fois plus grande que `MSA` ou `EG` avant nettoyage ; l'évaluation publiera donc les résultats par classe. Toute pondération sera calculée sur `train` seulement.

Avant de figer le manifeste, vérifier l'unicité de chaque couple (`sentID.BTEC`, ville), l'absence d'identifiant source dans plusieurs partitions et le nombre final d'exemples par classe après contrôles.

## 4. Partitionnement de TSAC

Les fichiers `train_pos.txt` et `train_neg.txt` constituent le seul réservoir pour l'entraînement et la validation. Les fichiers `test_pos.txt` et `test_neg.txt` conservent leur statut de test officiel. Les comptages locaux portent actuellement sur les lignes de fichiers : 7 154 POS et 6 515 NEG pour `train`, puis 1 700 POS et 1 700 NEG pour `test`. Les nombres d'exemples utilisables seront établis après recherche des lignes vides, doublons et conflits.

Une validation stratifiée sera produite à partir des seuls fichiers d'entraînement : les proportions POS/NEG seront approximativement préservées et les variantes identifiées resteront dans le même groupe. La proportion de validation, la graine aléatoire et la règle de groupement seront arrêtées après l'audit, puis inscrites dans le manifeste. Aucun choix ne sera fondé sur les performances du test.

Les chevauchements éventuels entre `train_*` et `test_*` seront comptés. Le test officiel restera intact ; le traitement des copies présentes dans l'entraînement sera documenté. Si un jeu décontaminé est construit, son résultat sera distingué du résultat sur le test officiel.

## 5. Contrôles de qualité et normalisation

L'audit doit produire, par fichier et partition, les empreintes SHA-256, effectifs bruts et non vides, effectifs par classe et écriture, champs manquants, identifiants répétés, doublons textuels exacts, conflits d'étiquettes et candidats quasi doublons. Les candidats proches seront revus avant décision. Les suppressions seront comptées et justifiées. Si TSAC ne fournit pas d'identifiant de conversation ou d'auteur exploitable, l'impossibilité de vérifier l'indépendance de ces groupes sera déclarée.

La normalisation préservera `text_raw`. Une version ultérieure pourra harmoniser Unicode et espaces ou remplacer URL et mentions par des marqueurs stables. La négation, les chiffres de l'Arabizi, les mots français/anglais et les émojis porteurs de sentiment devront être conservés ou faire l'objet d'une ablation explicite. Les règles et leurs versions seront apprises ou sélectionnées sur `train`/`validation`, jamais sur `test`.

## 6. Guide d'annotation des futurs jeux locaux

Les étiquettes existantes de MADAR et TSAC gardent le statut `original`. Pour un éventuel jeu local de défi :

- `POS` désigne une appréciation favorable explicite de la cible ; `NEG`, une appréciation défavorable explicite.
- Un avis mixte, sans cible claire, ironique de manière incertaine ou dépourvu d'opinion est marqué `ambiguous` et n'est pas forcé dans les deux classes d'entraînement.
- Une variété `MSA`, `TN` ou `EG` n'est retenue que si des indices linguistiques suffisants permettent d'identifier la variété dominante. Sinon, le cas est `indeterminate` dans le jeu de défi.
- Les jugements initiaux, les désaccords et l'arbitrage sont conservés séparément. L'incertitude d'annotation est distincte de l'abstention du système.

Le cahier des charges prévoit une calibration indépendante de 50 exemples, puis une double annotation d'au moins 150 avis et 100 phrases dialectales. Le projet étant individuel, un second jugement humain compétent devra être obtenu pour déclarer un accord interannotateurs. Une seconde passe du même étudiant ou une proposition d'IA ne sera pas présentée comme une annotation humaine indépendante. Si ce dispositif n'est pas réalisable, la limite et tout écart au cahier seront explicités au responsable pédagogique et dans le rapport ; aucun coefficient kappa ne sera inventé.

## 7. Carte de données et preuves à produire

Une carte distincte sera remplie pour MADAR et TSAC après l'audit. Elle indiquera : origine et version ; usages autorisés et citation ; fichiers et empreintes ; étiquettes et effectifs par classe/partition/écriture ; règles de nettoyage ; doublons et conflits ; groupes et protocole de partitionnement ; provenance des annotations ; limites de domaine et de représentativité.

| Rubrique de la carte | Contenu à renseigner après l'audit |
| --- | --- |
| Identification | Nom, version, date d'acquisition, URL officielle, citation. |
| Droits | Licence, restrictions de réutilisation et de redistribution. |
| Fichiers | Noms, formats, encodage et empreintes SHA-256. |
| Taille et composition | Effectifs bruts et retenus par classe, partition, écriture et source. |
| Préparation | Transformations versionnées, exclusions, doublons et conflits. |
| Partitions | Origine, groupes, graine éventuelle et version du manifeste. |
| Annotation | Provenance des étiquettes, guide, arbitrage et accord si mesuré indépendamment. |
| Limites | Domaine, ancienneté, variété linguistique, métadonnées absentes et biais connus. |

Le passage à l'apprentissage exige un manifeste versionné avec identifiants stables, `group_id`, étiquettes, partitions et empreintes des sources ; un contrôle démontrant l'absence de groupe MADAR entre partitions ; une preuve que le test TSAC officiel est préservé ; et un compte rendu des exclusions. Ces livrables sont **à produire** : le présent document ne certifie pas encore que les données sont prêtes pour l'entraînement.

## Références

- CAMeL Lab, [MADAR Parallel Corpus](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), distribution v1.1 utilisée dans le projet.
- Bougares et collaborateurs, [Tunisian Sentiment Analysis Corpus (TSAC)](https://github.com/fbougares/TSAC).
- Cahier des charges du module, sections 5 à 7 et 17 à 18, document interne du projet.
