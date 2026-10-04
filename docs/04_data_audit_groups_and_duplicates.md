# TuniScope — Audit des groupes et des doublons exacts

**Livrables concernés :** L02 (contrôles de qualité et prévention des fuites) et L03 (script reproductible), contributions partielles.

**Date :** 4 octobre 2026. **Périmètre :** lecture seule des fichiers originaux ; aucune exclusion, correction de label ni nouvelle partition.

## 1. Structure des deux sources

MADAR n'est pas une base relationnelle : chaque variété dispose d'un fichier TSV, dont chaque ligne représente une traduction. Les quatre colonnes sont `sentID.BTEC` (identifiant de la phrase source), `split` (partition officielle), `lang` (code de variété) et `sent` (texte). Un même `sentID.BTEC` peut donc apparaître dans quatre fichiers différents. Le nom du fichier permet de distinguer MSA, Tunis, Sfax et Le Caire ; le projet regroupe Tunis et Sfax sous l'étiquette `TN`.

TSAC contient quatre fichiers texte, `train_pos.txt`, `train_neg.txt`, `test_pos.txt` et `test_neg.txt`. Chaque ligne non vide est un commentaire. Le fichier, et non une colonne, indique à la fois la partition et l'étiquette. TSAC ne fournit pas d'identifiant de ligne dans ces fichiers ; le couple (nom de fichier, numéro de ligne d'origine) sert de référence stable pour l'audit.

Le schéma commun défini dans le [protocole](02_data_protocol.md) est **prévu**, pas encore matérialisé en données transformées. Aucun texte brut protégé n'est présent dans ce rapport.

## 2. Définitions et méthode

Le script `python3 scripts/audit_duplicates.py` lit les quatre TSV MADAR et les quatre fichiers TSAC, puis affiche uniquement des effectifs agrégés. Un *doublon exact* TSAC signifie que les caractères du commentaire sont identiques après retrait du seul terminateur de ligne ; les espaces internes ou périphériques ne sont pas modifiés. Une vérification supplémentaire avec `strip()` repère les différences limitées aux espaces en début ou fin de ligne, sans constituer une normalisation linguistique. Les lignes vides déjà signalées sont exclues de ces comparaisons.

Un *groupe MADAR* est l'ensemble des traductions partageant le même `sentID.BTEC`. Le contrôle textuel distinct recherche une chaîne `sent` identique dans deux partitions officielles différentes, même si les identifiants source diffèrent. Le script ne publie ni phrases, ni commentaires, ni empreintes individuelles de texte. Un second comptage indépendant avec `awk` a confirmé les effectifs principaux.

## 3. MADAR Corpus-26

| Contrôle | Résultat |
| --- | ---: |
| Identifiants source distincts retenus | 2 000 |
| Groupes avec les quatre variétés présentes | 2 000 |
| Lignes par variété retenue | 2 000 |
| Groupes train / validation / test | 1 600 / 200 / 200 |
| Groupe incomplet, couple (ID, variété) répété ou ID traversant plusieurs partitions | 0 |
| Lignes dont le code `lang` ne correspond pas au fichier | 0 |
| Chaînes de texte strictement identiques traversant plusieurs partitions | 4 groupes, 8 lignes |

Le regroupement **par identifiant source** est donc cohérent pour ce sous-ensemble : aucune traduction d'un même ID n'a été placée dans une autre partition. Toutefois, quatre chaînes identiques apparaissent dans des partitions différentes sous des identifiants distincts. Il s'agit d'un risque de recouvrement textuel à examiner, pas encore d'une décision d'exclusion. Garder un même identifiant ensemble ne garantit pas à lui seul que tous les textes soient uniques.

## 4. TSAC

| Contrôle des commentaires non vides | Résultat |
| --- | ---: |
| Lignes non vides dans les quatre fichiers | 17 065 |
| Textes exacts distincts | 15 291 |
| Groupes de doublons exacts | 846 groupes, contenant 2 620 lignes |
| Lignes en excès si un seul exemplaire était conservé par texte | 1 774 ; **aucune suppression effectuée** |
| Groupes de doublons présents uniquement dans le train / uniquement dans le test / dans les deux | 507 / 23 / 316 |
| Textes exacts partagés entre train et test | 316 groupes : 963 lignes train et 457 lignes test impliquées |
| Textes exacts portant à la fois POS et NEG | 13 groupes, 112 lignes impliquées |
| Conflits POS/NEG présents au sein du train / au sein du test | 9 / 1 groupes |
| Groupes de textes distincts ne différant que par les espaces aux extrémités | 14 |
| Recouvrements train–test après simple retrait des espaces aux extrémités | 319 groupes, contre 316 sans ce retrait |

Les catégories se recouvrent : par exemple, un groupe de textes présent dans le train et le test peut aussi contenir un conflit POS/NEG. Les 1 774 « lignes en excès » sont une **mesure de répétition**, pas un nombre de suppressions recommandées. Les 457 lignes test impliquées représentent environ 13,4 % des 3 400 lignes du test officiel ; ce chiffre ne signifie pas que le test peut être modifié.

Ces recouvrements peuvent rendre un futur score trop optimiste si des copies restent dans l'entraînement. Les conflits d'étiquettes demandent une revue avant toute décision : une même chaîne peut aussi être ambiguë hors contexte. Les fichiers originaux et le test officiel demeurent intacts.

## 5. Décisions non prises et suite

- Examiner de manière privée les quatre chaînes MADAR traversant les partitions, les 13 groupes TSAC à labels contradictoires et les recouvrements train–test ; ne publier aucun commentaire protégé dans Git.
- Rechercher les quasi-doublons TSAC avec une méthode et un seuil documentés, puis vérifier les candidats avant toute exclusion.
- Un [premier dépistage train–test](05_data_audit_near_duplicate_screen.md) a depuis produit des candidats et des index de revue privés ; il ne couvre pas encore les quasi-doublons internes ni leur validation humaine.
- Choisir et tracer une politique de décontamination du train sans modifier le test officiel ; conserver des résultats officiels et, si nécessaire, des résultats décontaminés clairement séparés.
- Après cette revue, établir les effectifs exploitables, les manifestes de partitions et les cartes de données MADAR/TSAC.

Cet audit ne certifie donc pas encore que L02 ou L03 soit terminé, ni que les données soient prêtes pour l'entraînement.
