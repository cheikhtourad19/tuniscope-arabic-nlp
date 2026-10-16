# TuniScope — Dépistage des similarités dans le train TSAC

**Livrables concernés :** L02 (contrôle des quasi-doublons et futur regroupement des partitions), contribution partielle ; L03 (script reproductible de préparation), dépistage seulement.

**Date :** 16 octobre 2026. **État :** contrôle en lecture seule ; aucune famille de validation, partition ou ligne nettoyée n'a encore été créée.

## Pourquoi ce contrôle est nécessaire

Après l'[audit de l'union des exclusions candidates](07_audit_union_exclusions_tsac.md), le train de travail projeté contient **11 827 lignes et 11 827 formes distinctes** après la clé légère Unicode NFKC/espaces. Même sans doublons exacts, deux commentaires presque identiques pourraient être séparés lors de la création de la validation. Le modèle verrait alors une variante pendant l'apprentissage et une autre pendant la validation, ce qui pourrait rendre cette validation trop facile.

Le but de ce dépistage est donc de repérer des **liens candidats entre lignes du train**. Un lien candidat n'est ni un jugement de même sens, ni une correction de label, ni une raison de supprimer les deux lignes. Pour le futur split, plusieurs formes liées pourraient être placées dans une même famille (*group_id*) sans prétendre qu'elles sont linguistiquement équivalentes.

## Méthode

`python3 scripts/screen_tsac_within_train.py` charge uniquement les références train qui resteraient après les exclusions candidates. Il vérifie l'empreinte des fichiers TSAC contre le relevé privé, puis applique les mêmes seuils de recherche que le [dépistage train–test](05_data_audit_near_duplicate_screen.md) : clé NFKC/espaces, suppression d'un caractère pour les formes de 8 à 19 caractères avec similarité d'au moins 0,90, et trigrammes informatifs plus similarité d'au moins 0,92 à partir de 20 caractères. Les formes de moins de huit caractères font l'objet d'une recherche **séparée** à une insertion, suppression ou substitution : elle est beaucoup moins probante et ne reçoit pas de seuil de sens.

Le script n'examine **aucune ligne du test**. Il affiche seulement des effectifs. Un index privé, `data/interim/tsac_within_train_near_2026-10-16.json`, contient les paires de références `(fichier, ligne)`, les scores et les étiquettes source, sans commentaire de corpus ; il est exclu de Git. Les paires y sont uniques et ne pointent que vers des fichiers train.

## Résultats

| Catégorie de recherche | Paires candidates | Formes train concernées | Paires à étiquettes source opposées |
| --- | ---: | ---: | ---: |
| Filtre principal, longueur ≥ 8 | 203 | 329 | 4 |
| Formes très courtes, longueur < 8 sur au moins un côté | 171 | 226 | 4 |

Si nous reliions **provisoirement tous ces liens candidats** pour le seul partage train/validation, nous obtiendrions **209 familles de plusieurs formes**, couvrant **552 formes distinctes** ; la plus grande famille contient **10 formes**. Sept familles comporteraient à la fois des étiquettes POS et NEG. Les 11 275 autres formes resteraient isolées, soit 11 484 groupes potentiels en tout. Ces nombres décrivent un graphe de candidats, pas des équivalences sémantiques confirmées. Les paires courtes sont particulièrement incertaines : un seul caractère différent peut inverser le sens, et le ratio de similarité peut même être nul pour des symboles d'un caractère.

## Interprétation et prochaine décision

Pour **prévenir une fuite de validation**, placer conservativement tous les voisins candidats dans la même partition est possible : aucune famille candidate ne dépasse dix formes dans cet audit. Cette règle peut regrouper à tort des commentaires différents, surtout très courts ; elle réduirait légèrement le nombre de groupes indépendants disponibles, mais ne supprimerait aucun exemple. Nous devons encore choisir et versionner la règle de groupement, fixer une graine et vérifier que le futur split garde une répartition POS/NEG utilisable. Les sept familles mêlant POS et NEG ne prouvent pas une erreur d'étiquette ; elles demandent au minimum un suivi dans le manifeste.

La recherche reste heuristique : ses seuils peuvent manquer des reformulations plus éloignées. L'indépendance par auteur ou conversation n'est pas vérifiable si ces identifiants ne figurent pas dans TSAC. **Aucun manifeste n'a été produit ; L02 et L03 restent incomplets.**
