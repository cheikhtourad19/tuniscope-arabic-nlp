# TuniScope — Chargement des données de travail en mémoire

**Livrables concernés :** L02 (schéma, provenance et contrôle des partitions), contribution partielle ; L03 (chargement reproductible depuis les sources), contribution partielle.

**Version :** `raw-v0`, 16 octobre 2026. **État :** chargeur vérifié pour train et validation ; aucune normalisation linguistique ni table textuelle sauvegardée.

## Principe

Nous chargeons le [manifeste v1](11_manifestes_donnees_v1.md), qui indique les références des lignes retenues et leur partition, puis nous retrouvons le texte correspondant dans la copie locale autorisée des fichiers originaux. Le résultat est une **liste de lignes structurées en RAM**. Une ligne contient notamment `id`, `text_raw`, `text_norm`, `task`, `label`, `source_id`, `group_id` et `split`. C'est l'équivalent d'une table de travail ; aucune copie de ses textes n'est écrite dans le dépôt.

À cette version, `text_norm` est **strictement égal** à `text_raw` et `normalization_version` vaut `raw-v0` : aucune transformation linguistique n'a encore eu lieu. `script` vaut `UNKNOWN` et `languages` vaut `und` (« indéterminé »), car leur détection n'est pas encore définie. L'étiquette reste celle de la ligne source ; la sélection des représentants en cas de conflit est tracée par le manifeste et le relevé privé, pas par une réécriture du corpus.

Le code de chargement est `scripts/load_working_data.py`. Il n'exige que la bibliothèque standard de Python. Lors de sa première vérification du 16 octobre, pandas n'était pas installé ; `as_pandas(records)` restait une conversion optionnelle. **Mise à jour du 17 octobre :** l'[environnement Python du projet](20_environnement_python.md) inclut désormais pandas 3.0.6, et cette conversion a été vérifiée sur les deux corpus train/validation.

## Protection des partitions et contrôles

L'appel par défaut `load_records("tsac")` ou `load_records("madar")` lit seulement `train` et `validation`. L'accès au test figé exige de demander explicitement `splits=("test",)` **et** `allow_test=True`. Ce mécanisme évite un accès accidentel pendant le développement ; il ne remplace pas la discipline scientifique de réserver le test à l'évaluation finale.

Avant de lire les lignes, le chargeur vérifie l'empreinte SHA-256 du manifeste et des fichiers source requis. Il contrôle ensuite la présence des lignes, leurs métadonnées d'origine, l'unicité des références et des identifiants, et rejette une ligne de travail vide. Il n'affiche aucun texte protégé. Les fichiers originaux restent inchangés.

| Chargement par défaut | Train | Validation | Total en mémoire | Test lu |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 10 053 | 1 774 | 11 827 | Non |
| MADAR Corpus-26 | 6 388 | 800 | 7 188 | Non |

La commande `python3 -B -m unittest discover -s tests -p 'test_*.py' -v` a réussi **4 tests**, dont un suivi des fichiers ouverts confirmant que le chargement TSAC par défaut ne lit que `train_pos.txt` et `train_neg.txt`. Une vérification indépendante des manifestes v1 reste disponible dans `scripts/verify_data_manifests.py`.

## Limites et prochaine étape

Cette table en mémoire est utilisable pour explorer le schéma et les effectifs, mais ce n'est pas encore un dataset final pour apprendre un modèle : l'analyse de l'écriture/langue, la normalisation versionnée et le protocole complet de préparation ne sont pas achevés. Nous devons ensuite mesurer la composition des textes retenus par écriture et longueur, compléter les cartes de données, puis fixer les premières règles de normalisation prudente. **L02 et L03 restent incomplets.**
