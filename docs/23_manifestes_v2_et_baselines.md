# TuniScope — Révision des groupes TSAC et baselines sur manifestes v2

**Date :** 17 octobre 2026. **Livrables :** L02, L04, L07 et L12, contributions partielles. **État :** manifestes v2 produits et contrôlés, smoke test et six résultats complets vérifiés. Les manifestes et expériences v1 sont conservés.

## Justification indépendante des modèles

La [revue humaine](22_revue_variantes_repetitions.md) confirme le même sens pour les 12 paires de variantes à lettres répétées qui traversaient train et validation v1. Elles représentent neuf familles et 21 références source. La sélection repose sur les textes et la règle d'audit, sans utiliser les scores, erreurs ou prédictions des modèles.

Nous devons empêcher que des variantes très proches soient utilisées à la fois pour apprendre et pour mesurer la généralisation. Nous révisons donc les groupes de partitionnement, sans modifier la normalisation donnée aux modèles, sans supprimer de texte et sans changer les étiquettes d'origine.

## Règle de groupement et nouveau tirage

`scripts/build_manifests_v2.py` conserve les composantes de la règle v1 fondée sur 374 liens candidats. Il ajoute les familles dont la clé NFC/espaces devient identique après réduction des suites d'au moins trois lettres arabes ou latines identiques à deux. Ce filtre sert **uniquement au groupement**. Les textes utilisés par les modèles conservent leurs répétitions, émojis, négations et chiffres Arabizi.

Le filtre repère 49 familles répétées : neuf traversaient les partitions v1 et ont été relues ; les 40 autres se situaient déjà dans une seule partition. Nous les gardons aussi indivisibles, car un nouveau tirage pourrait sinon les séparer. Cette précaution reprend la logique conservatrice de v1 ; elle **ne déclare pas les 40 familles restantes équivalentes en sens** et ne corrige aucun conflit de label.

La fermeture transitive des contraintes donne 11 441 groupes, dont 225 groupes multi-lignes couvrant 611 références. Le plus grand compte dix lignes ; huit groupes ont des labels différents. Ces contradictions résiduelles restent explicites, sans correction automatique. Le même algorithme de répartition est réutilisé, avec graine 42 et quotas de validation de 15 % par label. Le changement des groupes modifie l'entrée du tirage : une graine identique ne garantit pas les mêmes affectations.

| Partition TSAC | POS | NEG | Total |
| --- | ---: | ---: | ---: |
| Train v2 | 5 255 | 4 798 | 10 053 |
| Validation v2 | 927 | 847 | 1 774 |
| Test officiel, références copiées | 1 700 | 1 700 | 3 400 |

Les effectifs restent ceux de v1, mais **1 476 lignes passent de train à validation et 1 476 dans l'autre sens**. Les scores TSAC v1 et v2 portent donc sur des exemples différents : leur écart ne mesure pas isolément l'effet de la décontamination. Les 1 842 exclusions précédentes sont inchangées. Le manifeste MADAR et les deux relevés d'exclusions sont copiés à l'identique ; aucune ligne texte du test officiel n'est chargée pour cette révision.

## Conservation et vérification

Les fichiers sans texte sont dans `data/manifests/v2/`. Le manifeste TSAC conserve toutes les références, étiquettes et autres métadonnées de v1 ; seuls les groupes et affectations train/validation peuvent changer. Les lignes de son test sont identiques à celles de v1. `file_hashes.csv` protège les CSV et `summary.json` contient les comptes ainsi que les empreintes de v1 et du registre humain privé.

Les quatre nouveaux tests de `tests/test_manifests_v2.py` vérifient la reconstruction déterministe, l'identité des fichiers copiés, la préservation des références/labels/test, les quotas, l'unicité de partition pour toutes les familles répétées, le regroupement des douze paires humaines et le verrou d'accès au test. La suite complète compte **24 tests réussis**. Aucun groupe v2 ni aucune des 49 familles du filtre ne traverse train/validation. Ce constat concerne les règles contrôlées, pas toute similarité possible.

Le chargeur accepte désormais `load_records("tsac", manifest_version="v2")`. Le défaut reste explicitement v1 pour préserver la reproduction des scripts historiques. Les nouveaux entraînements utilisent `configs/baselines_grouped_v2.json`, qui choisit v2. Il ne faut pas mélanger les versions entre systèmes comparés.

## Relance des références classiques

Nous avons exécuté un smoke test de 100 lignes par partition et par tâche, puis les entraînements complets dans des dossiers nouveaux. Les paramètres des modèles sont inchangés par rapport à la première configuration : majorité issue du train, TF-IDF caractères 3–5 et régression logistique, C=1, un thread CPU, variantes brut et NFC/espaces. Nous n'avons pas sélectionné d'hyperparamètre sur les nouveaux scores.

Le [rapport agrégé généré](../reports/baselines_grouped_v2_2026-10-17/README.md) et les [métriques](../reports/baselines_grouped_v2_2026-10-17/metrics.json) sont la référence de cette relance. Sur TSAC, la macro-F1 est 0,928898 en brut et 0,928893 après normalisation minimale. Deux prédictions diffèrent, avec 126 erreurs dans chaque variante : il n'y a pas de gain établi de cette normalisation ici. MADAR retrouve 0,949244 dans les deux variantes, avec 39 erreurs et les mêmes prédictions. Ces mesures sont des résultats de validation, pas des scores de test ni des preuves finales sur trois graines.

Les six checkpoints ont été rechargés ; les identifiants, labels, empreintes, prédictions et métriques recalculées indépendamment ont été vérifiés pour les runs smoke et complet. Le run historique v1 reste également vérifiable après l'ajout du choix de version. Les checkpoints et prédictions détaillées restent ignorés par Git.

## Reproduction

Avec les corpus autorisés, le registre privé de revue et l'environnement verrouillé disponibles, la construction refuse d'écraser un dossier v2 existant. Pour contrôler les manifestes présents et relancer des modèles dans des dossiers nouveaux :

```sh
sh scripts/project_uv.sh run --locked python -m unittest discover -s tests -v
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --config configs/baselines_grouped_v2.json --mode smoke --output outputs/baselines/nouveau_smoke_v2
sh scripts/project_uv.sh run --locked python scripts/verify_baseline_run.py outputs/baselines/nouveau_smoke_v2
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --config configs/baselines_grouped_v2.json --mode full --output outputs/baselines/nouveau_complet_v2
sh scripts/project_uv.sh run --locked python scripts/render_baseline_report.py outputs/baselines/nouveau_complet_v2 --report-dir reports/nouveau_complet_v2
```

Le registre privé est nécessaire pour reconstruire les groupes à partir de leurs décisions, mais les manifestes distribués ne publient pas les commentaires. La reproduction par une autre personne, l'exécution Kaggle, les règles linguistiques, TF-IDF mots/combinaisons, l'analyse détaillée des erreurs et le protocole d'évaluation final restent à réaliser. **Cette étape ne clôture ni L02, ni L04, ni L07, ni L12.**
