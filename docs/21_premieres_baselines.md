# TuniScope — Premières baselines CPU et comparaison brut/normalisé

**Livrables concernés :** L04 (majorité et première baseline apprise), L07 (métriques et prédictions) et L12 (traçabilité), **contributions partielles**. **Date :** 17 octobre 2026. **État :** six résultats de validation exécutés et vérifiés ; aucun résultat de test final.

**Mise à jour :** ce document décrit l'expérience historique sur v1. La [revue des variantes et relance sur manifestes v2](23_manifestes_v2_et_baselines.md) ont depuis été effectuées ; utiliser leur configuration explicite pour les nouvelles comparaisons. Les scores et fichiers v1 restent conservés.

## Méthode exécutée

Nous avons implémenté `scripts/run_baselines.py` avec la configuration versionnée `configs/baselines_initial.json`. Le programme apprend sur `train` et prédit uniquement `validation` ; il ne propose pas d'option d'ouverture du test. Pour chaque corpus, nous avons comparé : (1) la classe majoritaire issue des fréquences train, (2) TF-IDF caractères 3–5 et régression logistique sur `raw-v0`, (3) la même configuration sur `norm-nfc-space-v1`. Le vocabulaire et les poids TF-IDF sont ajustés sur train seulement. Aucun hyperparamètre n'a été sélectionné après consultation des résultats.

Les références majoritaires sont `POS` pour TSAC et `TN` pour MADAR. Le classifieur linéaire apprend des poids sur des suites de caractères : il constitue un premier modèle appris pour le projet, sans modèle préentraîné ni réseau profond. Les deux variantes de texte emploient les mêmes exemples et les mêmes étiquettes. La configuration préenregistrée est respectée et les calculs ont été limités à un thread CPU sur Mac ARM64.

## Runs et preuves

Deux essais techniques sur 100 lignes train et 100 validation par tâche ont précédé l'exécution complète. Le premier a produit des avertissements de métrique pour des sous-groupes d'une seule classe ; le calcul a été explicité à partir de la matrice de confusion à classes fixes, puis un second essai a réussi sans ces avertissements. Ces sous-échantillons conservent les partitions d'origine ; ils ne servent pas à une conclusion scientifique ni à une sélection de paramètres.

Le run complet se trouve dans `outputs/baselines/2026-10-17_initial_v1/`, ignoré par Git. Il contient les prédictions par identifiant, les checkpoints, les paramètres et les empreintes. `scripts/verify_baseline_run.py` recalcule indépendamment macro-F1 et accuracy depuis les matrices de confusion, vérifie l'alignement des identifiants/étiquettes, les effectifs train, les empreintes et les prédictions de tous les checkpoints rechargés. Les classes non prédites restent présentes avec `zero_division=0`.

Le [rapport agrégé généré](../reports/baselines_initial_2026-10-17/README.md) et son [fichier de métriques](../reports/baselines_initial_2026-10-17/metrics.json) contiennent les scores, résultats par classe et écriture, coûts et limites. Les valeurs n'ont pas été saisies manuellement dans ce rapport. Les tests locaux comptent désormais 20 contrôles réussis.

## Interprétation et limites

Les premières baselines TF-IDF dépassent la majorité sur les deux validations. Les versions brute et normalisée donnent exactement les mêmes prédictions : cela soutient une absence de gain **pour cette normalisation et ce classifieur précis**, sans généralisation aux futurs Transformers. Le vectoriseur harmonise lui-même les suites d'espaces, ce qui limite l'effet supplémentaire observable de v1.

Ces résultats sont **provisoires**. Les neuf groupes TSAC qui deviennent identiques après réduction des lettres répétées, signalés dans l'[audit précédent](18_audit_tatweel_repetitions.md), suggèrent des parentés possibles entre train et validation même lorsque nous conservons les répétitions. Cela ne prouve pas leur équivalence sémantique ; il faut examiner ces candidats avant de considérer la validation comme définitivement consolidée. Nous ne modifions pas les partitions d'après les scores de ce premier run. Si une règle de regroupement est révisée sur justification de données, elle sera versionnée et les expériences relancées sur les nouveaux manifestes.

La moyenne/dispersion finale sur trois graines, les comparaisons mots/caractères/combinaison, les règles linguistiques et l'analyse d'erreurs restent à réaliser. Les résultats ne sont pas directement comparables à ceux d'une publication utilisant d'autres splits. La mémoire rapportée est le pic cumulé du processus et non une mesure isolée par modèle.

## Reproduction

Après installation de l'environnement verrouillé et accès aux sources autorisées :

```sh
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --mode smoke --output outputs/baselines/nouveau_smoke
sh scripts/project_uv.sh run --locked python scripts/verify_baseline_run.py outputs/baselines/nouveau_smoke
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --mode full --output outputs/baselines/nouveau_run
sh scripts/project_uv.sh run --locked python scripts/verify_baseline_run.py outputs/baselines/nouveau_run
sh scripts/project_uv.sh run --locked python scripts/render_baseline_report.py outputs/baselines/nouveau_run --report-dir reports/nouveau_run
```

Les dossiers de run doivent être nouveaux ; le générateur de rapport ne peut réécrire que les deux fichiers générés pour le même run. Les audits et décisions humaines restent tracés indépendamment des modèles. **L04, L07, L12 et J2 restent incomplets.**
