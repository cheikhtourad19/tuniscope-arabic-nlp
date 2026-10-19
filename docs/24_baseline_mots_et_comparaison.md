# TuniScope — Baseline TF-IDF mots et comparaison aux caractères

**Livrables :** L04 (baselines), L07 (comparaison sur validation) et L12 (reproduction), contributions partielles. **Date :** 17 octobre 2026.

**Suite réalisée :** la [combinaison mots + caractères](25_baseline_mots_caracteres_combines.md) a depuis été entraînée et comparée aux deux références. Le présent document conserve le protocole et les résultats de l'étape mots seuls.

## Protocole fixé avant exécution

Nous ajoutons la référence mots 1–2 grammes et régression logistique prévue dans le cahier des charges. Nous comparons ses résultats à ceux de la référence caractères 3–5 déjà exécutée sur les manifestes v2. Les identifiants, labels et groupes de validation doivent être strictement identiques entre les systèmes comparés. Le test officiel reste fermé.

La configuration `configs/baselines_words_v2.json` fixe C=1, solver=lbfgs, max_iter=1000, min_df=2, sublinear_tf=True, normalisation L2 et un thread CPU, sans recherche d'hyperparamètres. Nous conservons les variantes raw-v0 et norm-nfc-space-v1. Le vocabulaire, les fréquences documentaires et les poids du classifieur sont ajustés sur train seulement. La graine 42 fixe les sous-échantillons smoke ; elle ne représente pas trois répétitions finales.

La segmentation utilise la règle Unicode `(?u)\b\w+\b` : suites de caractères alphanumériques et de soulignements, y compris mots d'une lettre et chiffres Arabizi. Il n'y a ni liste de mots vides, ni racinisation, ni découpe morphologique arabe, ni mise en minuscules. Les espaces séparent les mots ; ponctuation et émojis ne sont pas représentés par cette baseline lexicale. La ponctuation peut donc aussi disparaître entre deux éléments d'un bigramme. Les chiffres Arabizi sont conservés, pas translittérés. Cette définition des « mots » est opérationnelle, pas une analyse linguistique parfaite ; les signes diacritiques combinants peuvent aussi produire des découpes indésirables.

Nous mesurons les lignes dont le vecteur final est nul après segmentation et min_df, sur train et validation. Ces lignes sont conservées dans les scores : la régression logistique prédit alors à partir de ses biais appris, sans indice lexical. Nous ne supprimons pas les émojis du corpus source ; nous documentons la limite de cette représentation.

Un essai de 100 lignes par partition et tâche précède l'exécution complète. Les checkpoints, prédictions, matrices et empreintes sont sauvegardés dans un dossier privé nouveau. Le vérificateur contrôle ensuite les métriques et toutes les prédictions après rechargement. Le rapport comparatif vérifie les empreintes des partitions, l'alignement des identifiants/groupes/labels, la même configuration de classifieur et le même environnement verrouillé avant de calculer les écarts mots − caractères.

La comparaison porte sur deux représentations opérationnelles, pas sur une différence de segmentation isolée : étendues de n-grammes, vocabulaires et traitement des symboles diffèrent. Un écart de validation seul ne prouve ni significativité ni supériorité générale. La combinaison mots/caractères et l'analyse qualitative d'erreurs restent des étapes distinctes.

## Exécution et résultats vérifiés

Le protocole et la configuration ci-dessus ont été écrits avant le premier run mots. Le smoke test de six systèmes (majorité et mots brut/normalisé sur deux tâches) a réussi, puis le run complet a été exécuté dans `outputs/baselines/2026-10-17_words_v2_full/`. Les six checkpoints ont été rechargés et leurs prédictions vérifiées, avec recalcul indépendant des métriques. L'environnement est inchangé : Python 3.12.15, scikit-learn 1.9.1, Mac ARM64, CPU, un thread. Les empreintes du code, de la configuration, des données et du verrouillage sont enregistrées. La suite locale comprend **31 tests réussis**.

Le [rapport mots](../reports/baselines_words_v2_2026-10-17/README.md) et le [rapport comparatif généré](../reports/word_vs_character_v2_2026-10-17/README.md) contiennent les résultats vérifiés. En brut, TSAC obtient une macro-F1 de **0,872023** avec les mots contre **0,928898** avec les caractères. MADAR obtient **0,917075** contre **0,949244**. Les mots brut/normalisé donnent exactement les mêmes prédictions dans les deux tâches. Les matrices par classe, slices par écriture et coûts figurent dans les fichiers JSON des rapports ; aucune valeur n'est saisie manuellement dans leurs tableaux.

La baseline mots produit des vecteurs nuls pour 744 textes train et 172 validation TSAC, puis 46 train et 11 validation MADAR. Ces nombres concernent l'absence de toute caractéristique **retenue**, pas nécessairement l'absence de mots : segmentation, termes inconnus et min_df peuvent tous intervenir. Nous ne les assimilons pas à des commentaires tous composés d'émojis. Ils restent présents dans le dénominateur des métriques.

La comparaison appariée révèle, en brut, 41 erreurs TSAC de la référence caractères corrigées par les mots, contre 142 dans l'autre sens. Pour MADAR, ces nombres sont 15 contre 38. Les caractères sont donc meilleurs globalement dans cette configuration, sans être corrects sur tous les cas où les mots le sont. Cette complémentarité motive l'essai séparé d'une combinaison ; elle ne permet pas de présumer son gain.

Les temps d'ajustement complets sont d'environ 0,10–0,11 seconde pour TSAC et 0,12 seconde pour MADAR. Ils concernent de petits modèles classiques CPU et ne préjugent pas du temps nécessaire à un Transformer ou à LoRA. Les vocabulaires mots comptent respectivement 10 296 et 7 070 colonnes. La mémoire mesurée reste un pic cumulatif du processus, pas le coût isolé d'un modèle.

## Reproduction et état des livrables

`scripts/run_baselines.py` reste le point d'entrée commun ; `scripts/baseline_models.py` construit le modèle selon l'analyseur déclaré dans la configuration. Les configurations historiques caractères restent compatibles et le run v2 caractères a été revérifié sans réentraînement ni écrasement.

```sh
sh scripts/project_uv.sh run --locked python -m unittest discover -s tests -v
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --config configs/baselines_words_v2.json --mode smoke --output outputs/baselines/nouveau_mots_smoke
sh scripts/project_uv.sh run --locked python scripts/verify_baseline_run.py outputs/baselines/nouveau_mots_smoke
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --config configs/baselines_words_v2.json --mode full --output outputs/baselines/nouveau_mots_complet
sh scripts/project_uv.sh run --locked python scripts/render_baseline_report.py outputs/baselines/nouveau_mots_complet --report-dir reports/nouveau_mots
sh scripts/project_uv.sh run --locked python scripts/compare_baseline_runs.py outputs/baselines/2026-10-17_grouped_v2_full outputs/baselines/nouveau_mots_complet --report-dir reports/nouvelle_comparaison
```

Les dossiers de run et de comparaison doivent être nouveaux. Les sources, manifestes et labels n'ont pas été changés. Le test officiel n'a pas été chargé. **L04, L07 et L12 restent partiels** : combinaison mots/caractères, règles linguistiques, analyse humaine des erreurs, autres références et évaluation finale restent à réaliser.
