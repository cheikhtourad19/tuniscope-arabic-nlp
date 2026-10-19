# TuniScope — Combinaison des représentations mots et caractères

**Livrables :** L04, L07 et L12, contributions partielles. **Date :** 17 octobre 2026.

## Protocole fixé avant entraînement

La comparaison précédente montre une complémentarité des erreurs entre les mots et les caractères. Nous testons maintenant une combinaison réellement entraînée, sans choisir après coup la bonne prédiction à partir du label de référence. L'hypothèse est qu'un classifieur disposant des deux représentations pourrait exploiter cette complémentarité. Aucun gain n'est présumé.

La configuration `configs/baselines_combined_v2.json` fixe les partitions v2, les deux variantes raw-v0 et norm-nfc-space-v1 et les paramètres de régression logistique inchangés : C=1, lbfgs, max_iter=1000, sans pondération de classe, un thread CPU. Les vocabulaires et IDF de chaque branche sont ajustés sur train seulement. Le test officiel reste fermé et aucun hyperparamètre n'est sélectionné après lecture des nouveaux scores.

Nous juxtaposons deux blocs creux : mots 1–2 grammes et caractères 3–5 grammes. Chaque branche reprend exactement la configuration de sa référence seule, y compris min_df=2, casse préservée et normalisation L2. Les poids des blocs sont fixés à 1:1. Une normalisation L2 globale suit leur concaténation avant l'apprentissage d'une seule régression logistique. Aucune recherche de poids ou de C n'est réalisée dans cet essai.

Si les deux blocs d'un texte sont non nuls, leur norme finale vaut 1/√2 chacun ; si un seul est non nul, sa norme finale vaut 1. Si les deux sont nuls, le vecteur reste nul et le commentaire est néanmoins évalué. Ce réglage empêche d'augmenter systématiquement la norme globale en ajoutant un bloc, mais modifie la géométrie et la régularisation effective des coefficients par rapport à une branche seule. La comparaison est donc celle de systèmes configurés explicitement, pas la preuve d'un effet isolé sans changement d'échelle des branches.

Les colonnes de mots et de caractères restent distinctes même si leur graphie coïncide. Le bloc mots ne représente toujours ni ponctuation ni émojis ; le bloc caractères peut représenter leurs suites si elles survivent au vocabulaire train et à min_df. Il n'y a ni correction de l'orthographe, ni analyse morphologique, ni réécriture du corpus.

## Vérifications prévues

Un smoke test sur 100 textes par partition et tâche précède l'exécution complète. Les tests fictifs vérifient l'ajustement train seul des deux vocabulaires, la juxtaposition des dimensions, la normalisation, les vecteurs nuls conservés et la relecture des checkpoints. Le vérificateur du run contrôle aussi les paramètres de chaque branche, les effectifs et les prédictions après rechargement. Les références historiques restent inchangées et relançables.

Le rapport comparatif doit vérifier l'identité des manifestes, des paramètres de chaque branche, du classifieur, de l'environnement verrouillé et des identifiants/groupes/labels de validation avant de calculer les écarts combinaison − mots et combinaison − caractères. Nous rapporterons macro-F1, erreurs appariées, couverture, dimensions et coûts, sans présenter la validation comme un score final de test ou une preuve de significativité.

## Exécution et résultats vérifiés

Le protocole et la configuration ci-dessus ont été enregistrés avant le premier run combiné. Six systèmes ont été vérifiés lors du smoke test, puis six lors du run complet `outputs/baselines/2026-10-17_combined_v2_full/`. Les checkpoints ont été rechargés et toutes les prédictions vérifiées. La suite locale compte **40 tests réussis**. Le run complet utilise Python 3.12.15, scikit-learn 1.9.1, Mac ARM64, un thread CPU et les mêmes données v2 que les références.

Le [rapport du run combiné](../reports/baselines_combined_v2_2026-10-17/README.md) et le [rapport comparatif des trois représentations](../reports/three_representations_v2_2026-10-17/README.md) sont générés depuis les prédictions vérifiées. Les vocabulaires **et** les vecteurs IDF des branches combinées sont identiques à ceux des modèles mots/caractères seuls de même variante : le contrôle porte sur les paramètres appris, pas seulement leurs noms de configuration.

En brut, la combinaison atteint une macro-F1 de **0,923817 pour TSAC** (135 erreurs), contre 0,928898 pour les caractères et 0,872023 pour les mots. Sur MADAR, elle atteint **0,938384** (46 erreurs), contre 0,949244 et 0,917075 respectivement. Les variantes brute et NFC/espaces de la combinaison donnent les mêmes prédictions pour chaque tâche. Les variantes caractères TSAC ne sont pas strictement identiques, mais cela ne change pas la conclusion de ce contraste.

Par rapport aux caractères en brut, la combinaison corrige 21 erreurs TSAC mais en introduit 30, soit neuf erreurs supplémentaires ; sur MADAR, elle en corrige 12 et en introduit 19, soit sept supplémentaires. Elle dépasse les mots seuls, sans dépasser les caractères seuls. Nous conservons les caractères comme référence principale actuelle. Nous n'avons pas poursuivi une recherche de poids ou de C après ces résultats ; cette première configuration ne démontre pas que toute combinaison serait moins bonne.

Les vecteurs combinés sont nuls pour 14 textes train et sept validation TSAC, et aucun texte MADAR. Ces lignes restent évaluées. En brut, les dimensions sont 10 296 + 92 733 = 103 029 colonnes pour TSAC et 7 070 + 53 079 = 60 149 pour MADAR. La variante normalisée présente quelques colonnes caractères différentes ; les dimensions par branche sont consignées dans les résultats.

Les temps d'ajustement complets sont d'environ 0,76/0,69 seconde pour TSAC brut/normalisé et 0,97/0,94 pour MADAR. Ces coûts comprennent la construction des deux représentations et l'ajustement du classifieur. Ils ne sont pas extrapolables au futur fine-tuning neuronal. Le pic mémoire rapporté reste cumulé sur le processus.

## Reproduction et suite

Le point d'entrée reste `scripts/run_baselines.py`. `scripts/baseline_models.py` construit une `FeatureUnion` des deux vectoriseurs, suivie d'une normalisation globale et d'une régression logistique. `scripts/compare_combined_baselines.py` contrôle les trois runs, les branches apprises et les prédictions appariées avant de produire les tableaux.

```sh
sh scripts/project_uv.sh run --locked python -m unittest discover -s tests -v
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --config configs/baselines_combined_v2.json --mode smoke --output outputs/baselines/nouveau_combine_smoke
sh scripts/project_uv.sh run --locked python scripts/verify_baseline_run.py outputs/baselines/nouveau_combine_smoke
sh scripts/project_uv.sh run --locked python scripts/run_baselines.py --config configs/baselines_combined_v2.json --mode full --output outputs/baselines/nouveau_combine_complet
sh scripts/project_uv.sh run --locked python scripts/render_baseline_report.py outputs/baselines/nouveau_combine_complet --report-dir reports/nouveau_combine
sh scripts/project_uv.sh run --locked python scripts/compare_combined_baselines.py outputs/baselines/2026-10-17_grouped_v2_full outputs/baselines/2026-10-17_words_v2_full outputs/baselines/nouveau_combine_complet --report-dir reports/nouvelles_trois_representations
```

Les dossiers doivent être nouveaux. Les sources, labels, manifestes et runs historiques restent inchangés. Aucune dépendance ajoutée et aucune ligne du test officiel chargée. L'ablation exploratoire mots/caractères/combinaison est exécutée ; **L04, L07 et L12 restent partiels**, notamment pour les règles linguistiques, les autres références, l'analyse humaine des erreurs, la reproduction indépendante et l'évaluation finale. La prochaine étape est de comprendre les erreurs et les phénomènes linguistiques avant de passer aux modèles neuronaux.
