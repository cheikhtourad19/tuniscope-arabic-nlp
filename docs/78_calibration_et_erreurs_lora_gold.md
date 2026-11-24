# Calibration de CAMeLBERT et diagnostic de Qwen LoRA

**22 novembre 2026 · L07, L10 et L12.** Les deux analyses portent sur la validation v2. La première utilise les scores de six checkpoints de CAMeLBERT. La seconde recharge l’adaptateur LoRA de la graine 42 pour créer des prédictions individuelles à relire.

## Les scores de confiance de CAMeLBERT

Un modèle est bien calibré si, parmi les textes auxquels il accorde environ 90 % de confiance, il se trompe environ une fois sur dix. Nous testons ici une correction par température. Une température supérieure à 1 rend les scores moins extrêmes tout en préservant l’ordre des classes et donc la prédiction.

Pour chaque tâche, nous divisons la validation en cinq blocs. Les exemples qui partagent un groupe restent ensemble. Pour chaque checkpoint et chaque bloc, nous ajustons la température sur les quatre autres blocs et mesurons le résultat sur celui qui a été laissé de côté. Ainsi, aucun score calibré n’est mesuré sur un exemple utilisé pour ajuster cette température particulière. Les résultats regroupent trois séries de prédictions sur les mêmes avis ou phrases.

| Corpus | Écart moyen confiance–réussite (ECE, 10 intervalles) | Perte logarithmique | Taux de réponses correctes |
| --- | ---: | ---: | ---: |
| TSAC | 4,87 → 1,40 points | 0,262 → 0,199 | 93,10 % avant et après |
| MADAR | 3,39 → 1,13 points | 0,198 → 0,148 | 95,58 % avant et après |

Le résultat indique une amélioration exploratoire des scores de confiance. Les classes prédites restent les mêmes. La validation a déjà servi aux choix des modèles. L’estimation de calibration reste donc à confirmer sur l’évaluation finale. Les seuils d’abstention mesurés précédemment s’appliquent aux scores bruts, sur une échelle différente.

Avec le **seuil brut de 0,90 sur TSAC**, la couverture agrégée atteint **98,39 %** pour les 785 textes en écriture arabe et **91,99 %** pour les 932 textes en écriture latine. Une couverture globale de 95,00 % masque ces 6,40 points d’écart. La catégorie latine ne désigne pas exclusivement l’Arabizi. Les autres catégories ont de petits effectifs. Nous conservons leur détail dans la sortie JSON.

## Où se trompe Qwen avec LoRA

Nous rechargeons l’adaptateur de la graine 42 avec le même modèle de base. La vérification des empreintes précède la génération. Nous sélectionnons **144 exemples par tâche** avec des quotas équilibrés par label, un représentant par groupe et un classement stable des identifiants. Les consignes ne contiennent aucun exemple. Ce diagnostic CPU dure **34,21 secondes**. Il crée de nouvelles prédictions locales avec les poids Kaggle archivés. Les résultats décrivent cet échantillon et cette graine. La campagne CUDA historique reste la référence sur la validation entière.

| Tâche | Échantillon | Erreurs de label | Sorties hors format | Cas relus avec assistance IA |
| --- | ---: | ---: | ---: | ---: |
| TSAC · sentiment | 144 | 46 | 0 | 30 |
| MADAR · dialecte | 144 | 48 | 0 | 30 |

Sur TSAC, **34 des 72 avis positifs** reçoivent NEG, contre 12 avis négatifs classés POS. Dans les 30 cas relus, les hypothèses incluent l’Arabizi, les expressions locales, l’ambiguïté et la portée de la négation. Sur MADAR, **29 des 48 phrases tunisiennes** reçoivent EG. Dans le lot relu, les formulations communes et les indices de vocabulaire dialectal reviennent régulièrement.

Les 60 cas ont des groupes distincts dans leur tâche. Pour chaque cas, une fiche privée conserve le texte, le label source, la prédiction, la catégorie et une hypothèse. Codex est l’auteur déclaré des lectures. Une hypothèse relie le contenu du texte à une difficulté possible. Elle ne démontre pas le mécanisme interne de l’erreur. Les catégories décrivent les cas sélectionnés et ne constituent pas une estimation des causes sur le corpus entier. La revue humaine indépendante reste ouverte.

### Comparaison appariée avec Qwen de base

Nous exécutons aussi Qwen de base sur les **mêmes 144 textes par tâche**, dans le même ordre, avec les mêmes consignes et le même décodage CPU en FP32. Nous calculons 1 000 écarts par rééchantillonnage apparié, avec les groupes comme unités. Chaque groupe du sous-échantillon contient un seul texte. Cette version pilote isole l’effet de l’adaptateur pour une graine en zero-shot.

| Corpus | Macro-F1 de base | Macro-F1 avec LoRA | Écart | Intervalle à 95 % |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 33,80 % | 67,29 % | +33,49 pts | [+24,05 à +42,38] pts |
| MADAR | 0,00 % | 65,10 % | +65,10 pts | [+56,80 à +72,59] pts |

Sur MADAR, les **144 sorties de la base** sont hors format selon le parseur strict. L’adaptateur respecte le format pour les 144 textes et fait encore 48 erreurs de label. Cette distinction compte dans l’interprétation du gain. Les intervalles décrivent cette sélection équilibrée. Le contraste principal reste à évaluer sur le test officiel, sur trois graines et dans les deux conditions de consigne prévues.

## Preuves et reproduction

- [Résultats de calibration](../reports/gold_diagnostics_2026-11-22/calibration.json) et [graphique de confiance](../reports/gold_diagnostics_2026-11-22/figures/calibration_camelbert.png).
- [Diagnostic Qwen LoRA](../reports/gold_diagnostics_2026-11-22/diagnostic.json), [synthèse des 60 cas](../reports/gold_diagnostics_2026-11-22/review_summary.json) et [matrices](../reports/gold_diagnostics_2026-11-22/figures/erreurs_qwen_lora.png).
- [Comparatif apparié Qwen avant et après LoRA](../reports/gold_diagnostics_2026-11-22/paired_qwen.json). Les 288 inférences de base, avec chargement et calcul des intervalles, ont pris **136,14 secondes**.
- Prédictions et fiches de revue privées sous `outputs/gold_lora_diagnostic_2026-11-22/`. Les empreintes lient les fiches, les prédictions et les agrégats.

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/run_gold_calibration.py --output outputs/gold_diagnostics_reproduction/calibration.json
sh scripts/lora_uv.sh run --locked --offline python scripts/run_gold_lora_diagnostic.py --private outputs/gold_diagnostics_reproduction/prive --output outputs/gold_diagnostics_reproduction/diagnostic.json
sh scripts/lora_uv.sh run --locked --offline python scripts/compare_gold_qwen_diagnostic.py --diagnostic outputs/gold_diagnostics_reproduction/diagnostic.json --private outputs/gold_diagnostics_reproduction/prive --output outputs/gold_diagnostics_reproduction/paired_qwen.json
sh scripts/encoder_uv.sh run --locked --offline python scripts/plot_gold_diagnostics.py outputs/gold_diagnostics_reproduction --output outputs/gold_diagnostics_reproduction/figures
```

La calibration est exploratoire. Les jeux de défi indépendants, l’abstention sur données finales, l’évaluation officielle et le bootstrap principal suivent le protocole final distinct.
