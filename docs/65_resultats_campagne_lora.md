# Résultats de la campagne Qwen LoRA sur Kaggle

La campagne a entraîné trois adaptateurs, un par graine, chacun partagé entre TSAC et MADAR. Les graines 13 et 42 ont tourné en parallèle sur deux Tesla T4. La graine 2026 a utilisé la carte libérée par la graine 42. Chaque adaptateur a parcouru les 4 000 exemples SFT deux fois, avec 500 mises à jour. Le commit exécuté est `c7f979f5d7c6d3e1127e2e96025a28ce88b633a5`. [Protocole fixé avant entraînement](64_protocole_campagne_lora.md)

## Macro-F1 sur la validation v2

Chaque comparaison reprend 1 774 exemples TSAC ou 800 exemples MADAR et les consignes de la référence Qwen. La base a été évaluée sur Mac MPS et LoRA sur Kaggle CUDA, tous deux en FP32 avec le même décodage glouton. L’écart-type est calculé sur les trois graines.

| Corpus | Consigne | Qwen de base | LoRA 13 | LoRA 42 | LoRA 2026 | LoRA moyenne ± écart-type | Écart vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSAC | zero-shot | 0,3441 | 0,6995 | 0,7224 | 0,6977 | 0,7065 ± 0,0138 | +0,3624 |
| TSAC | few-shot | 0,3432 | 0,5718 | 0,6971 | 0,6110 | 0,6266 ± 0,0641 | +0,2834 |
| MADAR | zero-shot | 0,0000 | 0,8009 | 0,6191 | 0,7763 | 0,7321 ± 0,0986 | +0,7321 |
| MADAR | few-shot | 0,1836 | 0,1675 | 0,2381 | 0,1866 | 0,1974 ± 0,0365 | +0,0138 |

Les **15 444 réponses LoRA** produites sur les quatre conditions et les trois graines sont toutes des labels valides. Sur MADAR en few-shot, le modèle prédit `EG` sur 760 à 787 des 800 exemples selon la graine, alors que 200 lignes portent ce label. Une réponse peut respecter le format attendu tout en se trompant de variété. La condition zero-shot obtient le meilleur macro-F1 pour les deux tâches sur chacune des trois graines. La dispersion MADAR en zero-shot reste notable, avec un rappel tunisien de 74,0 %, 35,5 % et 66,8 % selon la graine.

Sur la même validation, TF-IDF caractères atteint 0,9289 sur TSAC et 0,9492 sur MADAR. CAMeLBERT confirmé atteint en moyenne 0,9308 et 0,9535. Ces références restent devant le meilleur mode LoRA. [Résultats TF-IDF](23_manifestes_v2_et_baselines.md) · [Résultats CAMeLBERT](57_resultats_confirmation_encodeur.md)

## Temps et mémoire

| Graine | GPU | Apprentissage sur deux époques | Génération de validation | Pic CUDA alloué | Pic CUDA réservé |
| --- | --- | ---: | ---: | ---: | ---: |
| 13 | cuda:0 | 12 min 19 s | 3 min 24 s | 6,60 Go | 12,55 Go |
| 42 | cuda:1 | 12 min 12 s | 3 min 24 s | 6,60 Go | 10,59 Go |
| 2026 | cuda:1 | 12 min 05 s | 3 min 22 s | 6,60 Go | 8,78 Go |

La génération de validation cumule les quatre passages. Chaque contrôle court a pris environ 20 secondes de calcul. Le temps de bout en bout de la session est absent des archives reçues.

## Intégrité et livrables

Les trois ZIP ont été contrôlés, avec un plan identique et des empreintes concordantes pour le commit, le modèle de base, les manifestes, le corpus SFT, la configuration et les fichiers de chaque adaptateur. Les matrices et leurs sous-groupes d’écriture reproduisent les métriques agrégées. Les trois adaptateurs se rechargent sur Mac avec les mêmes empreintes de paramètres que sur Kaggle. Chaque adaptateur génère un label valide sur deux exemples inventés. Les poids et archives d’origine restent privés.

La synthèse de campagne a été reconstruite à partir des trois ZIP. Les prédictions individuelles de Kaggle sont absentes de ces archives. Leur alignement ligne par ligne a été contrôlé par le script sur Kaggle et la vérification locale s’appuie sur les matrices, les effectifs par classe et les sous-groupes. Le test officiel reste réservé à l’évaluation finale. [Résultats agrégés](../reports/qwen_lora_cuda_2026-11-07/summary.json)

**L06** est réalisé avec le notebook, la référence zero/few-shot, le SFT, les trois adaptateurs partagés et les contrôles de génération. **L07** et **L12** restent partiels. L’évaluation finale, le bootstrap apparié et la relance indépendante font partie des travaux restants.
