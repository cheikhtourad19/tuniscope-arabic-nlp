# Référence Qwen3-0.6B sur la validation v2

Inférence avec consignes v2 et décodage glouton. Les métriques couvrent toutes les lignes. Une sortie invalide compte comme une erreur. Le macro-F1 est la moyenne des F1 des labels réels.

| Corpus | Mode | Lignes | Sorties valides | Exactitude | Macro-F1 | Durée de génération |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | zero_shot | 1774 | 1758/1774 (99.10%) | 0.5209 | 0.3441 | 76.15 s |
| TSAC | few_shot | 1774 | 1774/1774 (100.00%) | 0.5225 | 0.3432 | 114.32 s |
| MADAR | zero_shot | 800 | 0/800 (0.00%) | 0.0000 | 0.0000 | 170.90 s |
| MADAR | few_shot | 800 | 800/800 (100.00%) | 0.2800 | 0.1836 | 70.06 s |

**5,148 prédictions**. Exécution sur mps en torch.float32, avec des lots de 16 et un tri par longueur des prompts. Durée de la session mesurée à **436.61 s**.

Le JSON contient les matrices avec une colonne `INVALID`, les métriques par classe, les groupes d’écriture et les empreintes de vérification.
