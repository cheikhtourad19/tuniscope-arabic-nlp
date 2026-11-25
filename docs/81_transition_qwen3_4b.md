# Transition du volet LLM vers Qwen3-4B

Le 23 novembre 2026, nous avons retenu Qwen3-4B pour la suite du projet après la levée du plafond de paramètres pour le volet LLM, selon l’accord de l’encadrant rapporté par l’auteur du projet. Les futures comparaisons principales, y compris pour la présentation Platinum, porteront sur **Qwen3-4B avant et après adaptation**. Les runs et présentations Qwen3-0.6B restent les références historiques de leurs étapes.

Le checkpoint choisi est [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) en mode sans raisonnement, épinglé à [la révision officielle 1cfa9a7](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c). « Avant LoRA » signifie que nous utilisons le checkpoint public sans notre adaptateur. Nous conservons le même checkpoint pour les deux conditions.

## Quantification et entraînement unique

Sur le Mac Apple Silicon de 16 Go, nous visons une base **quantifiée à 4 bits** avec MLX. Les poids seuls représentent environ 8 Go à 16 bits et environ 2 Go dans le calcul théorique à 4 bits. L’occupation réelle est plus élevée, car il faut aussi garder la mémoire nécessaire à l’exécution et stocker les informations de quantification.

L’adaptation utilise **QLoRA**. Elle conserve les poids de base quantifiés et gelés et apprend seulement un petit adaptateur LoRA. La [documentation MLX LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) décrit l’entraînement sur base quantifiée et les options de réduction de mémoire. Le chronométrage sur des séquences fictives donne environ 62 min par époque. Le cas de 384 tokens a atteint un pic de 3,44 Go alloués par MLX. La durée de la campagne complète reste à mesurer.

Nous retenons **une configuration, une graine fixée à 42 et un seul adaptateur partagé** entre le sentiment TSAC et la classification dialectale MADAR. La graine reprend la convention de nos premiers diagnostics. Le protocole utilise 4 000 exemples du train, deux époques et un exemple à la fois en mémoire. Nous accumulons les gradients de 16 exemples avant chaque mise à jour, comme dans le protocole historique. La perte porte sur les tokens de l’étiquette et de la fin de réponse. Le rang 8, la cible Q/V, le taux d’apprentissage de 0,0001 et les paramètres d’AdamW sont fixés avant la campagne.

## Comparaison et preuves

1. Vérifier les poids, le tokenizer et le format de conversation de la révision retenue. Mesurer la mémoire d’exécution.
2. Évaluer Qwen3-4B quantifié sur la validation de TSAC et MADAR en **few-shot**. La consigne inclut un exemple d’entraînement par étiquette, soit deux exemples pour TSAC et trois pour MADAR. Le mode few-shot reprend les exemples déjà fixés pour le 0.6B. Leur ordre et leurs identifiants sont contrôlés avant l’inférence.
3. Entraîner l’adaptateur partagé et évaluer la **même base quantifiée** avec l’adaptateur, sur les mêmes textes, avec le même décodage et les mêmes règles de lecture des labels. Comparer le macro-F1, les erreurs, les sorties hors format et le coût. Une mesure propre à l’Arabizi exige un sous-ensemble identifié comme tel.
4. Intégrer les nouveaux artefacts et résultats au protocole final versionné avant d’ouvrir le test officiel. Le [protocole v1](73_protocole_classification_finale.md) reste vérifiable comme référence du 0.6B.

Le [plan de transition](../configs/qwen3_4b_transition_v1.json) fixe le checkpoint, l’empreinte des trois fichiers de poids, les partitions et le périmètre de comparaison. L’utilitaire de préparation utilise un cache distinct.

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/prepare_qwen4b.py plan
sh scripts/project_uv.sh run --locked --offline python scripts/prepare_qwen4b.py download
sh scripts/project_uv.sh run --locked --offline python scripts/prepare_qwen4b.py verify
```

La commande `download` nécessite l’accès réseau à Hugging Face. En cas d’interruption, une relance réutilise les fichiers déjà reçus et demande au serveur la suite du fichier de poids partiel. Si le serveur refuse la reprise, le fichier partiel est conservé. Chaque fichier de poids complet est contrôlé par SHA-256. La reprise a récupéré les **244,7 Mo** manquants du deuxième fichier. La vérification locale a confirmé les douze fichiers de la révision amont, soit 8 060 925 056 octets.

## Reproduire l’acquisition et l’évaluation sur Mac

La conversion locale utilise l’environnement verrouillé `environments/qwen4b/` et MLX LM 0.32.2. Les matrices de poids compatibles sont quantifiées à 4 bits par groupes de 64 valeurs, avec un facteur d’échelle et un biais par groupe. Le fichier de poids MLX pèse 2 263 022 417 octets. Le reçu de conversion relie les fichiers MLX au reçu de la révision amont. La conversion est à exécuter une seule fois. La génération sauvegarde sa progression par blocs dans un répertoire ignoré par Git. Le rapport vérifie les identifiants et les étiquettes, puis recalcule les métriques à partir des prédictions sauvegardées.

```sh
UV_CACHE_DIR="$PWD/.tools/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" \
  .tools/bootstrap/bin/uv sync --locked --project environments/qwen4b --python 3.12.15
environments/qwen4b/.venv/bin/python scripts/run_qwen4b_few_shot.py convert
environments/qwen4b/.venv/bin/python -u scripts/run_qwen4b_few_shot.py evaluate
sh scripts/project_uv.sh run --locked --offline python scripts/run_qwen4b_few_shot.py report
```

Cette préparation contribue à **L06 — LLM**, **L07 — Évaluation**, **L11 — Présentation** et **L12 — Reproductibilité**.

## Résultats du modèle de base en few-shot

L’évaluation couvre les deux validations v2, avec les textes bruts, le format de conversation Qwen sans raisonnement et un décodage glouton limité à 32 nouveaux tokens. Le parseur attend une étiquette seule. Une sortie hors format compte comme une erreur. Le macro-F1 accorde le même poids à chaque étiquette réelle.

| Corpus | Exemples dans la consigne | Textes de validation | Exactitude | Macro-F1 | Sorties hors format |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 2 | 1 774 | 69,73 % | 67,66 % | 1 |
| MADAR | 3 | 800 | 35,125 % | 30,70 % | 0 |

Sur TSAC, le modèle a identifié 393 des 847 avis négatifs et 844 des 927 avis positifs. Sur MADAR, il a répondu `EG` pour 709 textes sur 800, dont 341 textes tunisiens. Le format est respecté sur MADAR et les prédictions sont fortement concentrées sur la classe égyptienne.

La génération a pris 542,68 s et l’allocateur MLX a indiqué un pic de 4 544 688 214 octets pendant l’inférence. Cette mesure est propre à l’allocateur MLX. Le [rapport agrégé](../reports/qwen3_4b_few_shot_v1/summary.json) conserve les matrices, le rappel par classe et les empreintes. Les scores et les matrices ont été recalculés indépendamment à partir des 2 574 prédictions privées. Le test officiel reste réservé.

## Lancer le QLoRA dans VS Code

Ouvrir le [notebook 11](../notebooks/11_qlora_qwen4b_mac.ipynb), sélectionner **TuniScope (notebooks)** et cliquer sur **Tout exécuter**. Le noyau Jupyter affiche la progression. L’entraînement et la génération fonctionnent dans le profil MLX local. La comparaison des métriques utilise le profil Python historique.

La cellule d’entraînement affiche l’époque, le nombre d’exemples traités, la perte et une estimation du temps restant. Toutes les 15 mises à jour, soit 240 exemples, et à chaque fin d’époque, elle sauvegarde l’adaptateur et l’état d’AdamW. Une relance avec le même dossier reprend au dernier point complet. La cellule suivante recharge l’adaptateur et évalue les mêmes textes de validation avec les mêmes exemples few-shot.

Le dossier `outputs/qwen/qwen3_4b_qlora_full_v1/` contient l’adaptateur `adapters.safetensors`, les journaux et `comparison.json`. Les prédictions individuelles y restent privées. Le code exact de la génération avant adaptation a été conservé avec le run de base dans `outputs/qwen/qwen3_4b_few_shot_v1/generation_source.py`.

## Résultats du run complet, 24 novembre 2026

Le notebook a entraîné un adaptateur partagé sur deux époques de 4 000 textes, pour un total de 500 mises à jour. L’adaptateur de 2 949 120 paramètres a été sauvegardé et rechargé pour évaluer les 2 574 textes de validation. La procédure d’import du rapport recalcule les prédictions agrégées, les matrices et les métriques en alignant les identifiants de validation. Elle contrôle la base de départ, les démonstrations et l’empreinte des poids de l’adaptateur.

| Corpus | Macro-F1 avant | Macro-F1 après | Écart en points | Exactitude après |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 67,66 % | 81,30 % | +13,64 | 81,91 % |
| MADAR | 30,70 % | 71,13 % | +40,43 | 69,00 % |

Le rappel TN reste de 42,75 %. Le modèle prédit EG sur 212 des 400 vrais textes TN, contre 341 avant adaptation. Sur TSAC, le rappel NEG passe de 46,40 % à 66,94 %. Le run complet a pris 1 h 50 min 23 s d’apprentissage selon le journal et environ 8 min 35 s de génération après rechargement. Le pic alloué par MLX pendant l’apprentissage était de 3,45 Go. Les [agrégats d’adaptation](../reports/qwen3_4b_qlora_v1/summary.json) alimentent le [rapport scientifique LaTeX](../rapport/README.md). L’ouverture du test officiel dépend du protocole final révisé.
