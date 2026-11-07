# Audit Qwen et budget des consignes

**7 novembre 2026.** Les mesures couvrent **19 015 lignes** en texte brut sur train et validation v2. Nous avons appliqué le tokenizer officiel de Qwen3-0.6B, révision `c1899de289a04d12100db370d81485cdf75e47ca`, à chaque texte seul puis à ses deux prompts complets. Les tableaux détaillés, empreintes et sondes inventées sont produits par le script d’audit. [Rapport agrégé](../reports/qwen_tokenizer_audit_v2_2026-11-07/README.md)

Le premier format mesuré contenait 501 tokens sur le plus long prompt few-shot TSAC du train. Trois lignes dépassaient les 480 tokens d’entrée disponibles dans le budget de 512 avec réserve de 32. Nous avons raccourci uniquement les instructions françaises, de 30 tokens pour TSAC et d’environ 31 pour MADAR sur les prompts observés. Les empreintes des lignes et des démonstrations concordent entre les deux audits. Le choix de formulation repose sur les longueurs mesurées. Le protocole initial est conservé avec son rapport et ses sources dans `outputs/qwen/2026-11-07_initial_prompt_audit/`. [Protocole initial](59_protocole_tokenizer_et_consignes_qwen.md)

| Corpus | Partition | Lignes | Max du texte seul | Max zero-shot | Max few-shot |
| --- | --- | ---: | ---: | ---: | ---: |
| TSAC | Train | 10 053 | 376 | 426 | 471 |
| TSAC | Validation | 1 774 | 255 | 305 | 350 |
| MADAR | Train | 6 388 | 76 | 136 | 232 |
| MADAR | Validation | 800 | 69 | 128 | 224 |

Le budget de **512 tokens** est retenu selon la règle du train, avec **480 pour le prompt et 32 pour la réponse**. Au budget de 256, le plafond d’entrée de 224 exclurait 38 prompts few-shot TSAC et 3 MADAR du train. Au budget de 512, chaque prompt mesuré tient en entier dans les deux partitions. Le plafond est une limite supérieure. Un prompt court conserve sa longueur réelle pendant l’audit. [Consignes actives et paramètres](../configs/qwen_prompting_v2.json)

Sur la validation TSAC, la tokenisation produit environ 2,37 tokens par champ séparé par un espace dans le groupe arabe et 2,76 dans le groupe latin. Les effectifs sont respectivement 785 et 932. Le groupe latin réunit des écritures de plusieurs langues. Les groupes de moins de 100 lignes sont signalés dans le rapport. Les 12 sondes inventées sont restituées exactement après encodage et décodage, notamment les négations, les emojis et les chiffres Arabizi. Le tokenizer expose `unk_token_id = None` et le rapport encode le taux de tokens inconnus comme non applicable.

Les démonstrations sont deux exemples TSAC et trois MADAR, avec une ligne du train par classe et des groupes distincts. Les cinq exemples sélectionnés sont en écriture arabe. La sélection équilibre les classes. La représentativité des écritures reste une question pour l’analyse des scores de référence.

Le format de dialogue officiel et le parseur ont été vérifiés avec le tokenizer local. L’encodage direct du dialogue correspond à la tokenisation du prompt rendu. Des chaînes inventées contenant des délimiteurs de rôle, des chevrons et des retours à la ligne conservent leur contenu sous forme JSON et restent à l’intérieur du message. Le parseur accepte les labels exacts après retrait des espaces extérieurs. Une ponctuation ajoutée, un label en minuscules ou un délimiteur interne entraîne un résultat invalide.

L’audit agrégé a été recalculé à l’identique. Depuis la racine du dépôt, la commande de vérification est la suivante.

```sh
sh scripts/llm_uv.sh run --locked --offline python scripts/run_qwen_tokenizer_audit.py \
  --verify reports/qwen_tokenizer_audit_v2_2026-11-07
```

Pour générer un nouveau rapport, passer `--output` et choisir un répertoire neuf sous `reports/`. Le JSON versionnable contient des agrégats, des empreintes et les douze sondes inventées.

**L06 et L07.** Les consignes, les démonstrations reproductibles, le budget d’entrée et le parseur sont prêts. La prochaine mesure consiste à générer les prédictions zero-shot et few-shot sur la validation, puis à compter les réponses invalides dans les erreurs. Le corpus SFT et l’adaptateur partagé suivent cette référence.
