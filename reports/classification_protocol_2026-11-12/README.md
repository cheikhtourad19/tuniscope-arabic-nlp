# Protocole final de classification vérifié

Choix fixés avant l’évaluation finale de classification. Les checkpoints sont déjà entraînés. Le contrôle vérifie les fichiers et les métadonnées des manifestes. Les textes bruts du test ne sont pas chargés et aucun score de test de classification n’est produit.

36 entrées de modèles ou adaptateurs, 48 conditions tâche/modèle, 224 fichiers liés par empreintes.

| Famille | Conditions principales | Comparaisons complémentaires |
| --- | ---: | ---: |
| majority | 2 | 0 |
| tfidf_char | 2 | 2 |
| tfidf_word | 0 | 2 |
| bigru | 6 | 0 |
| camelbert_confirmed | 6 | 0 |
| camelbert_paired | 0 | 12 |
| qwen_base | 4 | 0 |
| qwen_lora | 12 | 0 |

Les trois tests ciblés refusent une graine LoRA dupliquée, un checkpoint associé à la mauvaise tâche et un artefact dont l’empreinte diffère du gel. Les modèles, les tokenizers et les dépendances locales sont vérifiés par le contrôleur. Les premières métadonnées de registre sont conservées dans le dossier privé de préparation. La version courante est liée à la vérification ci-dessous.

[Protocole expliqué](../../docs/73_protocole_classification_finale.md) · [Registre gelé](../../configs/classification_final_v1.json) · [Vérification et archive](verification.json)
