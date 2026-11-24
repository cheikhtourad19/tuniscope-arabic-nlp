# Réconciliation de la traçabilité de classification et LoRA

Le 12 novembre 2026, nous avons vérifié les trois archives LoRA d’origine et les traces du Qwen de base afin de distinguer les preuves conservées des informations historiques absentes. Le programme `scripts/report_classification_traceability.py` produit un [bilan vérifiable](../reports/classification_traceability_2026-11-12/summary.json). Il effectue uniquement des lectures de fichiers et de l’historique Git local.

## Archives LoRA contrôlées

Les trois ZIP des graines 13, 42 et 2026 ont les mêmes empreintes que celles enregistrées à leur import. Leurs copies présentes dans Downloads sont identiques. L’intégrité ZIP et l’égalité octet par octet de chaque fichier extrait sont vérifiées.

Chaque archive comporte six fichiers utiles. Le plan commun, les informations d’apprentissage, le contrôle court, le résultat de validation de la graine, la configuration de l’adaptateur et ses poids sont présents. Les résumés de validation incluent les matrices, les effectifs et les empreintes attendues des prédictions individuelles.

Les douze fichiers de prédictions CUDA attendus, correspondant à trois graines et quatre conditions, ne sont pas contenus dans ces archives. Une recherche locale des fichiers candidats, avec comparaison aux empreintes enregistrées, n’en retrouve aucun. Les poids sont disponibles et vérifiés. Les scores agrégés restent recalculables depuis les matrices. L’alignement original ligne par ligne de CUDA ne peut pas être rejoué localement à partir de ces seuls résumés.

Le contrôle des candidats porte sur les fichiers de conditions sous `outputs/qwen/`. Une [recherche complémentaire dans Downloads](../reports/classification_traceability_2026-11-12/download_search.json), limitée aux noms d’exports de ces conditions, ne trouve aucun JSON individuel supplémentaire. Ces recherches décrivent les emplacements contrôlés localement. Elles ne constituent pas une recherche dans une session Kaggle distante.

Le résumé original global de campagne et son temps mural complet ne sont pas reçus. Le rapport de campagne publié est identifié comme une reconstruction depuis les résumés originaux de chaque graine. La somme de durées parallèles ne remplace pas un temps mural absent. Nous conservons les durées par graine déjà mesurées et leur périmètre.

| Élément | Preuve disponible | État enregistré |
| --- | --- | --- |
| Modèle Qwen exact | Identifiant, révision et empreinte du modèle de base | Conservé |
| Trois adaptateurs | Poids et configurations comparés aux ZIP originaux | Conservés et vérifiés |
| Formation LoRA | Logs, pertes, gradients, gel de la base et rechargement par graine | Conservés |
| Scores de validation | Matrices, métriques et effectifs par condition | Recalculés et vérifiés |
| Prédictions CUDA individuelles | Empreintes attendues dans les résumés | Originaux non retrouvés localement |
| Résumé global original | Trois résumés par graine disponibles | Index dérivé identifié comme reconstitué |
| Temps mural global | Durées partielles disponibles | Valeur globale non disponible |

Ces réserves portent sur les exports historiques. Une future génération de prédictions depuis les poids conservés constituerait une nouvelle exécution, avec son matériel, ses empreintes et ses résultats propres. Elle ne serait pas présentée comme une récupération des fichiers CUDA d’origine.

## Sources du Qwen de base

Le plan de la référence Qwen de base n’enregistre pas de commit Git d’exécution. Il conserve cependant les empreintes SHA-256 de quatre fichiers de code. Les quatre fichiers locaux correspondent exactement à ces empreintes.

La recherche dans l’historique trouve quatre révisions dont ces sources sont compatibles, abrégées ici en `6c74ce2`, `c7f979f`, `e00cc4e` et `9c4fb77`. Leurs identifiants complets sont enregistrés dans le rapport. Plusieurs révisions peuvent contenir les mêmes fichiers. Cette compatibilité ne permet pas d’identifier le commit effectivement actif au moment de la référence. Le champ historique reste donc non renseigné.

La traçabilité de la référence repose sur les empreintes exactes du code, du modèle, des manifestes, du tokenizer, de la configuration de consignes et des générations conservées. Le protocole final de classification enregistre désormais son propre état Git et ses empreintes avant le test, puis impose un enregistrement complet lors de l’exécution future.

## Disposition retenue

Les preuves récupérables localement sont vérifiées et indexées. Les informations absentes sont signalées avec leur portée précise. Aucun score ni fichier de prédictions n’est reconstruit artificiellement à partir d’une matrice. Aucun commit historique n’est attribué par supposition. Cette réconciliation documente les réserves pour **L06**, **L07**, **L08** et **L12** et permet de préparer la présentation avec des preuves identifiées.

Pour répéter le contrôle avec les fichiers locaux conservés, choisir une nouvelle destination.

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/report_classification_traceability.py \
  --output reports/ma_reconciliation_classification
```
