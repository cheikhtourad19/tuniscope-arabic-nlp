# Évaluation technique des 40 questions documentaires

Le 12 novembre 2026, nous avons évalué la recherche documentaire et les réponses extractives de l’application avec un protocole gelé avant calcul des scores. Le test documentaire contient 40 questions, dont 32 documentées et huit sans réponse dans les fiches. Il comporte 20 questions françaises et 20 questions arabes. Le catalogue, les deux moteurs et leurs critères de réponse sont identiques à ceux du gel. Les scores présentés restent conditionnés par les références proposées par IA et l’absence de double revue humaine. [Protocole](70_protocole_gel_documentaire.md) · [Gel technique](../configs/rag_documentary_frozen_v1.json)

## Résultats globaux

| Mesure | BM25 applicatif | E5 multilingue |
| --- | ---: | ---: |
| Recall@3 sur les questions documentées | 0,843750 | 0,937500 |
| MRR@3 sur les questions documentées | 0,776042 | 0,885417 |
| Réponses pertinentes émises sur 32 questions documentées | 19/32 | 13/32 |
| Réponses émises sur l’ensemble des 40 questions | 28/40 | 16/40 |
| Citations pertinentes parmi les réponses émises | 19/28 — 67,86 % | 13/16 — 81,25 % |
| Réponses avec citation non pertinente | 9 | 3 |
| Abstentions attendues sur huit questions sans réponse | 4/8 | 6/8 |
| Abstentions sur 32 questions documentées | 8/32 | 18/32 |

Chaque question documentée possède ici une seule fiche pertinente proposée. Recall@3 correspond donc aussi à 27 questions sur 32 pour BM25 et 30 sur 32 pour E5. E5 retrouve davantage de fiches attendues et classe mieux la première fiche pertinente. Il répond moins souvent avec le seuil retenu. BM25 émet davantage de réponses pertinentes au total, mais aussi davantage de passages qui ne répondent pas à la demande.

Pour BM25, les neuf réponses non pertinentes comprennent cinq réponses à des questions documentées et quatre réponses à des questions sans réponse dans les fiches. Pour E5, les trois réponses non pertinentes comprennent une question documentée et deux questions sans réponse. Les décisions se réconcilient sur 40 questions par moteur, en distinguant réponses pertinentes, réponses non pertinentes, abstentions attendues et abstentions sur questions documentées. [Bilan des décisions](../reports/rag_documentary_test_v1_2026-11-12/figures/decisions.png)

## Résultats par langue

| Langue | Moteur | Recall@3 | MRR@3 | Réponses pertinentes sur 16 questions documentées | Citations pertinentes parmi les réponses émises | Abstentions attendues sur quatre questions sans réponse |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Français | BM25 | 0,812500 | 0,750000 | 9/16 | 9/15 | 1/4 |
| Arabe standard | BM25 | 0,875000 | 0,802083 | 10/16 | 10/13 | 3/4 |
| Français | E5 | 0,937500 | 0,895833 | 6/16 | 6/7 | 3/4 |
| Arabe standard | E5 | 0,937500 | 0,875000 | 7/16 | 7/9 | 3/4 |

Ces effectifs restent petits. Les questions françaises et arabes couvrent des sujets appariés. Les différences descriptives ne constituent pas une démonstration de significativité ou de supériorité sur toute question future.

## Erreurs observées

Les deux moteurs acceptent un passage de résultats CAMeLBERT sur validation lorsque la question demande le score du test officiel TSAC. Le passage cite bien des résultats existants et les décrit comme validation, mais ne satisfait pas la demande d’un résultat de test officiel. La proximité thématique ne suffit donc pas à garantir la pertinence.

BM25 accepte aussi une fiche TF-IDF pour la question portant sur la performance de Qwen sur des enregistrements audio tunisiens. Le catalogue ne fournit aucune telle mesure. Plusieurs autres erreurs concernent une fiche liée à LoRA qui ne précise pas les éléments nécessaires au rechargement, ou une description générale de TSAC qui ne répond pas au traitement des étiquettes contradictoires.

Les 44 réponses émises au total sont des extraits littéraux de leurs fiches, avec une référence existante. Ce contrôle établit la fidélité de l’extraction au catalogue. Une citation existante et correctement reproduite peut cependant être non pertinente pour la question. La validité factuelle des sources et la pertinence des références restent soumises aux limites de revue déclarées.

## Traçabilité et limites

Le gel lie 35 fichiers par empreinte, dont le code, les 44 fiches, les questions et références proposées, les sources documentaires, la calibration et le verrou applicatif. Les empreintes sont vérifiées avant chargement des moteurs et avant sauvegarde. Les 80 réponses applicatives sont rejouées. Les mesures et les passages cités sont revérifiés après relecture du JSON enregistré. La durée observée est de 4,11 secondes sur CPU, chargement et rejeu inclus, hors démarrage de l’interpréteur. Les versions, le matériel logiciel et le commit de départ avec arbre de travail modifié sont enregistrés dans le [rapport détaillé](../reports/rag_documentary_test_v1_2026-11-12/results.json).

Les contrôles de gel et de citation ont réussi, ainsi que les 48 tests backend/recherche, dont trois tests ciblés sur des fichiers modifiés ou absents et une réponse artificiellement altérée. Les sources gelées sont archivées dans le dossier privé `outputs/rag/documentary_test_v1_2026-11-12/`. Leur [manifeste](../reports/rag_documentary_test_v1_2026-11-12/source_archive.json) lie l’archive au protocole. Les figures de [recherche](../reports/rag_documentary_test_v1_2026-11-12/figures/recherche.png), de [réponses](../reports/rag_documentary_test_v1_2026-11-12/figures/reponses.png) et de [décisions](../reports/rag_documentary_test_v1_2026-11-12/figures/decisions.png) sont inspectées et liées aux résultats par empreintes.

Les questions Q033, Q034, Q057 et Q058 figuraient déjà dans l’échantillon de revue publié. Les 40 formulations ne sont donc pas présentées comme inconnues de l’auteur. Leurs scores n’avaient pas servi à régler cette version avant le gel. Les références de pertinence restent proposées par IA, avec deux grilles préremplies par le même assistant. Aucun accord interévaluateurs humain ni dispense pédagogique n’est annoncé. Les résultats constituent une évaluation technique du système gelé, conditionnée par ces références provisoires.

Les paramètres et les références gelés sont conservés après lecture de ces résultats. Une amélioration future fondée sur leurs erreurs devra être déclarée comme développement après consultation du test. Le test officiel de classification TSAC/MADAR reste réservé à sa phase propre.

## Reproduction

Avec l’environnement neural applicatif déjà installé, vérifier le gel depuis la racine.

```sh
App/backend/.venv/bin/python rag/final_evaluation.py check \
  --protocol configs/rag_documentary_frozen_v1.json
```

Les résultats enregistrés et leurs figures suffisent pour consulter cette expérience. Une reproduction explicite du calcul doit utiliser une nouvelle destination.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  App/backend/.venv/bin/python rag/final_evaluation.py evaluate \
  --protocol configs/rag_documentary_frozen_v1.json \
  --output reports/ma_reproduction_test_documentaire
```

Cette étape apporte des preuves techniques pour **L07**, **L08**, **L09** et **L12**. La comparaison documentaire sur 40 questions est exécutée. La conformité de la revue humaine et la consolidation du dossier global Silver restent ouvertes.
