# Première validation du RAG documentaire

**9 novembre 2026.** Nous avons préparé 44 fiches réparties entre le français et l’arabe standard et 60 questions, dont 12 sans réponse dans les fiches. Les essais portent sur les 20 questions de validation, dont 16 avec réponse et 4 sans réponse. Les annotations de pertinence restent des propositions avant double relecture. [Protocole](66_protocole_rag_documentaire.md) · [Grille de relecture](../rag/review_sample_v1.md)

## Recherche des fiches

| Méthode | Recall@3 | MRR@3 | Questions avec réponse acceptées au seuil | Questions sans réponse rejetées au seuil |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0,875 | 0,875 | 9/16 | 4/4 |
| E5 multilingue | 1,000 | 1,000 | 8/16 | 4/4 |

E5 place la fiche attendue au premier rang pour les 16 questions avec réponse. Les deux erreurs de recherche de BM25 concernent des questions arabes. Après application du seuil, les deux méthodes rejettent aussi des questions auxquelles les fiches permettaient de répondre. La couverture observée au seuil est de 9/16 pour BM25 et 8/16 pour E5. Toutes les réponses extractives émises citent la fiche attendue et reproduisent son passage. Ces mesures proviennent du petit jeu qui a servi à ajuster le seuil. [Données de recherche](../reports/rag_validation_v1_2026-11-09/retrieval.json)

## Génération avec ou sans fiches

Qwen3-0.6B de base a répondu aux mêmes 20 questions avec les trois fiches classées par E5, puis avec un contexte vide. La consigne imposait une citation au format `[identifiant]` pour chaque fait ou le marqueur exact `INSUFFISANT`. Les sorties sont reproduites dans le rapport JSON.

| Condition | Citations toutes présentes dans les fiches fournies | Réponses avec citations toutes pertinentes | Abstentions lisibles parmi les 4 questions sans réponse |
| --- | ---: | ---: | ---: |
| Avec fiches E5 | 1/20 | 0/20 | 2/4 |
| Contexte vide | 0/20 | 0/20 | 4/4 |

Le parseur strict compte zéro marqueur exact. La dernière colonne reconnaît aussi le marqueur avec un point final ou une casse différente. Avec contexte, six sorties utilisent ce marqueur lisible, dont quatre pour des questions avec réponse. Sans contexte, 19 sorties l’utilisent. Plusieurs réponses avec contexte déforment les fiches. Par exemple, le modèle ajoute une classe `MIXED` à la tâche de sentiment, alors que la fiche mentionne uniquement `POS` et `NEG`. Nous conservons la réponse extractive pour ce prototype. La fidélité globale reste à annoter humainement. [Sorties de génération](../reports/rag_validation_v1_2026-11-09/generation.json)

## Reproduire les essais

Depuis la racine, préparer une fois les fichiers E5 officiels. Le profil LLM verrouillé est réutilisé. La comparaison générative utilise également le cache Qwen épinglé de l’expérience LoRA.

```sh
sh scripts/llm_uv.sh run --locked --offline python rag/fetch_e5.py --download
sh scripts/llm_uv.sh run --locked --offline python rag/evaluate.py \
  --output outputs/rag/ma_validation_recherche.json
sh scripts/llm_uv.sh run --locked --offline python rag/generate.py \
  --retrieval outputs/rag/ma_validation_recherche.json \
  --output outputs/rag/ma_validation_generation.json
```

Pour essayer la réponse extractive avec le seuil BM25 mesuré dans cette étude

```sh
sh scripts/project_uv.sh run --locked --offline python rag/search.py \
  --calibration reports/rag_validation_v1_2026-11-09/retrieval.json \
  --lang fr --question 'Que signifie macro-F1 et comment calcule-t-on la moyenne des F1 des classes ?'
```

**L08, L07 et L12 progressent.** Les fiches, le prototype, les deux recherches et les deux conditions de génération disposent de preuves sur validation. La relecture des fiches, la double annotation d’un échantillon et l’évaluation sur les 40 questions de test documentaire restent à réaliser avant de clôturer Silver. L’interface applicative et la relance indépendante constituent des étapes de recette. Le test officiel de classification reste réservé à son évaluation finale.
