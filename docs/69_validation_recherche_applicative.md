# Validation de la recherche documentaire applicative

Le 11 novembre 2026, nous avons comparé les politiques de recherche et d’abstention utilisées par l’application. Les 44 fiches sont inchangées. L’essai porte sur les mêmes 20 questions de validation que la première expérience, soit 16 questions documentées et quatre sans réponse. Dix questions sont en français et dix en arabe standard. Les seuils et les règles existants sont conservés sans nouvel ajustement pendant cet essai.

## Recherche et décision de répondre

Pour mesurer la recherche, nous conservons les trois premières fiches avant l’abstention. Recall@3 indique la part des fiches pertinentes retrouvées dans ces trois positions. MRR@3 tient compte du rang de la première fiche pertinente. Pour mesurer le comportement applicatif, nous relevons séparément le passage réellement affiché après les critères de réponse.

| Moteur applicatif | Recall@3 | MRR@3 | Réponses avec citation pertinente sur 16 questions documentées | Abstentions attendues sur 4 questions sans réponse |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0,875 | 0,875 | 13/16 | 4/4 |
| E5 multilingue | 1,000 | 1,000 | 8/16 | 4/4 |

Le classement BM25 retrouve la fiche attendue au premier rang pour 14 des 16 questions documentées. Il affiche une réponse pertinente pour 13 questions et s’abstient sur les trois restantes. Deux de ces refus concernent des recherches arabes où la fiche attendue manque aux trois premiers rangs. Le troisième concerne une question française dont la fiche attendue est bien classée, mais dont le score ne satisfait pas les critères de réponse.

E5 place la fiche attendue au premier rang pour les 16 questions documentées. Son seuil de cosinus reste toutefois supérieur au score obtenu pour huit d’entre elles. Il affiche donc huit réponses pertinentes et refuse les huit autres. Un bon classement et une forte couverture des réponses émises sont deux propriétés distinctes. Les deux moteurs rejettent les quatre questions sans réponse de cette validation.

| Langue | Moteur | Fiche attendue dans les trois premiers rangs | Réponses pertinentes émises | Questions sans réponse rejetées |
| --- | --- | ---: | ---: | ---: |
| Français | BM25 | 8/8 | 7/8 | 2/2 |
| Arabe standard | BM25 | 6/8 | 6/8 | 2/2 |
| Français | E5 | 8/8 | 3/8 | 2/2 |
| Arabe standard | E5 | 8/8 | 5/8 | 2/2 |

Les paramètres BM25 utilisés dans l’application ont évolué depuis la première expérience du document 67. Sur cette validation, les réponses pertinentes émises passent de 9 à 13 sur 16. Les résultats historiques restent conservés. Les scores E5 et son seuil restent ceux de la politique précédente.

## Politique évaluée

BM25 utilise une politique commune aux sujets pour les questions comportant au plus cinq termes utiles. Les mots grammaticaux sont retirés, certaines variantes lexicales sont rapprochées selon le vocabulaire existant et les correspondances dans les titres reçoivent un bonus. La première fiche doit couvrir au moins 40 % du poids des termes utiles dans son titre et son passage. Son avance sur la deuxième doit atteindre au moins 5 % de son propre score.

Les questions lexicales plus longues conservent le seuil initial de 8,187184890704186. Une réponse est aussi acceptée si le premier score atteint 3,5 et dépasse le deuxième d’au moins 25 % du premier score. E5 utilise le seuil de cosinus 0,8687087297439575. Ces valeurs sont des critères de décision propres à chaque moteur. Elles ne sont pas des probabilités mesurées de correction.

## Preuves et reproduction

L’évaluateur réutilise `Documents.retrieve` et rejoue les 40 réponses via `Documents.ask`, soit 20 questions par moteur. La recherche E5 utilise le cache officiel épinglé déjà acquis et s’exécute sur CPU. La durée observée est de 3,95 secondes, chargement et rejeu inclus, hors démarrage de l’interpréteur. Le classement de validation et les décisions sont conservés dans un [rapport détaillé](../reports/rag_application_validation_2026-11-11/results.json). La [vérification](../reports/rag_application_validation_2026-11-11/verification.json) recalcule les métriques à partir des classements enregistrés et contrôle l’alignement des questions et les passages cités. Les 45 tests du backend et du moteur documentaire réussissent.

Avec l’environnement applicatif neural déjà installé, depuis la racine du projet, choisir un nouveau dossier.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  App/backend/.venv/bin/python rag/evaluate_application.py \
  --output reports/ma_validation_applicative
```

Les figures se produisent depuis le rapport vérifié, dans le profil encodeur qui dispose déjà de Matplotlib.

Les deux images vérifiées montrent le [classement avant abstention](../reports/rag_application_validation_2026-11-11/figures/recherche.png) et les [réponses effectivement affichées](../reports/rag_application_validation_2026-11-11/figures/reponses.png). Une archive privée conserve les sources exécutées et les seules questions de validation. Son [manifeste d’intégrité](../reports/rag_application_validation_2026-11-11/source_archive.json) permet de vérifier la correspondance avec cette expérience.

```sh
MPLCONFIGDIR=.tools/matplotlib sh scripts/encoder_uv.sh run --locked --offline \
  python scripts/plot_rag_validation.py \
  --run reports/ma_validation_applicative
```

## Limites et suite

Les annotations de pertinence restent des propositions assistées par IA, avec deux grilles préremplies par le même assistant. La revue éditoriale des 44 fiches a été reçue de l’auteur, mais elle ne constitue pas une double annotation indépendante des questions. La réalisation individuelle et l’absence de deuxième avis humain restent déclarées. Une adaptation de cette exigence doit être acceptée par l’enseignant pour la clôture du jalon.

Ces 20 questions ont déjà servi à la mise au point. Leurs résultats décrivent le développement du système et ne fournissent pas une estimation indépendante de ses performances futures. Le français et l’arabe couvrent des sujets appariés dans ce petit jeu. Nous conservons les réglages mesurés et les empreintes comme version candidate avant une évaluation finale. Les 40 questions de test documentaire et le test officiel de classification ne sont pas évalués par cette expérience.

Cette étape apporte des preuves pour **L07**, **L08**, **L09** et **L12**. La validation indépendante des annotations, l’évaluation documentaire finale et la clôture de Silver restent ouvertes.
