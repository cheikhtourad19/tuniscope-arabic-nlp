# TuniScope — Application locale

Interface **React + Vite**, backend **FastAPI**, deux conteneurs **Docker Compose**. L’atelier réunit TF-IDF, CAMeLBERT adapté, Qwen de base et Qwen + LoRA. Le module documentaire retrouve des passages sourcés avec BM25 ou E5. Les résultats de classification et la validation documentaire sont affichés séparément. Cette version contribue à **L09**.

## Démarrage avec Docker

Depuis la racine du projet, avec Docker Desktop ou OrbStack démarré, préparer les deux bundles puis construire les conteneurs. Ces commandes supposent les campagnes vérifiées et les caches Qwen et E5 du projet disponibles localement.

```sh
docker compose -f App/compose.yaml up --build -d
```

- Interface : <http://localhost:8080>
- Documentation interactive de l'API, via le même serveur : <http://localhost:8080/api/docs>
- État des modèles : <http://localhost:8080/api/health>

Les ports sont liés à l'interface locale uniquement. Une autre application utilise déjà 8000 ou 8080 ?

```sh
API_PORT=8001 WEB_PORT=8081 docker compose -f App/compose.yaml up --build -d
```

Consulter les journaux et arrêter :

```sh
docker compose -f App/compose.yaml logs --tail 50
docker compose -f App/compose.yaml down
```

## Quels modèles sont servis ?

Le dossier **`App/models/current/`** contient les modèles retenus pour l'application. Il est ignoré par Git, sauf son `.gitkeep` ; un clone neuf doit préparer ce dossier. Les corpus ne sont pas requis pour l'inférence.

La sélection initiale est le run explicite `outputs/baselines/2026-10-17_grouped_v2_full`, TF-IDF caractères avec régression logistique. Les caractères sont la meilleure référence classique comparée sur notre validation v2 actuelle ; cela ne constitue pas une sélection sur le test officiel ni un modèle final universel.

Préparer le bundle depuis un run existant :

```sh
sh scripts/project_uv.sh run --locked --offline python App/prepare_models.py \
  --run outputs/baselines/2026-10-17_grouped_v2_full
```

Le script copie **quatre pipelines** (sentiment et dialecte, brut et normalisé) et leurs métadonnées agrégées `results.json`. Il contrôle les empreintes, le mode `full`, les manifestes v2 et l'absence d'évaluation test dans ce profil de bundle. Il ne copie ni corpus, ni prédictions individuelles, ni données de validation. Le run source est conservé.

### Préparer CAMeLBERT, Qwen et E5

La copie privée `App/models/neural/` est distincte du bundle classique. La commande sélectionne un checkpoint CAMeLBERT par tâche et un adaptateur LoRA partagé, puis vérifie les empreintes. La **graine 42** est le choix de démonstration fixé pour les deux familles. Les moyennes sur trois graines restent des résultats de campagne distincts. Qwen de base et Qwen + LoRA partagent une seule instance de Qwen, avec l’adaptateur désactivé pour la référence. Les deux modes Qwen utilisent la consigne directe zero-shot de validation.

```sh
sh scripts/project_uv.sh run --locked --offline python App/prepare_neural.py \
  --encoder-run outputs/encoder_finetuning/confirmation/notebooks/177d9f5ad78b4ef4907e97d97abe96f8_full \
  --lora-run outputs/qwen/2026-11-07_cuda_lora_c10d02666e154607ac9d7faddbd47838_full \
  --seed 42
```

Ajouter `--replace` pour remplacer la copie applicative. Le Dockerfile copie les deux bundles et contrôle le chargement avant le démarrage de l’API. Dans Docker sur Mac, l’inférence tourne sur CPU. En exécution native, CAMeLBERT et Qwen utilisent MPS lorsqu’il est disponible. E5 utilise le CPU, comme lors de la calibration. Les fichiers sources et archives d’expérience sont conservés à leur emplacement. Les dossiers de poids et l’image Docker restent locaux.

### Remplacer les modèles après une meilleure expérience

Choisir la nouvelle expérience d'après le protocole de validation, puis remplacer uniquement la copie applicative :

```sh
sh scripts/project_uv.sh run --locked --offline python App/prepare_models.py \
  --run outputs/baselines/NOM_DU_NOUVEAU_RUN_FULL \
  --system tfidf_char --replace
docker compose -f App/compose.yaml up --build -d
```

`--system` accepte aussi `tfidf_word` et `tfidf_word_char`, avec les mêmes quatre variantes/tâches. Aucun dossier « dernier run » ni choix automatique du meilleur résultat. `--replace` retire les anciennes **copies dans `current/` seulement**, après vérification de la nouvelle source. Les archives d'expérience dans `outputs/` restent intactes.

Le copier-coller manuel est également possible : déposer le `results.json` du run et les checkpoints TF-IDF correspondants dans `current/`, retirer les anciennes copies, puis reconstruire. Ne pas renommer les checkpoints : le JSON référence leurs noms et SHA-256. Le script est préférable pour éviter un mélange de runs. Les checkpoints de majorité ne sont pas servis par cette version.

**Trois moments distincts :**

1. Préparation : copie des fichiers retenus dans `current/` et `neural/`.
2. Build Docker : copie du bundle dans l'image, puis chargement de contrôle. Le build échoue si le bundle est absent, corrompu ou incompatible.
3. Démarrage : chargement en mémoire une seule fois, avant l'ouverture du serveur HTTP. Chaque requête réutilise ces objets ; aucun entraînement ni accès aux corpus.

Un simple `docker compose restart` ne copie pas de nouveaux fichiers. Utiliser `up --build -d`. Aucun volume ne masque les modèles intégrés à l'image.

Les versions scikit-learn, NumPy, SciPy, joblib et threadpoolctl sont comparées à celles du run. Si une future expérience change ces dépendances, mettre à jour le profil backend, régénérer `uv.lock` et `requirements.lock`, puis vérifier le rechargement. Les poids neuronaux utilisent le manifeste préparé par `App/prepare_neural.py`.

## Fonctionnalités et API

| Route | Fonction |
| --- | --- |
| `GET /health` | Versions, disponibilités et raisons d'indisponibilité ; aucune saisie utilisateur |
| `GET /ready` | 200 si les deux tâches ont un modèle utilisable ; sinon 503 |
| `GET /models` | Catalogue des modèles |
| `POST /predict` | `text`, `task`, `model`, `id` facultatif ; label, scores, statut, textes brut/normalisé et versions |
| `POST /compare` | Même texte et identifiant pour toutes les variantes disponibles d'une tâche |
| `POST /batch` | Multipart : `file`, `task`, `model` ; une réponse par ligne, erreurs conservées |
| `GET /metrics` | **Validation sauvegardée uniquement**, `test_evaluated: false` |
| `POST /ask` | `question` requise, `lang` et `method` facultatifs ; passage, citations ou abstention |
| `GET /documentation/{path}` | Référence d’une fiche du catalogue, servie en texte brut |

Les routes sont préfixées par `/api` via le serveur frontend. Exemple :

```sh
curl http://localhost:8080/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"el service behi barcha","task":"sentiment","model":"tfidf_char__raw-v0"}'
```

Les tâches sont indépendantes : `sentiment` ou `dialect`. Le sentiment ne filtre pas les entrées d'après une prédiction dialectale préalable.

- Texte : au maximum **2 000 points de code Unicode**, sans troncature silencieuse.
- CSV : UTF-8, BOM accepté, séparateur virgule, colonnes `id,text`, **100 lignes et 1 Mo** maximum ; identifiants non vides et uniques.
- Exports JSON/CSV : prédictions, erreurs et versions. Le CSV neutralise les débuts de cellule pouvant être interprétés comme formules par un tableur.
- Normalisation : réutilisation de `scripts/normalize_text.py` (NFC et espaces). Le modèle brut reçoit le texte brut ; l'autre reçoit la normalisation.
- Scores : sorties non calibrées pour TF-IDF et CAMeLBERT. Qwen rend un label ou une abstention, avec une liste de scores vide.
- CAMeLBERT applique le plafond de 128 tokens utilisé en validation et signale une troncature. Qwen s’abstient lorsque le prompt dépasse 480 tokens ou que la sortie ne correspond à aucun label.
- RAG : sélectionner la langue des fiches et BM25 ou E5 dans **Le projet**. Le backend construit l’index BM25 et les embeddings E5 une fois au démarrage. Chaque question E5 encode seulement la question. La réponse cite un extrait de la fiche, ou signale que les sources sont insuffisantes.
- Abstention TF-IDF : absence totale de caractéristiques connues dans le vocabulaire. Ce contrôle couvre seulement une partie des textes hors domaine.
- L'écriture AR/LATIN/MIXED/UNKNOWN indique la présence de caractères, pas une langue identifiée.
- Aucune métrique n'est calculée sur les saisies libres, qui n'ont pas de labels de référence.

Le cahier demande à terme les métriques du test figé. Cette version affiche explicitement la **validation**, car l'évaluation finale n'est pas encore ouverte.

## Développement sans Docker

Le backend possède son environnement et son verrouillage propres : l'environnement d'entraînement à la racine n'est pas modifié.

```sh
uv sync --project App/backend --locked --extra neural
App/backend/.venv/bin/python -m uvicorn App.backend.service.main:app \
  --host 127.0.0.1 --port 8000 --root-path /api --no-access-log
```

Dans un second terminal :

```sh
cd App/frontend
npm ci
npm run dev
```

Ouvrir <http://localhost:5173>. Vite relaie `/api` vers le backend local. Les polices sont intégrées au build, sans appel à Google Fonts.

## Vérifications

Depuis la racine :

```sh
App/backend/.venv/bin/python -m pytest App/backend/tests -q
```

Avec l'application démarrée par Compose :

```sh
cd App/frontend
npm ci
npx playwright install chromium --no-shell
npm run test:e2e
```

Pour viser le serveur de développement : `APP_URL=http://localhost:5173 npm run test:e2e`.

Les tests API utilisent des exemples inventés et des modèles minuscules construits dans un dossier temporaire. Les tests navigateur utilisent le bundle réellement servi et des textes fictifs, sans corpus protégé. La recette humaine reste à exécuter ; voir `docs/36_application_inference_et_recette.md`.

## Usage local et confidentialité

L'application ne conserve pas les saisies, ne les utilise pas pour entraîner et n'active pas les journaux d'accès Uvicorn. Les textes restent visibles dans le navigateur et dans les exports demandés. Le chargement joblib est réservé à nos checkpoints locaux de confiance : un fichier joblib externe peut exécuter du code lors du chargement. Aucun upload de modèle n'est exposé par l'API.

Le contexte Docker backend est limité au code d’inférence, aux dépendances, aux bundles, aux fiches et aux références autorisées. Ne pas publier les images contenant les poids et leur vocabulaire sans vérifier les droits des dérivés des corpus. Aucun envoi vers un registre n'est effectué par les commandes ci-dessus.

### Inspection sans modèles

Pour examiner uniquement les états indisponibles sur un clone sans bundle :

```sh
REQUIRE_MODELS=false REQUIRE_NEURAL=false docker compose -f App/compose.yaml up --build -d
```

L'interface affichera l'indisponibilité ; aucune prédiction de remplacement ne sera fabriquée. Le healthcheck Docker vérifie la disponibilité HTTP pour que l'interface reste accessible ; `/ready` indique séparément la disponibilité des modèles.

## Références techniques

- [FastAPI : lifespan et chargement avant le service des requêtes](https://fastapi.tiangolo.com/advanced/events/)
- [Docker Compose : dépendances et healthchecks](https://docs.docker.com/compose/how-tos/startup-order/)
- [scikit-learn : persistance, compatibilité et fichiers de confiance](https://scikit-learn.org/stable/model_persistence.html)
- [Vite : développement et construction du frontend](https://vite.dev/guide/)
