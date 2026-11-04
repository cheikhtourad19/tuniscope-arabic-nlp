# Application d'inférence et recette initiale

**Date :** 4 novembre 2026. **Livrables : L09, L08 et L12 partiels.** Nous démarrons explicitement l'intégration applicative à notre demande. Cette étape ne remplace pas les références manquantes de L04, les entraînements encodeur/LoRA ou l'évaluation finale.

## Architecture mise en œuvre

Nous utilisons React avec Vite pour l'interface et FastAPI pour l'inférence. Docker Compose construit deux images ; Nginx sert les fichiers statiques et relaie `/api` vers FastAPI. Les conteneurs s'exécutent sans privilège, avec un système de fichiers en lecture seule et un espace temporaire dédié. Les ports hôte sont limités à localhost.

L'application est contenue dans `App/`. Le backend réutilise directement le normaliseur versionné `scripts/normalize_text.py`, sans importer les scripts d'apprentissage ou les chargeurs de corpus. Les requêtes n'entraînent aucun modèle et n'accèdent pas aux partitions.

Nous distinguons la préparation du bundle, sa copie lors du build et son chargement au démarrage :

```text
run full v2 explicitement sélectionné
    → App/prepare_models.py
    → App/models/current/ : quatre checkpoints et résultats agrégés
    → build : COPY et contrôle de chargement
    → démarrage : chargement en mémoire avant le service HTTP
    → requêtes : préparation du texte puis prédiction
```

Le choix initial est `2026-10-17_grouped_v2_full`, système TF-IDF caractères avec régression logistique. Il est fondé sur les comparaisons exploratoires de validation documentées, sans nouvelle sélection sur le test. Les deux tâches et les variantes brut/normalisé sont disponibles. Les runs d'origine et leurs modèles sont préservés.

Les checkpoints et `results.json` dans `App/models/current/` sont exclus de Git. Le `.dockerignore` du backend limite le contexte aux dépendances, à l'inférence, au normaliseur et au bundle ; les corpus bruts, les prédictions individuelles, les secrets et les notebooks ne sont pas copiés dans l'image. Les images locales contenant ces dérivés ne sont pas publiées.

## Contrats effectivement implémentés

| Route | Comportement |
| --- | --- |
| `/predict` | Une tâche indépendante, un texte et un modèle explicites ; identifiant, label, statut, scores et versions |
| `/compare` | Les variantes disponibles d'une tâche reçoivent le même texte et le même identifiant |
| `/batch` | CSV UTF-8 `id,text`, maximum 100 entrées et 1 Mo ; succès ou erreur pour chaque ligne |
| `/health`, `/ready`, `/models` | Disponibilité, versions et catalogue ; aucune saisie utilisateur |
| `/metrics` | Résultats de validation sauvegardés du bundle, explicitement séparés des saisies libres |
| `/ask` | Contrat réservé au futur RAG ; statut non implémenté, sans réponse ni citations inventées |

Les formats exacts et exemples sont disponibles dans la documentation OpenAPI servie par l'application. L'interface offre un export JSON ou CSV, y compris pour les erreurs d'un batch. Les débuts de cellule susceptibles de devenir des formules dans un tableur sont neutralisés à l'export CSV.

Le mode scientifique ne filtre pas le sentiment selon le dialecte prédit. Le texte brut et sa version normalisée restent visibles ; chaque modèle utilise la variante déclarée. Les saisies dépassant 2 000 points de code Unicode sont refusées, sans troncature silencieuse.

## Chargement, intégrité et indisponibilité

Nous contrôlons les empreintes SHA-256 des checkpoints, le mode full, la version v2 des manifestes et les labels attendus. Le backend impose les mêmes versions de scikit-learn, NumPy, SciPy, joblib et threadpoolctl que celles enregistrées dans le run. Son environnement est verrouillé séparément : les définitions d'apprentissage à la racine restent inchangées.

Le build échoue par défaut si le bundle est absent, incohérent ou incompatible. Le démarrage strict refuse également un bundle inutilisable. Un profil explicite `REQUIRE_MODELS=false` permet seulement d'inspecter l'interface en état indisponible ; aucune autre API ou prédiction fictive ne prend la place du modèle manquant.

Les fichiers joblib doivent provenir de nos expériences locales de confiance. Leurs empreintes détectent un changement par rapport aux métadonnées ; elles ne prouvent pas l'innocuité d'un modèle externe si ses métadonnées sont elles-mêmes non fiables. Aucun téléversement de modèle n'est exposé par l'application.

## Interprétation des sorties

Les scores sont les sorties du classifieur et ne sont pas calibrés. Nous ne les présentons pas comme une probabilité mesurée de correction. La seule abstention implémentée correspond à un vecteur TF-IDF entièrement nul : elle évite une décision sans indice connu, sans prétendre détecter tous les textes hors domaine.

Le sentiment reste limité aux classes POS/NEG de commentaires tunisiens ; aucune classe neutre n'a été apprise. Le domaine dialectal reste celui de MADAR, avec MSA, Tunis/Sfax et Le Caire. Les catégories d'écriture sont des présences Unicode et ne constituent pas une identification de langue ou d'origine.

Les métriques exposées sont les valeurs de **validation**, associées au run source et à ses effectifs. Le test officiel n'a pas été évalué. L'affichage des scores du test demandé par le contrat final reste donc ouvert et n'est pas remplacé par un renommage de la validation.

## Vérifications et recette

Les commandes et la procédure de remplacement sont documentées dans `App/README.md`. Les tests API utilisent des textes inventés et des modèles temporaires ; ils ne lisent aucun corpus protégé. Les vérifications navigateur utilisent les modèles réellement servis et des exemples fictifs.

Les empreintes, versions et identifiants d'images sont conservés localement dans `outputs/app_2026-11-04/verification.json`, avec les captures d'écran. Le runtime Docker vérifié est Linux aarch64, Python 3.12.16, UID 10001. Les cinq parcours navigateur ont été relancés après les retouches sur les images finales. La suite API réussit avec un avertissement de dépréciation du client httpx émis par Starlette ; aucun échec n'est masqué.

| Contrôle | Preuve actuelle |
| --- | --- |
| Rechargement réel des quatre modèles | Contrôle local Python et étape de build Linux ARM64 réussis |
| Build des deux images et démarrage Compose | Réalisés localement, backend HTTP healthy |
| Routes santé, disponibilité, métriques et OpenAPI via Nginx | Réponses HTTP 200 vérifiées |
| Prédiction réelle via Nginx | Exemple fictif de sentiment, label et provenance retournés |
| Contrats API, entrées invalides, normalisation, comparaison | 24 tests API réussis, avec les contrôles de bundle et de remplacement |
| CSV encodage, limite, doublons d'identifiants et erreurs par ligne | Cas couverts par les tests API |
| Recette navigateur et inspection visuelle | Cinq parcours Chromium réussis contre les conteneurs ; captures ordinateur 1 440 px et mobile 390 px inspectées, retouches de lisibilité appliquées |
| Recette personnelle | À réaliser par nous sur l'application livrée |
| Latence médiane et p95 sur 100 requêtes après 10 chauffes | Non mesurées ; les tests fonctionnels ne constituent pas cette preuve |

### Recette personnelle à cocher

- [ ] Ouvrir l'application et retrouver la provenance des modèles.
- [ ] Analyser un avis positif et négatif fictif, puis une phrase dialectale.
- [ ] Comparer brut et normalisé et vérifier le texte reçu par chaque variante.
- [ ] Importer le CSV fictif et vérifier l'export JSON/CSV.
- [ ] Essayer une ligne vide, un identifiant en double, un fichier non UTF-8 et un texte trop long.
- [ ] Examiner les avertissements pour un texte français, un avis mixte et une saisie hors domaine.
- [ ] Vérifier que les résultats affichés sont ceux de validation et que le RAG reste annoncé indisponible.
- [ ] Remplacer le bundle par un run explicitement retenu, reconstruire et vérifier la nouvelle provenance.

## État des livrables

**L09 :** première application classique fonctionnelle et contrats documentés ; la recette humaine, les mesures de latence, les branches encodeur/LLM/RAG, leurs cas d'erreur et les scores finaux restent à compléter.

**L08 :** bundle local de modèles classiques avec provenance et contrôles ; cela ne termine ni les cartes de tous les modèles ni les artefacts RAG.

**L12 :** dépendances backend/frontend verrouillées, images de base référencées par empreinte, commandes et tests fournis. Une reconstruction Docker locale ne remplace pas une relance indépendante sur une autre machine.

Le niveau Bronze et les livrables globaux restent partiels. Aucun nouvel apprentissage, nouvelle partition, nouvelle annotation ou score du test officiel n'est produit par cette intégration.

## Références d'implémentation

- [FastAPI — Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) : chargement des ressources avant le service.
- [Docker Compose — Startup order](https://docs.docker.com/compose/how-tos/startup-order/) : dépendance à l'état healthy.
- [scikit-learn — Model persistence](https://scikit-learn.org/stable/model_persistence.html) : versions compatibles et confiance dans les fichiers.
- [Vite — Getting started](https://vite.dev/guide/) : serveur de développement et build statique.
