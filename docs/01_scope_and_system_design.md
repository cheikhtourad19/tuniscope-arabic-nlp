# TuniScope — Cadrage et architecture du système

**Livrable :** L01, version de travail.

**Réalisation :** nous menons ce projet individuellement en cinquième année du cycle ingénieur en informatique.

**État :** périmètre défini ; aucun résultat expérimental n'est encore disponible.

## 1. Objectif

TuniScope étudie l'identification de variétés de l'arabe et l'analyse de sentiment de textes tunisiens avec un budget limité en données et en calcul. L'étude compare des méthodes classiques, un encodeur Transformer adapté et un petit modèle génératif adapté par LoRA. Les limites liées à l'Arabizi et au mélange avec le français ou l'anglais sont examinées explicitement.

## 2. Problématique

**Question principale :** l'adaptation LoRA d'un petit modèle de langue améliore-t-elle l'identification dialectale et la classification du sentiment tunisien par rapport au même modèle sans adaptation ?

Les résultats seront également comparés à des classifieurs TF-IDF et à un encodeur finement ajusté. Cette comparaison mesure l'intérêt pratique de systèmes différents ; elle n'isole pas l'effet de LoRA à architecture constante.

## 3. Périmètre

| Élément | Décision de cadrage |
| --- | --- |
| Identification de variété | `MSA` (arabe standard moderne), `TN` (Tunis et Sfax) et `EG` (Le Caire). |
| Sentiment tunisien | `POS` et `NEG` ; aucune classe neutre n'est supposée. |
| Corpus | MADAR pour la variété ; TSAC pour le sentiment. |
| Références classiques | Classe majoritaire, règles linguistiques, TF-IDF de mots et de caractères. |
| Encodeur | CAMeLBERT Mix ; deux classifieurs distincts sont acceptés. |
| Modèle génératif | Qwen3-0.6B, avant et après un adaptateur LoRA multitâche partagé. |
| Recherche documentaire | Petit système RAG avec passages cités, limité aux questions sur la méthode et ses limites. |
| Application | Saisie de texte, import CSV, comparaison des modèles et export des résultats. |

L'audit des données déterminera les effectifs exploitables. Toute modification des classes ou du test officiel TSAC exigera une révision écrite du protocole.

## 4. Carte du système

```text
Corpus autorisés → audit → manifestes et partitions → normalisation tracée
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
                Baselines classiques     Encodeur adapté       Qwen3 + LoRA
                       └──────────────────────┼──────────────────────┘
                                              ▼
                              Évaluation commune et analyse d'erreurs
                                              ▼
                                      Application locale

Fiches documentaires vérifiées → recherche RAG → réponses sur le protocole
```

Le RAG renseigne sur les sources et la méthode ; ses passages ne constituent pas une preuve de la justesse d'une prédiction de dialecte ou de sentiment.

## 5. Principes expérimentaux

- Conserver le texte brut et versionner les transformations appliquées au texte normalisé.
- Regrouper les traductions MADAR d'un même `sentID.BTEC` dans une seule partition.
- Préserver les fichiers de test officiels TSAC et créer la validation à partir de ses seuls fichiers d'entraînement.
- Sélectionner paramètres, prompts, seuils et modèles sur validation ; réserver le test à l'évaluation finale.
- Utiliser le macro-F1 comme métrique principale, accompagné des mesures par classe et de matrices de confusion.
- Répéter les configurations finales stochastiques avec les graines 13, 42 et 2026, puis publier scores individuels, moyenne et écart-type.
- Consigner manifestes, empreintes de données, configuration, révision des modèles, matériel, durée, prédictions et erreurs.

## 6. Contraintes et exclusions

Le travail est réalisé seul sur quatorze semaines. Les modèles entraînés restent sous un milliard de paramètres ; aucune API payante n'est requise. Le développement se fait sur Apple Silicon et les entraînements plus lourds sur GPU CUDA, avec un code commun. Les données MADAR sont conservées hors du dépôt selon leurs conditions d'utilisation.

La couverture de tous les dialectes, l'audio, le déploiement commercial, l'entraînement d'un grand modèle à partir de zéro, le NER, la traduction et l'apprentissage actif sont exclus du socle. Une extension éventuelle ne sera étudiée qu'après validation du socle requis.

## 7. Preuves attendues au jalon J1

Le dossier de cadrage doit être accompagné d'un registre des sources et licences, d'une carte des données, d'un guide d'annotation, d'un schéma commun, d'un protocole de partitionnement et d'un plan de travail individuel réaliste. Le [protocole des données](02_data_protocol.md) précise la méthode retenue ; les cartes remplies et les manifestes restent à produire après l'audit.

## Sources de référence

- CAMeL Lab, [MADAR Parallel Corpus](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/).
- Bougares et collaborateurs, [TSAC](https://github.com/fbougares/TSAC).
- CAMeL Lab, [CAMeLBERT Mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix).
- Équipe Qwen, [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
