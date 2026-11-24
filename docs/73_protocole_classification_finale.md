# Protocole final de classification

Ce protocole du 12 novembre 2026 fixe les systèmes, les modèles sauvegardés, les entrées et les mesures de l’évaluation finale de sentiment TSAC et de variété MADAR. Les choix reposent sur les expériences de classification train/validation déjà réalisées. Les résultats du test documentaire ne servent pas à sélectionner les classifieurs. Le fichier `configs/classification_final_v1.json` enregistre les choix et les empreintes avant toute lecture des textes du test officiel.

## Systèmes retenus

| Groupe | Système | Texte et conditions | Répétitions |
| --- | --- | --- | --- |
| Principal | Classe majoritaire | Classe apprise sur le train v2 | Référence fixe par tâche |
| Principal | TF-IDF caractères et régression logistique | Texte brut, configuration caractères v2 conservée | Modèle déterministe conservé par tâche |
| Principal | BiGRU | Texte brut, vocabulaire appris sur train, longueur maximale de 128 unités | Trois graines par tâche |
| Principal | CAMeLBERT confirmé | Texte brut, tête CLS native, longueur maximale de 128 tokens | Trois graines par tâche |
| Principal | Qwen3-0.6B de base | Zero-shot et few-shot, consignes v2 et décodage fixés | Une génération fixée par condition et tâche |
| Principal | Qwen3-0.6B avec LoRA partagé | Mêmes conditions zero-shot et few-shot que la base | Trois adaptateurs, chacun commun aux deux tâches |
| Comparaison contrôlée | TF-IDF caractères normalisé | NFC et espaces, comparaison au même système brut | Modèle conservé par tâche |
| Comparaison contrôlée | TF-IDF mots | Texte brut, comparaison aux caractères | Modèle conservé par tâche |
| Comparaison contrôlée | CAMeLBERT CLS gelé et adapté de la campagne initiale | Même architecture et taux de 2e-5 dans les deux branches | Trois graines par branche et tâche |

Cela correspond à 32 conditions tâche/modèle principales et 16 conditions complémentaires, soit 48 conditions d’évaluation. Elles utilisent 36 modèles ou adaptateurs distincts dans le registre, avec une base Qwen partagée. Les conditions LoRA réemploient le même adaptateur d’une graine pour les deux tâches et les deux consignes. Une condition d’évaluation ne correspond donc pas à un nouvel entraînement.

Les références lexicales, la combinaison mots/caractères, la représentation gelée par moyenne des vecteurs et les essais de sélection restent documentés sur validation. La combinaison n’ayant pas amélioré la référence caractères, elle n’est pas ajoutée aux systèmes retenus pour le test final. La référence de modèle de langue caractères conserve son évaluation par perplexité, séparée de la classification.

## Modèles et choix figés

CAMeLBERT confirmé utilise les modèles déjà sauvegardés après sélection sur validation. Les taux sont 5e-5 pour TSAC et 2e-5 pour MADAR. Les trois graines 13, 42 et 2026 sont conservées avec leurs checkpoints sélectionnés par macro-F1 de validation. La campagne initiale appariée à 2e-5 reste séparée de cette confirmation pour mesurer le contraste gelé/adapté à architecture identique.

Les adaptateurs LoRA sont ceux entraînés sur Kaggle pendant deux époques, avec un taux de 1e-4, lot physique de deux et accumulation de huit. Le checkpoint final correspond à la deuxième époque fixée avant apprentissage. Les trois graines et les deux consignes sont retenues. Le choix ne remplace pas la campagne par la graine la plus favorable.

Chaque entrée du registre indique le checkpoint, ses fichiers, la configuration effective et sa source de validation. Le registre inclut les poids, mappings et tokenizers nécessaires. Les modèles préentraînés sont liés à leurs révisions exactes. Les configurations, les verrouillages des environnements, les fonctions d’inférence partagées et leurs dépendances Python locales sont également liés par empreintes.

## Données et partitions

La sélection utilise toutes les lignes `test` des manifestes v2, avec 3 400 avis TSAC et 800 phrases MADAR. Les labels sont `NEG`, `POS` pour TSAC et `MSA`, `TN`, `EG` pour MADAR. L’ordre de ces labels est fixé dans les matrices et les rapports. Les lignes, leurs étiquettes et les anomalies du test officiel sont conservées pour le score principal.

Le contrôle du protocole lit les manifestes sans texte pour enregistrer effectifs, groupes et labels. Il ne charge aucun texte brut du test et ne produit aucune prédiction. L’évaluation ultérieure devra appeler explicitement le chargeur avec `allow_test=True` après vérification du protocole. Chaque système d’une tâche reçoit les mêmes exemples, avec identifiants et groupes contrôlés. Aucune sélection ne dépend du dialecte prédit par un autre composant.

Les transformations restent en mémoire et le texte source est conservé. Les branches brutes gardent `raw-v0`. La comparaison de normalisation utilise seulement `norm-nfc-space-v1`. Les règles de tokenisation et de troncature sont celles des modèles déjà entraînés.

## Inférence et lecture des sorties

Les réseaux sont évalués en mode `eval`, en FP32, avec attention eager et sans quantification. Le dispositif choisit CPU, MPS ou CUDA à l’exécution. Les conditions Qwen appariées doivent utiliser le même appareil et les mêmes versions logicielles afin de limiter le facteur matériel. L’appareil réellement utilisé, les versions et les empreintes sont enregistrés.

Qwen reprend les consignes v2, le gabarit officiel sans réflexion et les démonstrations fixes sélectionnées uniquement dans le train. Le budget est de 480 tokens pour le prompt complet et 32 tokens de continuation. La génération est gloutonne, avec paramètres explicites. Le parseur accepte uniquement un label exact après retrait des espaces extérieurs et du délimiteur de fin prévu.

Une réponse Qwen invalide ou un prompt dépassant le budget reçoit `INVALID`, avec un motif enregistré. La ligne reste dans les dénominateurs des mesures. La colonne `INVALID` reste dans la matrice, mais elle n’ajoute pas une classe artificielle à la moyenne macro-F1. Nous ne modifions pas le budget après inspection des résultats de test.

Les autres classifieurs conservent leurs prédictions sur toutes les lignes. Un vecteur TF-IDF nul est signalé pour le diagnostic, tout en conservant la décision du modèle dans le score principal. Le traitement scientifique ne reprend pas automatiquement les règles d’abstention de l’interface. Une étude d’abstention des classifieurs devra annoncer séparément ses seuils et sa couverture.

## Mesures et comparaisons

Le macro-F1 est la mesure principale, calculée sur la liste fixe des classes de chaque tâche. Le traitement d’une classe sans prédiction vaut `zero_division=0`. Nous conservons accuracy, balanced accuracy, précision, rappel et F1 par classe, matrice de confusion et taux de labels valides pour les systèmes génératifs.

Les systèmes stochastiques présentent les trois scores et leur moyenne avec écart-type échantillonnal. Les références déterministes et la génération de base fixée sont présentées séparément. Les budgets d’apprentissage sont renseignés, notamment le SFT LoRA de 4 000 exemples, sans prétendre à une comparaison d’architectures à budget égal.

Le contraste principal compare LoRA à Qwen de base dans chaque tâche et chaque condition de consigne. Les contrastes complémentaires comparent CAMeLBERT confirmé aux caractères bruts, la campagne CLS adaptée à sa branche gelée, le texte normalisé au brut et les mots aux caractères. Toutes les différences reposent sur des prédictions appariées.

Le bootstrap prévoit 1 000 rééchantillonnages avec graine 42. L’unité est le `group_id` du manifeste, avec tous les exemples du groupe conservés ensemble. Les mêmes groupes rééchantillonnés servent aux deux systèmes et aux trois graines d’un contraste. La statistique est la différence des macro-F1 moyens sur les graines, ou la différence directe des références fixes. L’intervalle descriptif retenu est l’intervalle percentile à 95 %. L’indépendance par auteur ou conversation ne peut pas être déduite du seul identifiant de groupe.

Les analyses par écriture utilisent `AR`, `LATIN`, `MIXED`, `UNKNOWN`, avec avertissement lorsque l’effectif est inférieur à 100. Les classes de longueur en caractères sont fixées avant test à 0–20, 21–50, 51–100 et plus de 100. L’écriture ne constitue pas une identification de langue.

## Ressources et sorties

Chaque modèle sera évalué dans un processus isolé pour distinguer les coûts. Les temps d’inférence, effectifs, paramètres, tailles des fichiers et mesures mémoire sont consignés. Deux lots de chauffe précèdent une mesure de latence sur 100 exemples par tâche, sélectionnés par classement SHA-256 de `42|id`, avec lot unitaire. Médiane et percentile 95 sont enregistrés après synchronisation de l’appareil. Sur Apple Silicon, RSS CPU et instantanés MPS sont présentés séparément.

Les prédictions individuelles restent dans `outputs/final_classification/<run>/`, avec identifiant, groupe, vrai label, prédiction, modèle, graine, variante, mode de consigne et motifs d’invalidité ou de troncature. Les rapports publics sous `reports/final_classification/<run>/` contiennent les agrégats générés, sans textes protégés. Le protocole, le code d’exécution, les modèles, les données, les configurations, les versions, le matériel et les durées sont liés à chaque run.

## Vérification du gel

Le programme `scripts/classification_protocol.py` produit le registre puis contrôle les fichiers conservés. Une version existante est refusée lors de la création. Pour vérifier le protocole depuis la racine avec l’environnement classique préparé

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/classification_protocol.py check \
  --protocol configs/classification_final_v1.json
```

Le contrôleur refuse un artefact modifié, une tâche incohérente ou une sélection incomplète des graines. Cette préparation fixe les choix pour **L01**, **L07**, **L08** et **L12**. L’exécution et la validation technique du futur évaluateur de classification constituent l’étape suivante, avec le protocole conservé. Aucun texte ni score du test officiel n’est utilisé pendant ce gel.
