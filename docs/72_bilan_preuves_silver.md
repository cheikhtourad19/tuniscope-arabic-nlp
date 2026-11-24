# Bilan des preuves du jalon Silver

Le 12 novembre 2026, nous avons consolidé les résultats et les fichiers conservés des expériences classiques, neuronales et documentaires. Le programme `scripts/report_silver_evidence.py` vérifie les artefacts disponibles, recalcule les métriques depuis les prédictions ou les matrices et rassemble les réserves. Le [tableau central généré](../reports/silver_evidence_2026-11-12/README.md) présente les scores de classification sur la validation v2. Le [bilan structuré](../reports/silver_evidence_2026-11-12/summary.json) contient les ablations, les statistiques, l’inventaire des configurations et les empreintes.

## Preuves techniques disponibles

| Élément du socle Silver | Preuve contrôlée | État technique |
| --- | --- | --- |
| Références classiques | Majorité, règles, TF-IDF mots, caractères et combinaison sur validation v2 | Résultats et prédictions conservés |
| Modèle de langue statistique et BiGRU | Perplexités caractères vérifiées, BiGRU sur trois graines | Références L04 présentes |
| Encodeur adapté | Confirmation CAMeLBERT sur trois graines par tâche, six checkpoints et preuves de rechargement | Preuve de L05 obtenue |
| Adaptateur partagé | Trois adaptateurs LoRA, chacun commun à TSAC et MADAR, poids et contrôles de rechargement | Preuve de L06 obtenue |
| Recherche documentaire | BM25 et E5 évalués sur validation puis sur 40 questions documentaires après gel technique | Comparaison exécutée avec références provisoires |
| Répétitions | Graines 13, 42 et 2026, moyennes et écarts-types recalculés | Statistiques vérifiées |
| Ablations principales | Transformations, représentations, encodeur gelé/adapté, contexte et abstention documentaire | Comparaisons réunies dans le bilan |

L’audit lie 245 fichiers de preuve par empreinte, en complément des vérifications internes de campagne réutilisées. Il recalcule les statistiques de 14 groupes de résultats à trois graines. Plusieurs groupes LoRA correspondent à des évaluations du même adaptateur, selon la tâche et la consigne. Ces groupes ne représentent donc pas 14 campagnes d’entraînement indépendantes. L’inventaire des 21 configurations présentes comprend les configurations historiques et actuelles. Il ne constitue pas encore le protocole global d’ouverture du test de classification.

Les scores de classification des références neuronales sont contrôlés sur les mêmes identifiants, groupes et labels de validation que la référence caractères lorsque les prédictions individuelles sont présentes localement. Pour LoRA, le contrôle local repose sur les matrices, les effectifs et les preuves d’alignement enregistrées lors de l’exécution CUDA. Les prédictions individuelles CUDA ne sont pas présentes dans les archives reçues.

Les budgets d’apprentissage diffèrent entre familles. Le corpus SFT Qwen comprend 4 000 exemples, contre l’ensemble du train v2 pour les classifieurs principaux. Le tableau mesure le comportement des systèmes réellement étudiés. Il ne démontre pas une supériorité intrinsèque d’architecture à budget d’apprentissage égal.

## Comparaisons et ablations

| Lot | Contraste | Source de preuve | Interprétation conservée |
| --- | --- | --- | --- |
| A | Texte brut contre NFC et espaces | Baselines caractères v2 | Effet très faible dans cette expérience |
| B | Mots, caractères et combinaison, avec règles lexicales documentées séparément | Comparaison v2 et baselines à règles | Les caractères restent la meilleure référence classique étudiée |
| C | CAMeLBERT CLS gelé contre adapté | Campagne appariée initiale, trois graines par branche et tâche | La même architecture CLS est conservée entre les branches |
| D | Réponse Qwen avec fiches E5 contre contexte vide | Validation documentaire sur 20 questions | Échec des citations et limites factuelles documentés, réponse extractive retenue |
| E | Abstention documentaire activée contre désactivée | Analyse des classements déjà sauvegardés des 20 questions de validation | Compromis entre couverture et pertinence des réponses émises |

Le contraste C utilise la campagne initiale appariée à 2e-5. La confirmation finale utilise les taux sélectionnés pour chaque tâche. La référence gelée par moyenne des vecteurs est également conservée, mais elle emploie une représentation différente. Ces expériences sont identifiées séparément pour éviter d’attribuer au seul fine-tuning un effet qui pourrait aussi dépendre de la représentation ou du réglage.

L’analyse E retire seulement la décision d’abstention des classements enregistrés. Elle conserve le premier passage et sa langue. Elle n’ajuste aucun seuil et ne lance aucune inférence. Les références pertinentes restent les propositions déjà utilisées sur validation. Cette analyse documentaire ne remplace pas une étude de calibration et d’abstention des classifieurs de sentiment.

## Rechargement, métriques et traçabilité

Les empreintes des checkpoints et fichiers de prédictions accessibles sont contrôlées. Le vérificateur existant de confirmation encodeur contrôle aussi les sources, les partitions, les artefacts et les preuves de rechargement déjà enregistrées. Les scores Qwen de base sont reconstruits depuis les générations conservées, avec les sorties invalides comptées comme erreurs. Les scores et écarts-types LoRA sont recalculés depuis les matrices des trois graines.

Le contrôle des preuves de rechargement est distinct d’une nouvelle inférence. Cette consolidation charge des métadonnées et des fichiers de résultats, vérifie des empreintes et recalcule des mesures. Elle ne relance aucun modèle. Les données brutes, les vocabulaires, les checkpoints et les prédictions individuelles restent dans leurs dossiers privés.

## Éléments encore ouverts

| Élément | Situation précise | Suite attendue |
| --- | --- | --- |
| Revue indépendante du RAG | Deux grilles proposées par le même assistant, aucune double évaluation humaine attestée | Réaliser la revue ou faire accepter l’adaptation au travail individuel |
| Traçabilité LoRA CUDA | Poids et matrices disponibles, prédictions individuelles et résumé original de campagne non reçus | Récupérer les fichiers originaux si disponibles, conserver la limite autrement |
| Commit Qwen de base | Le plan conserve les empreintes du code, mais aucun champ de commit Git | Déclarer cette limite sans attribuer rétrospectivement un commit supposé |
| Historique de l’encodeur initial | Pertes et macro-F1 par époque présents, détails par classe de toutes les époques non conservés | Garder la réserve historique, utiliser la confirmation complète pour les preuves finales |
| Sauvegarde indépendante | Artefacts locaux contrôlés, sauvegarde sur un autre support non confirmée | Confirmer une copie indépendante des poids et des fichiers indispensables |
| Reproduction par un autre opérateur | Contrôles locaux et preuves de rejeu disponibles | Réaliser la relance indépendante attendue pour L12 |
| Protocole global de classification | Configurations et résultats inventoriés, test officiel encore réservé | Définir les systèmes et contrastes retenus avant son ouverture |

Les limites d’annotation des corpus, déjà décrites dans les cartes de données, restent également déclarées. Cette consolidation ne produit aucun accord interannotateurs pour les données ou le RAG. Les éléments liés au test final, au bootstrap, aux jeux de défi, au rapport final et à la reproduction complète appartiennent encore aux étapes d’évaluation et de remise du projet.

## Reproduire le bilan

Depuis la racine, avec les artefacts locaux conservés, choisir un nouveau dossier de rapport.

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/report_silver_evidence.py \
  --output reports/ma_consolidation_silver
```

Les tableaux sont générés depuis les résultats contrôlés. La commande refuse une destination existante. Le bilan apporte des preuves pour **L04**, **L05**, **L06**, **L07**, **L08** et **L12**. Il constitue un dossier technique de préparation de J4. La clôture complète de Silver reste soumise aux réserves de revue et de traçabilité explicitées ci-dessus.
