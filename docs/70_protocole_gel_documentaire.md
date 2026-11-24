# Gel du protocole documentaire avant évaluation

Le 12 novembre 2026, nous retenons la politique applicative décrite dans le document 69 pour comparer BM25 et E5 sur les 40 questions du test documentaire. Les réglages proviennent du développement sur les 20 questions de validation et des sondes applicatives déjà déclarées. Le gel enregistre les empreintes du code, du catalogue, des annotations proposées, des sources documentaires, des seuils et du verrou de dépendances avant le calcul des scores de test.

## Périmètre et limite de référence

Le catalogue comprend 44 fiches en français et en arabe standard. L’auteur a marqué les 44 versions conformes lors de la revue éditoriale. Les annotations de pertinence des questions restent proposées par IA. Les deux grilles préremplies proviennent du même assistant et ne constituent pas deux jugements humains indépendants. Aucune dispense pédagogique n’est confirmée à cette étape.

Nous poursuivons l’évaluation technique avec cette limite explicitement enregistrée. Les scores sont conditionnés par ces références provisoires. Ils ne suffisent pas pour déclarer la conformité complète de Silver. Toute annotation humaine ultérieure devra être conservée séparément et donner lieu à une version distincte des références, sans remplacer les premières mesures.

L’échantillon de revue déjà publié comprend aussi des questions du test documentaire. Leurs identifiants sont consignés dans le gel. Les questions sont déjà présentes dans le dépôt de travail. Nous ne revendiquons donc pas que les 40 formulations sont inconnues de l’auteur du système. Aucun score de ces 40 questions n’a servi à choisir les réglages de cette version gelée. La séparation documentaire ne doit pas être confondue avec le test officiel de classification TSAC/MADAR, qui reste hors du périmètre de cet essai.

## Moteurs, réponses et mesures

Les deux moteurs recherchent dans les fiches de la langue indiquée par la question. Ils conservent les trois premiers rangs avant abstention. BM25 reprend sa politique de questions courtes et son critère d’écart entre les premiers scores. E5 reprend la révision officielle épinglée et son seuil calibré. Les valeurs exactes et les empreintes sont consignées dans `configs/rag_documentary_frozen_v1.json`.

Le système répond en reproduisant le texte de la première fiche avec son identifiant et ses références, ou signale une insuffisance de sources. Il ne reçoit ni la prédiction du classifieur ni son état interne. Ses citations documentent la méthode du projet. Elles n’établissent aucune explication causale d’une prédiction particulière.

Les réponses génératives Qwen avec et sans contexte ont déjà été comparées sur validation, avec leurs échecs factuels et de citation dans le document 67. Conformément à la solution de repli prévue par le cahier des charges, l’évaluation applicative présente porte sur la réponse extractive sourcée.

Nous calculons Recall@3 et MRR@3 sur les questions documentées, la couverture des réponses, la proportion de citations pertinentes parmi les réponses, les refus sur questions sans réponse et les refus sur questions documentées. Les résultats sont présentés globalement et séparément pour le français et l’arabe. Nous vérifions que chaque passage émis est identique à la fiche citée. Cette égalité vérifie la fidélité de l’extraction au catalogue. La validité factuelle du catalogue et la pertinence des références restent soumises aux limites de revue déclarées.

## Protection du gel

Le programme `rag/final_evaluation.py` vérifie les empreintes avant de charger les modèles, puis à nouveau avant d’enregistrer le résultat. Il refuse une destination existante. Les questions, références, classements et réponses sont enregistrés pour les deux moteurs. Les réponses sont rejouées par le point d’entrée applicatif et les mesures sont recalculées après relecture du JSON sauvegardé.

Après lecture des scores de test, nous conservons cette version et ses limites. Une modification motivée par ces résultats devra être identifiée comme développement après consultation du test, avec un nouveau protocole et une nouvelle question d’évaluation. Le jeu actuel ne retrouverait pas son statut de mesure finale indépendante par une simple relance.

Cette procédure contribue à **L07**, **L08**, **L09** et **L12**. Le gel technique et les vérifications de sortie apportent des preuves reproductibles. La double revue humaine et l’acceptation d’une adaptation au travail individuel restent distinctes de l’exécution technique.
