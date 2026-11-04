# Baseline neuronale légère : embeddings appris et BiGRU

**Date :** 4 novembre 2026. **L04, L07, L10 et L12 : contributions partielles.** Nous démarrons la référence neuronale légère après la reprise explicitement demandée. Cela n'autorise pas le téléchargement ou l'entraînement des futurs encodeur et LLM.

## Question et architecture

Nous mesurons un classifieur séquentiel entraîné depuis zéro, sans poids préentraînés. Une matrice d'embeddings associe à chaque token un vecteur de 64 nombres appris ; une GRU bidirectionnelle de taille cachée 64 lit la séquence dans les deux sens. La moyenne des états correspondant aux vraies unités, hors padding, donne 128 valeurs. Un dropout de 0,2 puis une projection linéaire produisent deux logits pour TSAC et trois pour MADAR.

Le padding complète les lots, mais ne représente pas du texte. Nous utilisons des séquences compactées pour que la direction inverse ne lise pas ces positions artificielles, puis un masque pour les exclure de la moyenne. La couche d'embedding réserve l'identifiant zéro au padding ; l'identifiant un est réservé aux unités inconnues. La sortie est une classification, pas une génération libre.

## Données et représentation fixées avant apprentissage

Configuration : `configs/bigru_v1.json`. Manifestes v2, mêmes partitions et labels que les références classiques ; texte `raw-v0`, aucune nouvelle exclusion et aucun accès au test officiel. Les deux tâches disposent de modèles séparés et ne filtrent pas leurs exemples selon la prédiction d'une autre tâche.

Le tokenizer `unicode-word-symbol-v1` extrait des suites alphanumériques Unicode, dont les chiffres Arabizi, et conserve séparément les symboles, ponctuations et emojis. Il ne modifie pas la casse et ne translittère pas. Il ne constitue pas un analyseur morphologique arabe : affixes, marques combinantes et séquences emoji complexes peuvent être segmentés imparfaitement.

Le vocabulaire, construit sur le train seulement, retient les unités de fréquence au moins deux, avec un plafond de 20 000 entrées, marqueurs compris. Les égalités sont résolues par ordre lexical. Les unités exclues, y compris les unités rares du train, sont mappées vers UNK. Un texte sans unité reçoit un UNK, sans exclusion silencieuse.

La longueur maximale est 128 tokens. Nous conservons le début du texte en cas de dépassement et enregistrons le nombre de lignes et d'unités tronquées dans chaque partition. Les sources restent intactes ; ce choix n'efface pas les textes longs de l'évaluation.

## Audit préalable du 4 novembre

L'audit `scripts/audit_bigru_inputs.py`, exécuté avant apprentissage, est sauvegardé sous `outputs/bigru_input_audit_2026-11-04.json` avec empreintes de configuration et manifestes. Il ne publie aucun vocabulaire ou texte protégé.

| Corpus | Vocabulaire, PAD/UNK compris | Train | Validation | P95 tokens train | Lignes tronquées train / validation | Tokens UNK validation avant troncature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSAC | 6 702 | 10 053 | 1 774 | 23 | 8 / 1 | 4 171 / 14 144 : 29,49 % |
| MADAR | 4 016 | 6 388 | 800 | 14 | 0 / 0 | 1 286 / 5 911 : 21,76 % |

Le nombre élevé d'UNK est une limite attendue d'un vocabulaire lexical sur des variantes orthographiques ; ce n'est pas un taux de mots inconnus de la langue. Les classes et groupes restent ceux des manifestes v2. Les données de validation ne servent pas à enrichir le vocabulaire après cet audit.

## Apprentissage et sélection

- Une seule configuration initiale ; pas de recherche après lecture des scores.
- Graines **13, 42 et 2026**, pour chaque tâche. Même vocabulaire, mêmes données et même budget pour chaque graine.
- Lots de 64, Adam, taux d'apprentissage 0,001, sans weight decay, clipping de norme des gradients à 1.
- Entropie croisée sans pondération des classes, comme choix initial documenté ; la métrique de sélection reste le macro-F1 à labels fixes.
- Au maximum 12 époques, arrêt après trois époques consécutives sans hausse stricte du macro-F1 de validation. En cas d'égalité, le premier checkpoint est conservé.
- Chaque époque lit tout le train du mode choisi. `model.train()` et gradients pour le train ; `model.eval()` et `torch.inference_mode()` pour la validation.
- Perte train moyenne par exemple pendant l'apprentissage et perte validation après l'époque ; ces valeurs ne proviennent pas exactement du même état du modèle et le dropout est actif seulement pendant l'apprentissage.
- Sauvegarde du meilleur `state_dict`, tokenizer/vocabulaire, labels, configuration, historique, prédictions de validation, métriques par classe/écriture et courbes PNG.

Le smoke utilise 100 exemples par partition/corpus, la graine 42 et deux époques. Il construit son vocabulaire sur son seul train. Il doit montrer pertes finies, gradients non nuls et poids modifiés, puis passer le rechargement et la vérification avant tout full.

Le choix initial du matériel est **CPU, deux threads**, pour une première référence locale mesurée. Le code offre une détection CPU/MPS/CUDA et utilise `.to(device)` ; les appareils indisponibles ou les opérations non supportées produisent une erreur visible, sans repli silencieux. Une vérification CPU ne prouve pas une exécution MPS ou Kaggle CUDA.

## Vérification et comparaison

Les tests fictifs couvrent les inconnus sans fuite, le padding, la lecture bidirectionnelle, l'absence d'effet des positions complétées, les gradients, la mise à jour des poids, le rechargement et la répétabilité locale d'une courte exécution CPU.

Le vérificateur recharge tous les checkpoints retenus et rejoue toutes les prédictions de validation ; il reconstruit le vocabulaire depuis le train et recalcule les métriques à partir des labels/prédictions. Il vérifie l'alignement avec les exemples de la référence TF-IDF caractères v2. Les graines sont toutes présentées avec moyenne et **écart-type d'échantillon** ; aucune meilleure graine n'est sélectionnée pour représenter le système.

La comparaison BiGRU/mots et TF-IDF/caractères est une comparaison de systèmes complets : représentation et classifieur changent ensemble. Elle n'isole pas causalement l'effet de la récurrence. Les trois graines partagent la même validation utilisée pour l'arrêt ; elles ne constituent ni trois tests indépendants ni un intervalle d'incertitude sur la population.

## Environnement et frontières

Le profil `environments/bigru/` possède son propre `pyproject.toml`, `uv.lock` et export de dépendances. L'environnement classique et son verrouillage restent inchangés. Aucun poids de modèle n'est téléchargé ; les paramètres de cette référence sont initialisés aléatoirement et appris sur le train.

Les checkpoints, vocabulaires et prédictions restent sous `outputs/neural_baselines/`, ignoré par Git. Les rapports partagés ne contiennent que des statistiques agrégées. Les runs précédents sont conservés et aucun checkpoint applicatif n'est remplacé automatiquement.

Le caractère déterministe demandé à PyTorch réduit des sources de variabilité locales ; il ne garantit pas une identité bit à bit entre versions, systèmes ou appareils. La relance propre sur Kaggle, l'évaluation finale, la relecture humaine des erreurs et les autres composants de L04/L07/L12 restent des preuves distinctes.

## Références techniques

- [PyTorch — GRU](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html) : sorties et bidirectionnalité.
- [PyTorch — pack_padded_sequence](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.rnn.pack_padded_sequence.html) : longueurs et lots de séquences.
- [PyTorch — Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html) : graines et limites de répétabilité.
- [PyTorch — MPS backend](https://docs.pytorch.org/docs/2.15/notes/mps.html) : disponibilité et choix de l'appareil.
