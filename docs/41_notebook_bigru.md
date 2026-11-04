# Notebook BiGRU : apprentissage guidé, résultats et place dans le projet

**Date :** 4 novembre 2026. **Contributions partielles à L04, L07 et L12.** Nous ajoutons `notebooks/03_bigru.ipynb` pour expliquer et relancer les scripts de la référence neuronale, sans dupliquer les couches ou la boucle d'apprentissage dans les cellules. Les sorties personnelles enregistrées ensuite dans le fichier sont préservées.

## Fonction du notebook

Le notebook présente les données d'entrée, l'embedding, la GRU bidirectionnelle, la moyenne masquée, le dropout et la couche de classification. Il orchestre les mêmes scripts que l'expérience initiale, affiche les logs d'époque, un tableau des six résultats, la moyenne et l'écart-type par tâche, puis les six figures.

La logique scientifique reste dans `bigru_data.py`, `bigru_model.py`, `bigru_training.py`, `run_bigru.py` et `verify_bigru.py`. Le nouveau module `bigru_notebook_support.py` utilise la bibliothèque standard pour lancer ces scripts dans le profil PyTorch isolé, puis sélectionner explicitement le bon run. Le notebook et la ligne de commande utilisent ainsi la même implémentation.

## Modes explicites

| Mode | Effet |
| --- | --- |
| `full`, valeur par défaut | Nouveau smoke vérifié, puis six entraînements : deux tâches et trois graines sur tout le train/validation |
| `replay` | Rechargement et vérification d'un run full explicitement choisi, sans nouvel apprentissage |
| `smoke` | Deux petits entraînements techniques, 100 lignes par partition et une graine ; pas une expérience complète |
| `off` | Aucune exécution de script, aucun chargement de corpus ou de modèle |

La première cellule affiche le mode effectif. Une éventuelle variable `TUNISCOPE_BIGRU_MODE` est signalée ; elle ne modifie pas le mode silencieusement. Un smoke manquant ou en échec empêche le full. Les tableaux et les figures du full proviennent de son dossier, jamais du premier contrôle smoke ni d'un dossier « dernier run ».

Chaque nouvelle exécution d'apprentissage crée un dossier unique sous `outputs/neural_baselines/notebooks/`. Les figures sont exportées dans son sous-dossier `figures/`. Un export existant est réutilisé seulement si l'empreinte du `results.json`, la liste attendue des PNG et toutes leurs empreintes correspondent. Une incohérence provoque une erreur visible, sans écrasement ou substitution.

## Environnement et utilisation

Ouvrir le notebook dans le noyau **TuniScope (notebooks)** déjà utilisé pour les notebooks 01 et 02. Les calculs PyTorch sont exécutés dans `environments/bigru/.venv/` via `scripts/bigru_uv.sh`, avec `--locked --offline`. Le noyau ne doit pas importer PyTorch ; il utilise IPython pour afficher les images. L'environnement du noyau n'est pas transmis comme environnement virtuel de l'apprentissage.

Avant la première utilisation, le profil doit être installé avec `sh scripts/bigru_uv.sh sync --locked`, comme décrit dans le [guide d'environnement](39_environnement_bigru.md). Les corpus autorisés doivent être aux chemins locaux attendus. Aucun téléchargement de corpus ou de poids préentraînés n'est déclenché par les cellules.

Les cellules de code réalisent successivement :

1. Repérage du dépôt et choix du mode.
2. Apprentissage ou relecture, vérification et tableau lisible des résultats.
3. Export ou contrôle des PNG du même run et affichage du chemin exact.
4. Affichage des images et bilan explicite du mode.

Le défaut `full` relance réellement l'apprentissage. Pour relire l'expérience initiale, définir `TRAINING_MODE = 'replay'` et choisir le chemin `REPLAY_RUN` dans la première cellule. La répétition de la cellule d'apprentissage crée une nouvelle expérience ; les paramètres et la règle de sélection restent ceux de la configuration figée.

## Rôle de la BiGRU dans l'étude

La BiGRU est une référence neuronale obligatoire de L04, apprise depuis zéro. Elle doit être conservée dans les résultats, y compris lorsqu'elle est moins performante que TF-IDF. Elle fournit un point de comparaison entre les approches classiques et les systèmes préentraînés qui seront expérimentés ensuite.

Elle ne constitue pas une étape de transformation des poids vers un Transformer ou un LLM. Les futurs modèles auront leurs propres poids, tokenizers et expériences. Ce qui est réutilisé est le protocole : partitions, labels, métriques, contrôles, sauvegardes, rechargement et analyse d'erreurs. Une comparaison entre architectures ne suffit pas à isoler causalement l'effet du préentraînement.

L'intégration de la BiGRU à l'application est une tâche distincte : son exécution dans le notebook ne change pas les modèles servis par FastAPI. Nous n'abandonnons pas la référence expérimentale et nous ne la présentons pas automatiquement comme meilleur système utilisateur.

## Vérifications et limites

Le contrôle `scripts/check_bigru_notebook.py` valide la structure du fichier et peut exécuter séquentiellement les cellules. Pour un notebook contenant déjà nos sorties, l'option ci-dessous vérifie les sources sur une copie en mémoire sans effacer les résultats sur disque :

```sh
sh scripts/project_uv.sh run --locked --offline python -m unittest discover -s tests -p test_bigru_notebook.py
sh scripts/project_uv.sh run --locked --offline python scripts/check_bigru_notebook.py --allow-saved-outputs --execute --mode off
```

Sans `--allow-saved-outputs`, le contrôle refuse les sorties enregistrées : c'est un contrôle de nettoyage avant partage, pas un défaut du code. L'exécution séquentielle n'est pas à elle seule une preuve de rendu Jupyter.

Le 4 novembre, les 11 nouveaux tests du bridge/notebook réussissent. Le noyau réel `TuniScope (notebooks)` a exécuté le smoke puis les six apprentissages full sous `outputs/neural_baselines/notebooks/514ac0478d0c48528f2793d8f658d633_bigru_full/`. Les six métriques et empreintes de prédictions sont identiques à la référence initiale.

Le premier parcours natif a ensuite échoué à l'export des figures : le processus enfant héritait de `MPLBACKEND=module://matplotlib_inline.backend_inline`, composant absent du profil PyTorch. Le bridge et l'exporteur fixent désormais `Agg` pour le processus qui génère les PNG, sans ajouter cette dépendance ni modifier les entraînements. L'export a été rejoué avec la même variable héritée pour vérifier la correction. Un second noyau natif a exécuté les quatre cellules en mode replay sur ce run, avec six images affichées et aucun nouvel apprentissage. Ce sont deux parcours successifs, pas un full ininterrompu revendiqué après coup. La preuve de relecture est conservée dans `outputs/notebook_checks/2026-11-04/bigru_native_replay_after_fix.json`.

Des sorties personnelles sont apparues dans le notebook pendant la vérification ; nous les conservons et validons les sources sur une copie en mémoire. Aucun fichier de notebook précédent ni session utilisateur n'a été modifié pour nettoyer les sorties.

Les scores restent des métriques de validation, sans ouverture du test officiel. L04/L07/L12 restent globalement partiels : le notebook ne clôture pas les relectures humaines, les autres modèles, la relance indépendante ou les preuves finales.
