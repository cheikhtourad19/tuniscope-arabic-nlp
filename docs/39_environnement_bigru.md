# Environnement isolé de la référence BiGRU

**Date :** 4 novembre 2026. **L12 partiel**, support de L04/L07. Nous installons un profil propre à la baseline neuronale légère ; le profil classique racine et le backend applicatif conservent leurs environnements.

## Organisation

| Profil | Déclaration | Environnement installé | Usage |
| --- | --- | --- | --- |
| Classique | `pyproject.toml`, `uv.lock` | `.venv/` | Données, TF-IDF, règles, modèle de langage caractères |
| BiGRU | `environments/bigru/pyproject.toml`, `uv.lock` | `environments/bigru/.venv/` | Audit et apprentissage neuronal léger |
| Application | `App/backend/pyproject.toml`, `uv.lock` | `App/backend/.venv/` ou image Docker | Service d'inférence classique |

Les environnements installés sont ignorés par Git. Les déclarations, verrouillages et exports sont versionnés. Aucun environnement n'est copié du Mac vers Kaggle : il faut reconstruire les paquets de la plateforme visée.

## Versions locales vérifiées

Python 3.12.15, PyTorch **2.15.0**, NumPy **2.5.4**, scikit-learn **1.9.1**, Matplotlib **3.10.7**. Le profil est installé sur macOS ARM64. L'exécution initiale utilise **CPU, deux threads et FP32** ; l'export des courbes fonctionne sans fenêtre graphique via le backend Matplotlib Agg.

Le verrouillage couvre macOS ARM64 et Linux x86-64. Sa résolution Linux comporte les dépendances CUDA de la distribution PyTorch choisie ; leur présence dans un lockfile ne prouve pas que le pilote de la future session Kaggle est compatible. Nous vérifierons matériel, pilote et distribution avant une installation GPU sur Kaggle. Aucune exécution CUDA ou MPS n'est revendiquée par la vérification CPU locale.

Les mêmes scripts utilisent `.to(device)` avec détection `cpu`, `mps`, `cuda` ou `auto`. Une demande d'appareil indisponible échoue explicitement. Le choix CPU initial est enregistré dans le run ; il n'est pas un repli silencieux depuis le GPU. Il n'y a ni API payante ni téléchargement de poids préentraînés.

## Installation et contrôles

Depuis la racine :

```sh
sh scripts/bigru_uv.sh sync --locked
sh scripts/bigru_uv.sh run --locked --offline python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())'
sh scripts/project_uv.sh run --locked --offline python -m unittest discover -s tests -p test_bigru_data.py
sh scripts/bigru_uv.sh run --locked --offline python -m unittest discover -s tests/neural -p test_bigru.py
```

Les cinq tests d'entrée utilisent la bibliothèque standard. Les huit tests neuronaux utilisent PyTorch et des textes inventés : indépendance du padding, gradients, modification des poids, validation sans modification, baisse de perte sur un problème minuscule, répétabilité CPU locale, rechargement, contrôle de smoke et rejet de métriques altérées.

L'export pip s'obtient sans édition manuelle :

```sh
sh scripts/bigru_uv.sh export --locked --no-dev --format requirements-txt --output-file environments/bigru/requirements.txt
```

L'installation de ce profil ne prépare pas automatiquement Transformers, PEFT ou les futurs modèles. Leurs révisions, longueurs et compatibilités restent à définir dans leurs phases respectives.

## Exécution contrôlée

Chaque sortie doit avoir un nouveau nom. Les commandes ne remplacent jamais un ancien run.

```sh
sh scripts/bigru_uv.sh run --locked --offline python scripts/run_bigru.py --mode smoke --output outputs/neural_baselines/NOUVEAU_SMOKE
sh scripts/bigru_uv.sh run --locked --offline python scripts/verify_bigru.py outputs/neural_baselines/NOUVEAU_SMOKE
sh scripts/bigru_uv.sh run --locked --offline python scripts/run_bigru.py --mode full --smoke-run outputs/neural_baselines/NOUVEAU_SMOKE --output outputs/neural_baselines/NOUVEAU_FULL
sh scripts/bigru_uv.sh run --locked --offline python scripts/verify_bigru.py outputs/neural_baselines/NOUVEAU_FULL --reference-run outputs/baselines/2026-10-17_grouped_v2_full --report reports/NOUVEAU_RAPPORT_BIGRU
sh scripts/bigru_uv.sh run --locked --offline python scripts/plot_bigru_run.py outputs/neural_baselines/NOUVEAU_FULL
```

Le full entraîne deux tâches avec les trois graines configurées. `--device` peut être explicite ; le smoke préalable doit avoir la même configuration et le même appareil. Les prédictions et vocabulaires restent privés.

`verify_bigru.py` ne relance pas l'apprentissage. Il reconstruit le vocabulaire depuis le train, recharge les modèles et rejoue leur validation. Avec `--reference-run`, il exige les mêmes identifiants/groupes/labels que la référence TF-IDF caractères explicitement fournie et recalcule son score depuis ses prédictions. Sans cette option, il ne revendique pas une comparaison appariée avec un artefact externe.

`plot_bigru_run.py` vérifie d'abord le run, puis produit un PNG par modèle dans son sous-dossier `figures/`. Il affiche le chemin et refuse un dossier de figures déjà existant pour préserver les exports. Les courbes viennent des époques réellement exécutées ; aucune courbe n'est extrapolée.

## Frontières de validité

Trois graines mesurent ici la variabilité d'apprentissage sur un même train/validation, pas l'incertitude du corpus. La sélection d'époque utilise la validation ; les scores du test officiel restent absents. Le mode déterministe PyTorch ne garantit pas l'identité bit à bit entre CPU, MPS, CUDA, versions et plateformes.

Les checkpoints contiennent un `state_dict`, les labels, la configuration et le vocabulaire. Le rechargement utilise `torch.load(..., weights_only=True)` sur nos artefacts locaux. Leur ajout dans le dossier de modèles de l'application ne suffirait pas à intégrer la BiGRU : le backend actuel possède un chargeur scikit-learn, et son extension reste une tâche distincte.

La relance indépendante et Kaggle restent ouvertes pour L12. Les références techniques sont celles du [protocole BiGRU](38_protocole_bigru.md).
