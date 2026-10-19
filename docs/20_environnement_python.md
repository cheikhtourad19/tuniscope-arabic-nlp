# TuniScope — Environnement Python des données et baselines

**Livrable :** L12 (environnement et reproductibilité), contribution partielle ; prérequis de L03 et L04. **Date :** 17 octobre 2026. **État :** installé et vérifié sur Mac Apple Silicon ; exécution Kaggle/Colab et environnement neuronal encore à vérifier.

## Décision et fichiers de référence

Nous utilisons **Python 3.12.15** et **uv 0.13.1** pour les données et les premières baselines CPU. Python 3.9.6 reste l'interpréteur système du terminal ; Homebrew fournit également Python 3.14.7, employé uniquement pour installer l'outil uv dans un environnement local de démarrage. Nous lançons désormais les scripts du projet avec son propre Python 3.12.15.

`pyproject.toml` déclare les dépendances directes ; `uv.lock` fixe toutes les versions résolues et les empreintes des distributions pour macOS ARM64 et Linux x86-64. `.python-version` fixe l'interpréteur. `requirements.txt` est un **export généré** du verrouillage, avec versions et empreintes, pour les outils qui utilisent pip. Nous ne le modifions pas à la main. `.venv/` contient les paquets réellement installés ; `.tools/` contient l'outil, le Python géré et son cache local. Ces deux dossiers sont ignorés par Git.

| Élément vérifié localement | Version |
| --- | --- |
| Python | 3.12.15 |
| uv | 0.13.1 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |

Ce verrouillage couvre actuellement les bibliothèques CPU de préparation et de classification classique. Le choix des paquets PyTorch/Transformers/PEFT et des distributions MPS/CUDA sera ajouté et testé lors de la phase neuronale. Les mêmes sources de code seront partagées ; les environnements installés restent propres à chaque machine.

## Installation et utilisation sur le Mac

Depuis la racine du dépôt, pour une première installation sur ce Mac disposant de Homebrew Python :

```sh
/opt/homebrew/bin/python3 -m venv .tools/bootstrap
.tools/bootstrap/bin/python -m pip install --no-cache-dir --index-url https://pypi.org/simple uv==0.13.1
sh scripts/project_uv.sh sync --locked
```

Ces opérations ont déjà été exécutées sur le poste local. Pour les prochaines séances :

```sh
sh scripts/project_uv.sh run --locked python scripts/check_environment.py
sh scripts/project_uv.sh run --locked python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Le script `project_uv.sh` retrouve uv localement et place ses téléchargements sous `.tools/`. `--locked` refuse une désynchronisation entre les dépendances déclarées et le verrouillage. Aucune activation manuelle de l'environnement n'est nécessaire avec ces commandes.

L'export pip se régénère avec :

```sh
sh scripts/project_uv.sh export --locked --format requirements-txt --no-header --no-emit-project --output-file requirements.txt
```

Un environnement Python 3.12 séparé utilisant pip peut installer cet export avec `python -m pip install --require-hashes -r requirements.txt`. Le workflow de référence reste uv ; une mise à jour des dépendances exige un nouveau verrouillage, les contrôles et une entrée de journal avant de remplacer les versions d'une expérience.

## Procédure préparée pour Kaggle ou Colab

Nous devons disposer d'une copie du dépôt dans un dossier de travail inscriptible et nous placer à sa racine. L'accès Internet est nécessaire au premier téléchargement de Python et des paquets. Dans le notebook, les cellules suivantes construisent un environnement isolé et exécutent le contrôle **dans cet environnement**, même si le noyau du notebook utilise un autre Python :

```python
import subprocess
import sys

subprocess.run([sys.executable, "-m", "venv", ".tools/bootstrap"], check=True)
subprocess.run([
    ".tools/bootstrap/bin/python", "-m", "pip", "install",
    "--index-url", "https://pypi.org/simple", "uv==0.13.1"
], check=True)
subprocess.run(["sh", "scripts/project_uv.sh", "sync", "--locked"], check=True)
subprocess.run([
    "sh", "scripts/project_uv.sh", "run", "--locked",
    "python", "scripts/check_environment.py"
], check=True)
```

Les imports exécutés directement dans le noyau du notebook restent dans son environnement d'origine. Nous lancerons les scripts du projet par ces sous-processus tant qu'un noyau dédié n'a pas été configuré. Il ne faut pas copier `.venv/` depuis le Mac vers Linux : le verrouillage sert à reconstruire les paquets adaptés à la plateforme. Cette procédure est **préparée, pas encore exécutée sur Kaggle ou Colab**. Le contrôle fictif n'a besoin d'aucun corpus.

### Transfert des données : dépôt manuel

Nous avons choisi de **déposer manuellement les données sur Kaggle**, dans un espace privé et dans le cadre des conditions d'accès des corpus. Aucun téléchargement automatique des corpus ni accès à un lien reçu par courriel n'est prévu dans le notebook. Nous utiliserons les archives originales déjà acquises et identifiées dans `SOURCE_REGISTER.md`, puis les mêmes manifestes v1 et règles de chargement qu'en local. Les archives et textes restent hors du dépôt Git.

Avant le premier run sur Kaggle, nous devrons relever le chemin effectif des fichiers déposés, le raccorder à la configuration des sources, vérifier les SHA-256 et contrôler les effectifs. Les chargeurs utilisent encore les chemins locaux du projet : leur adaptation au montage Kaggle reste à faire. Nous ne recommencerons pas la revue humaine ; la reproduction des audits complets nécessitera aussi leurs relevés privés, alors que le chargement courant utilise les sources et manifestes déjà figés. Le dépôt manuel et ces vérifications n'ont pas encore été exécutés sur Kaggle.

## Preuves locales et limite de l'ablation

Le contrôle `check_environment.py` a réussi sur des exemples inventés : DataFrame Unicode, matrice creuse TF-IDF et minuscule ajustement d'un classifieur pour vérifier les bibliothèques. Il ne mesure aucune performance sur MADAR ou TSAC. Les 18 tests de données réussissent aussi sous Python 3.12, et `uv pip check` vérifie la compatibilité des dix paquets installés. Le chargeur a été converti réellement en DataFrames de **11 827 × 17** pour TSAC et **7 188 × 17** pour MADAR, sur train/validation, sans écrire les textes ni ouvrir le test officiel.

L'inspection du code installé de scikit-learn 1.9.1 montre que l'analyseur TF-IDF de caractères réduit lui-même les suites d'espaces à un espace. Un contrôle synthétique confirme ce comportement. Notre comparaison brut/normalisé doit donc interpréter l'effet **supplémentaire** de NFC et du retrait des espaces périphériques, sans attribuer au normaliseur une transformation déjà effectuée dans les deux branches par TF-IDF. Cette réserve est ajoutée au [protocole d'ablation](19_protocole_ablation_normalisation.md).

Les distributions verrouillées comportent des roues Python 3.12 pour les deux plateformes visées, mais cette vérification de disponibilité ne remplace pas un essai Linux réel. **L12 reste incomplet** jusqu'aux vérifications Kaggle, aux dépendances neuronales et à la relance complète du système.

## Références techniques

- [uv : installation](https://docs.astral.sh/uv/getting-started/installation/).
- [uv : installation d'un Python géré](https://docs.astral.sh/uv/guides/install-python/).
- [uv : structure du projet et verrouillage universel](https://docs.astral.sh/uv/concepts/projects/layout/).
- [scikit-learn : installation dans un environnement isolé](https://scikit-learn.org/stable/install.html).
