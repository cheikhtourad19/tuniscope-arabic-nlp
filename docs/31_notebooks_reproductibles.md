# TuniScope — Premiers notebooks reproductibles

**Date :** 19 octobre 2026. **Contributions partielles :** L03 (données), L04 (baselines), L07 (vérification des prédictions), L12 (reproductibilité). Nous organisons les traitements déjà vérifiés sans remplacer leurs scripts ni modifier le protocole expérimental.

## Points d'entrée et prérequis

- `notebooks/01_donnees_v2.ipynb` : sources → sélection par manifestes v2 → normalisation en copie → DataFrame en RAM → statistiques agrégées et assertions.
- `notebooks/02_baselines_classiques.ipynb` : contrôle fictif de l'environnement, puis relance explicite des familles TF-IDF caractères, mots ou combinaison. Les références majoritaires sont incluses.

Les cellules explicatives précisent objectifs, prérequis, mécanismes, contrôles et limites. Le code d'apprentissage reste dans `scripts/run_baselines.py` et `scripts/baseline_models.py`. `scripts/notebook_support.py` sert uniquement de passerelle et de résumé agrégé ; il ne réimplémente ni TF-IDF ni le classifieur. Les scripts de règles et d'ajout lexical restent documentés séparément dans les documents 29 et 30 ; leur orchestration n'est pas intégrée à ces notebooks.

Nous ouvrons les notebooks depuis le dépôt ou son sous-dossier `notebooks/`, après installation de l'environnement selon le document 20. Le noyau du notebook utilise seulement la bibliothèque standard. Les traitements de données et d'apprentissage sont exécutés par `scripts/project_uv.sh run --locked --offline python` : c'est l'environnement Python 3.12 du projet, pas nécessairement celui du noyau. Le verrouillage doit déjà être installé ; un paquet absent du cache provoque un échec, pas un téléchargement implicite. Aucun paquet Jupyter n'est ajouté au verrouillage CPU actuel.

## Modes explicites et sécurité des données

Le notebook données démarre avec `DATA_MODE = 'synthetic'` : quatre textes inventés vérifient le passage vers un tableau de 4 × 17. Le mode `'real'` charge les sources autorisées déjà présentes, avec contrôles d'empreintes et sélection explicite train/validation v2. Il ne télécharge pas les corpus et ne reconstruit pas les décisions humaines privées.

Le notebook baselines démarre avec `TRAINING_MODE = 'off'`. Son petit ajustement sur textes inventés est uniquement un contrôle d'installation. Les modes suivants exigent une activation explicite :

| Mode | Traitement réel |
| --- | --- |
| `off` | Aucun corpus lu par le notebook baselines ; contrôle fictif seulement |
| `smoke` | 100 lignes train et 100 validation par tâche, six résultats, puis vérification |
| `full` | Nouveau smoke vérifié, puis nouveau run complet et vérification |

`FAMILY` choisit `characters`, `words` ou `combined`, correspondant aux trois configurations v2 existantes. Un échec du smoke interrompt le mode full. Chaque exécution utilise un identifiant unique sous `outputs/baselines/notebooks/`, ignoré par Git ; les anciens dossiers ne sont jamais écrasés. Les modèles contiennent des vocabulaires issus du corpus et restent privés, comme les prédictions et leurs identifiants. Ne recharger que des checkpoints de confiance.

Nous ne sélectionnons aucune ligne du test officiel, ne changeons aucun label, manifeste ou fichier original, et n'imprimons aucun texte réel. La vérification d'intégrité lit les fichiers sources physiques, qui peuvent contenir plusieurs partitions MADAR : cela n'autorise pas l'utilisation des lignes test pour l'entraînement ou les choix de modèle.

Les notebooks distribués restent sans sorties ni compteurs d'exécution. Après une utilisation interactive, effacer les sorties avant partage ; le vérificateur refuse des cellules de code avec sorties sauvegardées. Une relance significative doit être consignée dans `EXPERIMENTS.md` ; les métadonnées du runner ne remplacent pas ce journal.

## Vérification locale effectivement exécutée

`scripts/check_notebooks.py` contrôle notre structure attendue au format 4.4 et compile les cellules Python. Avec `--execute`, il exécute les cellules de nos deux notebooks dans l'ordre, avec un espace de variables distinct par notebook. Il n'exécute que ces deux fichiers connus du dépôt, dont le code doit être de confiance.

**Limite importante :** il s'agit d'une exécution séquentielle Python, pas d'une validation complète du schéma par nbformat ni d'une exécution dans un véritable noyau Jupyter. L'affichage, les métadonnées gérées par l'interface, le lancement du noyau et la relance sur Kaggle/Colab restent à vérifier. Aucun de ces essais n'a été supposé réussi à partir du seul contrôle séquentiel.

Commandes vérifiées sur Mac ARM64, Python 3.12.15, CPU :

```sh
# Structure et modes sûrs, sans corpus
sh scripts/project_uv.sh run --locked --offline python scripts/check_notebooks.py --execute

# Chargement réel v2 et petit entraînement caractères
sh scripts/project_uv.sh run --locked --offline python scripts/check_notebooks.py --execute --data-mode real --baseline-mode smoke

# Deux autres chemins TF-IDF : petits entraînements seulement
sh scripts/project_uv.sh run --locked --offline python scripts/check_notebooks.py --execute --baseline-mode smoke --family words
sh scripts/project_uv.sh run --locked --offline python scripts/check_notebooks.py --execute --baseline-mode smoke --family combined

sh scripts/project_uv.sh run --locked --offline python -B -m unittest discover -s tests -p 'test_*.py' -q
```

Les **93 tests** passent, dont neuf nouveaux tests sur la structure sans sorties, les agrégats fictifs sans lecture de corpus, la préservation des entrées, les groupes séparés, le refus des lignes test, les manifestes v2 explicites, les modes autorisés, le sous-processus verrouillé/hors ligne et l'arrêt avant full en cas d'échec du smoke.

Le notebook données réel retrouve les formes TSAC **11 827 × 17** et MADAR **7 188 × 17**. Les effectifs train/validation sont respectivement **10 053 / 1 774** et **6 388 / 800**, sans groupe partagé. La normalisation change 917/144 textes TSAC et 18/3 MADAR ; les bruts sont préservés. Ces comptes correspondent à v2, pas aux anciens comptes par partition v1.

Les trois smoke tests aboutissent chacun à six résultats vérifiés : identifiants alignés, modèles rechargés, prédictions rejouées, métriques recalculées. Dossiers privés de cette vérification :

- caractères : `outputs/baselines/notebooks/a0b7f297741c4a57b2b67a43ea4274bb_characters_smoke/` ;
- mots : `outputs/baselines/notebooks/1b6923e152c6474eabf226b994971498_words_smoke/` ;
- combinaison : `outputs/baselines/notebooks/11872d5cc4dd45a78926b0c2764a0d8b_combined_smoke/`.

Un premier essai de la passerelle a été arrêté par la protection du runner : sa destination était hors de `outputs/baselines/`. La passerelle a été corrigée pour respecter cette contrainte, sans modifier le runner historique ; les trois essais ci-dessus ont ensuite réussi. Aucun run complet n'a été relancé par ces notebooks. Le chemin full est fourni et son arrêt sur échec du smoke testé, mais sa réussite de bout en bout via cette nouvelle passerelle reste à vérifier. Les runs complets historiques restent la preuve des scores de comparaison, pas ces nouveaux smoke tests.

## Ce que cette étape termine — et ne termine pas

Nous disposons de deux premiers points d'entrée commentés et de leurs contrôles locaux séquentiels. Cela ne termine aucun livrable global :

- **L03 :** l'audit de tokenisation et la relance cloud manquent encore ; les revues humaines ne sont pas recréées automatiquement.
- **L04 :** règles et augmentation lexicale ont leurs scripts, mais ne sont pas orchestrées ici ; BiGRU et modèle de langage caractères demandés restent à réaliser dans une phase explicitement autorisée. Ce ne sont pas de simples extensions facultatives du cahier.
- **L07 :** les vérifications de sauvegarde sont présentes ; évaluation finale, trois graines pertinentes, bootstrap et analyse complète restent ouverts.
- **L12 :** noyau/interface Jupyter, installation propre indépendante et Kaggle non vérifiés ; le dépôt manuel privé et l'adaptation des chemins de montage sont encore nécessaires. Nous n'avons pas copié l'environnement Mac sur Linux.

La suite est de fermer les lacunes documentaires des cartes/J1 et de relire les analyses assistées, avant de choisir explicitement la prochaine phase de modèles. Nous ne lançons pas automatiquement d'entraînement neuronal.
