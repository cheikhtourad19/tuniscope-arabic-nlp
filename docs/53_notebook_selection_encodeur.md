# Notebook 07 : exécution du criblage encodeur

**6 novembre 2026 — L05/L07/L12 partiels.** Le [protocole52](52_protocole_selection_encodeur.md) fixe les trois taux et le critère avant les nouveaux résultats. Le notebook `07_selection_encodeur.ipynb` pilote les scripts ; il ne contient pas une seconde implémentation de l’apprentissage.

## Modes et sorties

- **full**, valeur par défaut : un smoke apparié par taux, son rechargement, puis deux adaptations sur toutes les données à la graine42. Trois taux produisent six modèles FULL. Chaque modèle est rechargé et fournit une figure PNG. Une sélection exploratoire est calculée séparément pour les deux tâches.
- **smoke** : trois petits essais appariés, 12modèles techniques et12figures. Aucun taux n’est choisi sur ces scores.
- **replay** : contrôle d’une campagne explicitement désignée, synthèse lisible et figures existantes. Le contrôle recalcule les métriques depuis les prédictions et vérifie les empreintes/proofs ; il ne refait pas l’inférence des checkpoints.
- **off** : aucun corpus, modèle ou sous-processus d’apprentissage chargé.

Le notebook utilise le noyau léger `TuniScope (notebooks)` sur Mac. Les calculs sont délégués à `environments/encoder` via `scripts/encoder_uv.sh`. Sur Kaggle, le noyau fourni peut rester Python3.13 : les sous-processus utilisent Python3.12.15 et les dépendances verrouillées du projet.

Les campagnes créées depuis le notebook se rangent sous `outputs/encoder_finetuning/screening/notebooks/<identifiant>_<mode>/`. Chaque taux possède un sous-dossier smoke puis, en full, un sous-dossier full. `screening.json` enregistre les liens et empreintes des runs, `campaign_verification.json` les contrôles, `summary.md` la synthèse lisible. Les checkpoints et journaux par époque restent dans les sous-dossiers des modèles ; leurs PNG sont dans `figures/`.

Les modes échouent explicitement si une étape manque, si un contrôle préalable échoue, si un fichier a changé ou si le mode demandé ne correspond pas au run. Une erreur conserve les sorties partielles et `failure.json`. Il n’y a pas de reprise exacte d’optimiseur ni d’écrasement automatique des dossiers précédents.

## Mac

Ouvrir le notebook depuis le dépôt. Laisser `TRAINING_MODE='full'` pour la vraie comparaison et `DEVICE='auto'` pour utiliser MPS lorsqu’il est disponible. Un FULL prend plus de temps qu’un smoke ; les six entraînements sont successifs. Les anciennes sorties du notebook06 et les douze modèles initiaux sont préservés.

La commande équivalente, depuis un terminal à la racine, est :

```sh
sh scripts/encoder_uv.sh run --locked --offline python -u scripts/run_encoder_screening.py --mode full --device mps --output outputs/encoder_finetuning/screening/mac_lr_full
```

Le dossier doit être nouveau. Le même principe s’applique à `--device cuda` sur Kaggle. Aucun déploiement applicatif ni entraînement LoRA ne découle de cette commande.

## Kaggle : préparer la même version du projet

Le code nouveau doit d’abord être enregistré et poussé sur Git par nous. Ouvrir une session privée avec T4 et Internet, puis attacher l’entrée privée contenant les deux corpus. Ne pas copier l’environnement du Mac vers Linux.

Pour une première session, cloner `https://github.com/cheikhtourad19/tuniscope-arabic-nlp.git` dans `/kaggle/working/tuniscope`. Pour le clone précédemment détaché sur l’ancien commit, récupérer le code poussé :

```bash
%%bash
set -euo pipefail
cd /kaggle/working/tuniscope
git fetch origin main
git checkout --detach FETCH_HEAD
git rev-parse HEAD
```

La dernière commande indique le commit réellement utilisé ; il sera aussi archivé dans chaque run. Une modification de code non poussée sur Git n’apparaît pas dans le clone Kaggle.

Installer uv dans une cellule :

```python
%pip install uv==0.13.1
```

Préparer le profil et les ressources de la phase encodeur déjà autorisée :

```bash
%%bash
set -euo pipefail
cd /kaggle/working/tuniscope
sh scripts/encoder_uv.sh sync --locked --python 3.12.15
sh scripts/encoder_uv.sh run --locked --offline python scripts/fetch_camelbert_tokenizer.py --download
sh scripts/encoder_uv.sh run --locked --offline python scripts/fetch_camelbert_encoder.py --download
sh scripts/encoder_uv.sh run --locked --offline python scripts/prepare_kaggle_sources.py --input-root /kaggle/input/datasets/cheikhtourad19/tuniscope-data
```

Les téléchargements de modèles sont explicites et réutilisent les caches vérifiés lorsqu’ils existent. Le script de données ne télécharge aucun corpus : il trouve les sources déposées manuellement, raccorde les répertoires par liens et vérifie les empreintes/effectifs v2. Une découverte ambiguë ou un autre répertoire existant bloque l’opération plutôt que de remplacer des données. Si Kaggle présente un autre chemin de montage, adapter uniquement `--input-root` au chemin réel.

Ouvrir/importer le notebook07 à jour, puis exécuter ses cellules avec `TRAINING_MODE='full'` et `DEVICE='cuda'`. Il détecte le dépôt sous `/kaggle/working/tuniscope`. La procédure de campagne entière n’a pas encore été exécutée sur Kaggle ; le contrôle CUDA initial de la doc49 est une preuve distincte.

Conserver une version privée avec les sorties et récupérer les preuves avant de terminer la session. Une synthèse JSON seule n’inclut pas les poids nécessaires à une nouvelle inférence. Aucun corpus protégé ne doit être rendu public par la sauvegarde ou l’export.

## État des preuves locales

La campagne `outputs/encoder_finetuning/screening/2026-11-06_lr_screen_mps_smoke/` a réellement exécuté les trois taux en petits essais MPS. Chacun comporte quatre modèles et huit époques, rechargés et contrôlés : **12modèles,24époques et12PNG**. La sélection reste volontairement vide. Les durées des runs techniques sont26,99s,25,43s et24,95s, hors rechargement/export ultérieurs. Ces scores ne constituent pas une comparaison FULL des taux.

Le notebook07 a été exécuté dans le vrai noyau Jupyter en **replay smoke** : quatre cellules de code et12images affichées, preuve privée `outputs/notebook_checks/2026-11-06/screening_native_smoke_replay.json`. L’entraînement technique a été lancé par CLI ; ce rejeu n’est pas un entraînement FULL natif.

Les tests ciblent la restriction au seul taux, le maintien des trois graines du protocole initial, le criblage à42, la sélection par tâche et son départage, le rejet des scores smoke/incomplets/non finis, la séquence smoke→vérification→FULL et les erreurs visibles. Les neuf tests du moteur existant et cinq du notebook06 restent passants. Le mode off du notebook07 est contrôlé sans accès aux modèles.

Les six adaptations FULL de criblage et la confirmation finale sur trois graines restent à exécuter. Nous avons choisi Kaggle T4 pour lancer la campagne complète ; les preuves ci-dessus restent celles du contrôle local MPS. Le protocole de sélection et les scripts prêts ne suffisent pas à clore les livrables.

Le contraste initial CLS gelé/adapté à2e-5 est préservé comme comparaison contrôlée. Un encodeur optimisé avec un autre taux représente un nouveau système : son écart à une ancienne référence gelée ne doit pas être attribué uniquement au dégel du backbone.
