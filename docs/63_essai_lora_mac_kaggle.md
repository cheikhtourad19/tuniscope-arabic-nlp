# Comparaison du petit essai LoRA sur Mac et Kaggle

**7 novembre 2026.** Le notebook 09 lance le même essai de 64 exemples avec lot physique de 2, huit accumulations, FP32 et graine 42. Il compare les mises à jour chronométrées après la chauffe à la référence Mac de 4,76058 secondes par mise à jour. Une extrapolation donne le coût de deux époques d’apprentissage par graine. [Notebook portable](../notebooks/09_essai_lora_qwen.ipynb)

Le notebook part d’un dépôt et d’un profil LoRA préparés. Sur Mac, ouvrir le notebook avec le noyau **TuniScope (notebooks)**. Le choix `auto` utilise MPS lorsqu’il est disponible. Les scripts s’exécutent dans `environments/lora` et utilisent les poids officiels locaux et le corpus SFT vérifié. La variable `TUNISCOPE_LORA_MODE=off` permet de parcourir les cellules sans lancer l’essai.

Sur Kaggle, attacher le dataset privé déjà déposé manuellement, choisir le GPU et activer Internet pour la cellule de démarrage fournie dans la conversation. Cette cellule récupère le code Git dans un répertoire temporaire, installe le profil verrouillé, raccorde les sources et vérifie les poids Qwen. Elle transmet au notebook le chemin du dépôt et le commit récupéré. Ajouter cette cellule au-dessus du notebook importé dans la même session. [Réglages des notebooks Kaggle](https://www.kaggle.com/docs/notebooks)

## Pousser les sources depuis le Mac

Le code Kaggle est récupéré depuis `main`. Depuis la racine locale, nous pouvons enregistrer les fichiers de cette étape et la référence Qwen agrégée qui les précède.

```sh
git add README.md EXPERIMENTS.md configs/qwen_sft_v1.json \
  docs/61_protocole_sft_qwen.md docs/62_resultats_sft_et_essai_lora.md \
  docs/63_essai_lora_mac_kaggle.md environments/lora/pyproject.toml environments/lora/uv.lock \
  scripts/lora_uv.sh scripts/build_qwen_sft.py scripts/run_qwen_lora_smoke.py \
  scripts/check_qwen_lora_capacity.py notebooks/09_essai_lora_qwen.ipynb \
  scripts/run_qwen_validation.py scripts/report_qwen_validation.py reports/qwen_validation_v2_2026-11-07
git commit -m "Prepare portable Qwen LoRA benchmark"
git push origin main
```

## Contrôles et conservation

Le corpus SFT reconstruit doit produire exactement l’empreinte du Mac. Le notebook contrôle également les empreintes du script d’essai, du modèle, du verrou LoRA et de la configuration. L’essai utilise un seul appareil. La synthèse CUDA relève le nom du GPU, le nombre d’appareils visibles et l’index du GPU utilisé.

Après l’essai, un passage avant et arrière sur les deux séquences de 370 et 348 tokens vérifie la capacité du même lot physique. Le script enregistre un éventuel manque de mémoire dans le rapport. Sur CUDA, il relève les pics d’allocation du processus. Sur MPS, il relève des instantanés du pilote.

Les corpus, caches et poids de travail restent dans le dépôt temporaire de la session Kaggle. Le ZIP dans `/kaggle/working` rassemble uniquement quatre fichiers de synthèse. Télécharger cette archive depuis **Output** pendant la session, puis la rapporter dans le projet local. [Gestion des sorties Kaggle](https://www.kaggle.com/docs/notebooks)

## État des vérifications

Les cinq cellules ont été exécutées dans un noyau Jupyter natif isolé sur Mac. Le corpus a été reproduit, l’adaptateur rechargé, le lot long a réussi et le ZIP a été relu. L’exécution a mesuré 4,374 secondes par mise à jour après chauffe. La référence CLI précédente de 4,761 secondes reste le point de comparaison affiché. La preuve locale se trouve dans `outputs/notebook_checks/2026-11-07/lora_notebook_native_mps.json`.

Cette préparation contribue à L06 et L12. L’exécution CUDA, la comparaison des deux plateformes et la campagne LoRA complète restent à effectuer.
