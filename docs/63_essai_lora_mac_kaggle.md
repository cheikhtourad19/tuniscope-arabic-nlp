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

## Résultats CUDA reçus

Le ZIP transmis le 7 novembre 2026 contient les quatre synthèses attendues. Le corpus SFT agrégé correspond octet pour octet à celui du Mac. Les empreintes du modèle, du script, du verrou, de la configuration et de l’adaptateur initial concordent. Le commit `6c74ce20dcf2e66313a6d5a5bf7895f19f0a84b3` correspond aux fichiers exécutés. Les deux essais utilisent 64 exemples, FP32 et un lot effectif de 16, avec trois mises à jour chronométrées après chauffe.

| Plateforme | Secondes par mise à jour | Deux époques estimées |
| --- | ---: | ---: |
| Mac MPS, référence CLI | 4,761 | 39 min 40 s |
| Mac MPS, exécution notebook | 4,374 | 36 min 27 s |
| Kaggle, une Tesla T4 | 1,390 | 11 min 35 s |

Sur les mises à jour chronométrées, la Tesla T4 est **3,42 fois plus rapide** que la référence CLI Mac et **3,15 fois plus rapide** que son rejeu notebook. La projection suppose 250 mises à jour par époque. Le temps total du petit essai inclut aussi les chargements, les empreintes et le rechargement. Il est de 35,30 secondes sur CUDA et de 29,00 secondes pour la référence CLI Mac. Les durées projetées concernent l’apprentissage et la validation constitue un coût séparé.

Le lot de 370 et 348 tokens a terminé son forward et son backward sur la Tesla T4. Le processus a atteint **6,16 Go alloués** et **6,43 Go réservés** sur **15,64 Go** visibles par ce GPU. Deux GPU sont visibles et l’essai utilise le GPU d’index 0. La synthèse CUDA rapporte une perte finie, des gradients LoRA non nuls, l’empreinte de base préservée et un rechargement avec écart de logits nul. Le ZIP contient les résultats de ces contrôles exécutés à distance.

La plateforme retenue pour la prochaine campagne LoRA est **Kaggle CUDA**, avec le lot physique de 2 et huit accumulations vérifiés. Les trois graines à deux époques représentent environ 35 minutes d’apprentissage d’après l’extrapolation. La preuve de L06 et L12 couvre l’essai CUDA, la conservation et la vérification des mesures. L’entraînement complet et la validation de l’adaptateur constituent les étapes suivantes.

L’archive originale et sa vérification sont conservées dans `outputs/qwen/2026-11-07_cuda_lora_f552ba423f36439ea2d671ed124ede02/`. SHA-256 du ZIP `d2dc5f2076de5e237d8f06a132ffd50e0b12eafd2dbd002a558d7e96b3d3b5f4`.
