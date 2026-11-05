# Fine-tuning CAMeLBERT : dispositif, preuves et notebook 06

**5 novembre 2026 — contributions partielles à L05, L07, L08 et L12.** Le [protocole 46](46_protocole_fine_tuning_encodeur.md) fixe les choix avant les scores. La phase a commencé sur continuation explicite après la référence gelée.

## Différence avec la référence précédente

Le notebook05 évaluait BERT gelé, moyenne masquée et tête linéaire apprise. Le notebook 06 emploie la classe native `AutoModelForSequenceClassification` demandée au cahier : CLS/pooler et tête linéaire. Pour contrôler le changement de représentation, nous ajoutons un témoin **CLS gelé** à la même architecture. La comparaison directe avec l'ancien mean-pooling reste une comparaison de systèmes distincts.

Les deux nouvelles branches partagent base épinglée, initialisation de tête par graine, labels, textes bruts, longueur128, ordre des lots et optimiseur. Le dropout du backbone est désactivé dans les deux. `eval()` n'interdit pas les gradients ; le gel utilise séparément `requires_grad`. Le dropout0,1 de la tête est actif pendant l'apprentissage. Le témoin gelé utilise le même petit learning rate et budget que l'adapté ; il ne représente pas la meilleure tête gelée recherchée indépendamment.

## Configuration et frontières

`configs/camelbert_finetuning_v1.json` : batch physique8, accumulation2 (16 effectifs, dernière fenêtre pondérée selon effectif réel), AdamW2e-5 constant, weight decay0,01, clipping1, maximum3 époques, patience2, graines13/42/2026. Précisionfloat32, attentioneager, même profil `environments/encoder/` et verrou inchangé ; les autres environnements restent intacts.

Deux modèles indépendants : TSAC NEG/POS et MADAR MSA/TN/EG. En FULL :10053/1774 et6388/800train/validation v2. Le loader et les manifestes restent ceux des expériences précédentes ; aucune nouvelle exclusion, annotation, normalisation ou sélection de test officiel.

Modèle natif chargé :109082882paramètres TSAC et109083651MADAR. Tous sont entraînables dans l'adapté ; seuls1538/2307 le sont dans le témoin. Les gradients d'une matrice d'attention du premier bloc Transformer sont vérifiés, ainsi que son changement effectif, pour distinguer adaptation de BERT et simple apprentissage de la tête/pooler.

## Essais réalisés avant le complet

Le premier essai MPS s'arrête au premier backward adapté : `embedding_dense_backward_mps` n'a pas d'implémentation strictement déterministe. Le dossier partiel et `failure.json` sont conservés. Nous amendons seulement la politique de déterminisme MPS en `warn_only=True`, sans modifier architecture, données ou learning rate après un score adapté. Les graines restent fixées, CPU/CUDA restent stricts ; aucune identité bit à bit MPS n'est promise.

Le smoke corrigé `outputs/encoder_finetuning/2026-11-05_cls_paired_mps_warnonly_smoke/` termine en27,1s :100lignes par partition/tâche, deuxépoques, graine42 etquatre modèles. Les quatre checkpoints sont rechargés, toutes leurs prédictions/métriques de validation sont recalculées, dix sorties par tâche/branche sont explicitement contrôlées. Les témoins conservent le backbone et les adaptés ont des gradients et des poids Transformer modifiés. Les scores de ce smoke ne sont pas présentés comme performances FULL.

Le probe sur entrées **inventées**,8×128positions, réussit forward/backward/AdamW en3,07s. Allocation MPSdriver observée aprèsstep3344728064octets, tenseurs1826280704 ; ces instantanés ne constituent pas un pic absolu et ne doivent pas être additionnés auRSS. Aucun corpus n'est lu par ce probe. Le smoke atteint un picRSSprocessus1190658048octets ; les mémoires CPU/MPS partagent des ressources surApple.

L'extrapolation initiale à partir de la dernière époque du petit essai est61,4minutes pour les six adaptés et74,3minutes pour les douze modèles appariés, à troisépoques sansarrêtanticipé. Elle exclutI/O/rejeu et dépend des longueurs réelles et de la charge : c'est une **estimation**, pas la durée du run complet. Le Mac est retenu pour cette première expérience ; CUDA/Kaggle et drivercompatibleCUDA13 ne sont pas vérifiés.

## Tests et vérifications

Sept tests avec un petit BERT aléatoire et deux contrôles ciblés du journal couvrent gradients/poids dans les deux régimes, indépendance du gel et d'eval, accumulation y compris dernière fenêtre partielle, validation sans mise à jour, rechargement natifsafetensors avec prédictions identiques, rejet d'empreinte falsifiée et garde avantFULL.

Cinq tests du notebook couvrent structure etmodeoff sans chargement, smokeapparié obligatoire avantfull, replayexplicitementtypé, erreurs visibles, environnement séparé horsligne et suppression duVIRTUAL_ENVhérité. Le notebook 06 a été exécuté dans le vrai noyau **en replay du smoke** : quatrecellulesdecode, quatremodèlesvérifiés etquatrePNGaffichés. Preuve privée `outputs/notebook_checks/2026-11-05/finetuning_native_smoke_replay.json`. Ce replay n'est pas revendiqué comme entraînementFULLnatifJupyter.

## Run complet

Le run **FULL apparié est terminé et vérifié** : `outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full/`. Douze modèles couvrent les deux tâches, les deux régimes et les trois graines. Le [rapport généré et ses quatre cartes initiales](../reports/camelbert_finetuning_v1_2026-11-05/README.md) fournit tous les scores individuels, métriques par classe/écriture et empreintes.

| Corpus | Méthode | Macro-F1 de validation |
| --- | --- | ---: |
| TSAC | TF-IDF caractères brut, référence fixée | 0.928898 |
| TSAC | Gelé mean-pooling, protocole antérieur | 0.844608 ± 0.003337 |
| TSAC | Témoin CLS gelé à budget identique | 0.605756 ± 0.003936 |
| TSAC | CLS adapté | 0.935802 ± 0.004449 |
| MADAR | TF-IDF caractères brut, référence fixée | 0.949244 |
| MADAR | Gelé mean-pooling, protocole antérieur | 0.921926 ± 0.003579 |
| MADAR | Témoin CLS gelé à budget identique | 0.482670 ± 0.034343 |
| MADAR | CLS adapté | 0.950815 ± 0.002911 |

Les écarts-types sont échantillonnaux, sur trois graines. Les moyennes adaptées dépassent numériquement TF-IDF de **0,006904** sur TSAC et **0,001571** sur MADAR ; ces faibles écarts de validation ne prouvent pas un gain sur le test ni une différence significative. Le témoin CLS court/faible learning rate est nettement inférieur à la référence gelée antérieure ; il ne doit pas être présenté comme le meilleur encodeur gelé possible.

La durée réelle totale est **3757.42 secondes (62.62 minutes)**  chargements  douze apprentissages  validations et sauvegardes compris ; installation  smoke  rejeu et export ultérieurs sont exclus. Le pic RSS du processus est **1 685 798 912 octets**.
Le maximum échantillonné MPS atteint **1 781 848 576 octets** de tenseurs et **6 090 227 712 octets** côté pilote. Ce sont des maxima observés après backward/step  pas un pic absolu garanti ; ils ne s’additionnent pas au RSS sur mémoire unifiée Apple.
Les sous-dossiers adaptés occupent environ **438 Mo chacun**, tokenizer et journaux compris ; un témoin ne sauvegarde que sa tête avec les métadonnées et le tokenizer, en réutilisant la base exacte.

Les douze checkpoints ont été rechargés et leurs **15 444 prédictions de validation** rejouées (1 774 × 6 pour TSAC et 800 × 6 pour MADAR). Tous les labels et probabilités correspondent, différence absolue maximale observée **0** ; la tolérance annoncée reste 1e-4. Les dix premières sorties de chaque modèle sont aussi explicitement contrôlées. Le rapport vérifie l’alignement des identifiants/groupes/labels avec la référence TF-IDF et celui des partitions avec BiGRU/mean-pooling.

Le notebook a été exécuté dans le noyau réel en **replay FULL** : quatre cellules, douze modèles vérifiés et douze PNG affichés. L’entraînement lui-même a été lancé par le script ; ce replay n’est pas un second entraînement natif. Preuve privée : `outputs/notebook_checks/2026-11-05/finetuning_native_full_replay.json`. Les douze figures sont liées au même results.json par empreinte ; quatre figures représentatives ont été inspectées visuellement.

### Limite du journal initial et correction

Ce premier run conserve pertes et macro-F1 à chaque époque, mais les scores par classe et les prédictions ne sont archivés que pour le checkpoint retenu. Les détails des autres époques n’ont pas été conservés et **ne sont pas inventés rétroactivement**. C’est une limite à lever pour les runs finaux exigés au cahier.

Le producteur a été corrigé **après** ce run : il enregistre désormais métriques par classe et prédictions à chaque époque. Le vérificateur recalcule ces métriques lorsqu’elles existent et signale leur absence pour les anciens runs. Deux tests ciblés couvrent ce contrôle et le rejet de données incohérentes. Un nouveau smoke intégré (`2026-11-05_cls_epochlogs_mps_smoke`, 22,1 s) conserve et fait recalculer les métriques/prédictions de ses huit époques, avec quatre modèles rechargés ; il ne remplace pas le run complet historique.

Les sources exactes du run initial, sa configuration et son protocole sont archivés sous `outputs/encoder_finetuning/source_snapshots/2026-11-05_initial/` ; leurs empreintes correspondent à celles enregistrées au lancement. Le vérificateur actuel possède sa propre empreinte. Aucun poids ou score initial n’est écrasé.

## Utilisation du notebook et des scripts

Le [notebook 06](../notebooks/06_fine_tuning_encodeur.ipynb) garde le noyau **TuniScope (notebooks)**. Son bridge délègue les calculs au profilencodeur : il n'est pas nécessaire d'ajouterTorch au noyau existant.

- `TRAINING_MODE='full'`, valeur par défaut : nouveau contrôle100lignes, puis tout le train/validation, graines13/42/2026, contrôle de rechargement et figures. `REGIME='paired'` produit douze modèles ; `finetuned` produit les six adaptés seulement.
- `smoke` : petit contrôle uniquement, identifié comme tel dans les tableaux/figures.
- `replay` : aucun nouvel apprentissage. Choisir un dossier explicite et `REPLAY_EXPECTED_MODE='full'` ou `'smoke'` ; aucun repli silencieux.
- `off` : aucun modèle ni corpus chargé.

Les nouveaux runs se rangent sous `outputs/encoder_finetuning/notebooks/`. Chaque sous-dossier modèle contient tokenizer/configuration/mappings, `history.json`, `predictions.json` et `tuniscope.json`. Les adaptés sauvegardent tous les poids dans `model.safetensors` ; les témoins sauvegardent la tête dans `head.pt` et réutilisent la base officielle exacte. Le meilleur checkpoint est sauvegardé pendant l'apprentissage ; une reprise exacte d'optimiseur interrompu n'est pas implémentée.

L'exportPNG dérive les pertes, macro-F1 et matrices depuis le même run vérifié. Les sorties volumineuses restentGit-ignorées. L'inférence dédiée ne lit aucun corpus et ne réentraîne rien. Les scores affichés ne sont pas calibrés et la troncature est signalée.

Depuis la racine :

```sh
sh scripts/encoder_uv.sh run --locked --offline python scripts/verify_encoder_finetuning.py outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full
```

```sh
sh scripts/encoder_uv.sh run --locked --offline python scripts/encoder_finetuning_inference.py --checkpoint outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full/tsac_finetuned_seed_42 --text 'الخدمة باهية برشا'
```

Ce texte d'inférence est inventé ; ce n'est pas un score de benchmark. Le run complet indiqué est désormais terminé et vérifié.

Pour suivre un entraînement long avant l'apparition de `results.json` :

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/show_encoder_progress.py outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full
```

## Limites et suite

Le seul learning rate2e-5 est exécuté à ce stade ;1e-5/5e-5 ne sont pas présentés comme testés. La sélection finale, les ablations restantes, les cartes finales, l'analyse d'erreurs, la reproduction bibliographique et l'évaluation du test restent ouvertes. Les nouveaux poids ne sont pas déployés automatiquement dans l'application. Le LLM, LoRA et le RAG sont des phases séparées.
