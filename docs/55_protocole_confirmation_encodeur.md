# Confirmation des réglages encodeur sur trois graines

**6 novembre 2026 — protocole fixé avant les scores de confirmation. L05, L07 et L12 restent partiels.** Nous poursuivons après le criblage local conservé, sans modifier les corpus ni ouvrir le test officiel.

## Configuration retenue et provenance

Le rapport de la campagne MPS `708fce3a508b4bb6b1a7d2fbce54739c_full` contient les six résultats de criblage sur la graine42 et leurs empreintes. Le plan `configs/camelbert_confirmation_v1.json` fixe son SHA-256 : `330c5f3056c7e9ae9076889b72a86a5dd2f9b48ea2201dea7aaee1c78fe5815f`.

Nous retenons **5e-5 pour TSAC** et **2e-5 pour MADAR**, selon le critère de validation annoncé avant le criblage. Le journal Kaggle suggérait un autre taux TSAC ; il constitue une campagne distincte dont les artefacts n’ont pas été récupérés. Nous ne combinons pas leurs scores pour choisir une plateforme ou une graine favorable.

Le plan de confirmation vérifie que le rapport source comprend trois candidats par tâche, une seule graine42 et les maximums correspondant aux taux retenus. Aucun autre choix n’est ajusté à partir des futurs scores de confirmation.

## Expériences prévues

Deux expériences par tâche sont distinctes : un smoke apparié de100exemples par partition, puis une adaptation complète sur **13,42,2026**. La confirmation complète entraîne **six nouveaux modèles adaptés**, sans réutiliser les checkpoints du criblage et sans ajouter de témoins gelés FULL. Chaque modèle démarre depuis les poids généraux épinglés.

Les trois graines contrôlent initialisation de tête, ordre des lots et autres tirages. Les partitions v2 restent identiques : TSAC10053train/1774validation ; MADAR6388/800. Chaque tâche utilise exactement les mêmes identifiants sur ses trois graines. Les sources, labels et manifestes sont vérifiés par empreinte ; aucune ligne de test n’est sélectionnée.

Tous les réglages autres que le taux restent ceux du protocole46 : CAMeLBERT Mix général et tokenizer épinglés, architecture CLS native, raw-v0, longueur128, padding dynamique, float32/eager, lot8 et accumulation2, AdamW/weightdecay0,01, clipping1, plafond3époques/patience2, dropout du backbone désactivé et dropout0,1 de la tête. Chaque run de tâche contient un seul corpus avec son taux fixé ; cela ne crée pas de modèle multitâche.

## Matériel et contrôles

La campagne utilise un appareil et un environnement cohérents pour les deux tâches et les trois graines. Le Mac/MPS est la plateforme locale retenue pour cette exécution. Le code partagé conserve la détection CPU/MPS/CUDA ; une exécution sur un autre matériel constitue une campagne distincte et doit déclarer ce changement, sans mélange des scores entre campagnes.

Le FULL exige un smoke de même tâche, configuration, appareil et périmètre `confirmation`, dont les deux checkpoints sont rechargés et les quatre époques contrôlées. Les gradients et mises à jour du Transformer sont vérifiés dans le régime adapté ; le témoin garde son backbone. Le déterminisme strict non disponible pour certaines opérations MPS reste signalé par `warn_only`, sans promesse de répétabilité bit à bit.

À chaque époque : pertes train/validation, macro-F1, métriques par classe et prédictions sont archivés. Le meilleur checkpoint est choisi par macro-F1 de validation, première époque en cas d’égalité. Le programme recharge ensuite chacun des six checkpoints, rejoue toute la validation et vérifie toutes les prédictions/probabilités à la tolérance annoncée, dont dix sorties explicitement contrôlées.

## Rapport et limites

Nous publions les scores individuels de13/42/2026, la moyenne et l’écart-type échantillonnal. Aucune meilleure graine n’est sélectionnée pour le résultat principal. Les trois exécutions ne sont pas trois jeux de test indépendants. La graine42 a participé au choix du taux ; la validation a déjà servi à la recherche et ne fournit pas un score final indépendant.

Une baisse par rapport au meilleur score du criblage est conservée comme résultat. Nous ne changeons pas automatiquement le taux après lecture des moyennes. Une éventuelle nouvelle étude exigerait un protocole distinct et des limites supplémentaires sur l’usage répété de la validation.

Le contraste initial CLS gelé/adapté à2e-5 reste une ablation contrôlée séparée. Les réglages optimisés de cette confirmation constituent des systèmes sélectionnés ; leur écart à une ancienne référence n’isole pas à lui seul l’effet du dégel.

## Sauvegarde et périmètre de livraison

Tous les runs et poids initiaux sont préservés. Les nouvelles sorties vont dans un dossier privé neuf, avec une synthèse de confirmation, ses empreintes, six checkpoints, journaux complets, figures et cartes provisoires. Une archive de preuves pourra faciliter leur copie ; une copie sur le même disque ne constitue pas une sauvegarde indépendante. Les fichiers ignorés par Git doivent être sauvegardés séparément, notamment avant toute fin de session distante.

Le notebook08 distingue full, smoke, replay et off. Un smoke ne produit pas de moyenne finale ; un rejeu ne lance pas d’entraînement. Le protocole et ses tests ne prouvent pas l’achèvement d’un FULL.

Le test officiel, le bootstrap, la calibration/abstention, les composants LLM/RAG et le remplacement des modèles de l’application restent séparés. Les cartes et performances restent des résultats de validation jusqu’à l’évaluation finale du protocole global.
