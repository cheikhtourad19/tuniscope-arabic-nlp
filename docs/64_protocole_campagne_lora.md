# Campagne LoRA multitâche sur Qwen3-0.6B

## Hypothèse et réglages

Nous mesurons si un adaptateur partagé améliore la classification de sentiment TSAC et l’identification de variétés MADAR par rapport à Qwen avant adaptation. Chaque graine entraîne son propre adaptateur sur les deux tâches. La graine fixe l’initialisation et l’ordre des exemples, tandis que le corpus reste identique.

Nous fixons avant entraînement les graines 13, 42 et 2026, deux époques et un seul taux de 0,0001, déjà éprouvé au contrôle court. Le cahier autorise jusqu’à deux taux LoRA. Nous retenons une configuration fixe pour consacrer le calcul aux trois répétitions. Nous utilisons AdamW à taux constant, une décroissance des poids de 0,01 et un écrêtage de la norme des gradients à 1. Le lot physique de 2 et huit accumulations donnent 16 exemples par mise à jour. Chaque perte est pondérée par le nombre de tokens cibles de son sous-lot. Les 4 000 lignes SFT du train v2 sont parcourues à chaque époque dans un ordre mélangé propre à la graine.

Le modèle de base et le gabarit restent ceux de la référence. Le rang 8, alpha 16 et dropout 0,05 s’appliquent aux projections q et v. Seuls les 1 146 880 paramètres ajoutés sont optimisés. La précision FP32 et l’attention eager reprennent l’essai mesuré sur T4. FP32 est une adaptation au protocole proposé, validée par le passage arrière et le contrôle du lot long. Le label et sa fin de réponse fournissent la cible, tandis que le prompt et le remplissage sont masqués.

## Contrôles et validation

Avant chaque entraînement complet, nous exécutons une passe courte sur 100 exemples par tâche, avec échantillonnage par longueur et couverture des labels. Le modèle complet repart des poids initiaux. Nous contrôlons les gradients, la perte, le gel de la base et le rechargement de l’adaptateur sauvegardé. Le checkpoint complet correspond à la dernière époque, fixée à l’avance.

L’évaluation reprend les 1 774 lignes TSAC et 800 lignes MADAR de validation v2, pour chacune des deux conditions zero-shot et few-shot de la référence. Le décodage glouton, la limite de 32 tokens, les démonstrations du train et le parseur exact restent identiques. Les sorties invalides comptent comme erreurs. Nous conservons les matrices, métriques par classe, scores par système d’écriture et taux de labels valides. Le macro-F1 est résumé par moyenne et écart-type d’échantillon sur les trois graines, dans chacune des quatre conditions. Les scores de la base ont été mesurés sur MPS, tandis que la campagne est prévue sur CUDA. Le test officiel reste réservé à l’évaluation finale.

## Exécution et conservation

Le notebook 10 utilise les mêmes scripts sur Mac et Kaggle. Sur Kaggle, deux processus séparés peuvent travailler en parallèle, chacun sur sa propre T4. La troisième graine démarre dès qu’une carte se libère. Sur Mac, les graines se succèdent sur MPS. Le calcul par graine reste identique. Les temps réels incluront l’effet du chargement, des contrôles et de la validation.

Chaque graine écrit un dossier isolé. Un adaptateur déjà entraîné et vérifié est réutilisable si la validation est interrompue et que le dossier de travail existe toujours. Un échec pendant l’entraînement nécessite de recommencer cette graine dans un nouveau dossier. Sur Kaggle, une archive par graine terminée contient les poids LoRA, la configuration et les synthèses vérifiées. Les prédictions individuelles restent dans le dossier de travail privé. Télécharger les archives pendant la session. La synthèse de campagne est disponible après trois validations terminées.

Cette campagne apporte des preuves pour **L06** (adaptateur partagé), **L07** (validation comparative) et **L12** (procédure et traces). Les résultats, les poids rechargés et le parallélisme CUDA restent à vérifier sur l’exécution complète.
