# Criblage Kaggle : résultats du journal et sauvegarde des artefacts

**Journal reçu le 6 novembre 2026 — L05, L07 et L12 partiels.** Nous recevons la sortie console de la campagne `41ccf8e9d2bb4f658e1f3843b15ec90f_full`. Le dossier de travail distant est signalé comme disparu avant l’affichage des cellules suivantes. Nous archivons la sortie et distinguons les résultats imprimés des fichiers originaux non récupérés.

## Ce que montre le journal

Les trois taux passent chacun un smoke apparié, avec message de vérification. Les six adaptations FULL utilisent la graine42, TSAC10053/1774 et MADAR6388/800train/validation. Chaque candidat imprime un rechargement et une vérification des deux modèles, puis l’emplacement des figures. Le programme atteint enfin son tableau de synthèse. Aucun échec d’apprentissage n’apparaît dans la sortie fournie.

| Taux | Macro-F1 TSAC imprimé | Macro-F1 MADAR imprimé | Durée FULL imprimée |
| --- | ---: | ---: | ---: |
| 1e-5 | 0.934457 | 0.947274 | 533,2 s |
| 2e-5 | 0.935433 | 0.954740 | 539,6 s |
| 5e-5 | 0.924712 | 0.953027 | 537,2 s |

Le taux proposé par la synthèse est **2e-5 pour chaque tâche**, conformément au maximum de macro-F1 imprimé. L’accuracy correspondante vaut0.935738 sur TSAC et0.957500 sur MADAR. Les deux modèles retiennent époque3. La somme des durées FULL imprimées vaut1610,0s, soit26min50s ; elle exclut les contrôles préalables, les vérifications et les exports ultérieurs.

La marge TSAC sur1e-5 est0,000976, soit0,0976point de pourcentage. La marge MADAR sur5e-5 est0,001713, soit0,1713point. Ce sont de petits écarts sur une graine, sans preuve de significativité ni confirmation sur trois graines. Le taux initial2e-5 reste donc un candidat raisonnable, sans garantie d’optimalité sur de nouvelles données.

Pour TSAC, les pertes train diminuent tandis que les pertes de validation augmentent au fil des époques. Cela peut signaler du surapprentissage ou une confiance moins bien calibrée ; le journal ne permet pas d’en établir la cause. Le macro-F1, critère fixé avant le run, reste le critère de checkpoint. À5e-5, la meilleure époque TSAC est1, malgré la poursuite jusqu’à3 : le dernier état n’est pas automatiquement le meilleur.

## Limite de preuve et conservation locale

Le journal rapporte la réussite des rechargements sur Kaggle. Nous ne recevons ni `screening.json`, ni les `results.json`/`verification.json` originaux, ni poids, prédictions, journaux JSON d’époques ou figures. Nous ne pouvons donc pas contrôler leurs empreintes, recalculer indépendamment les métriques ou recharger les six checkpoints à partir de cette seule sortie.

Nous créons un fichier explicitement nommé `console_observations.json`, qui conserve les valeurs imprimées et leur niveau de preuve. Il ne remplace pas un fichier de résultats original. Aucun poids, matrice de confusion, intervalle ou métrique de précision supérieure à l’affichage n’est inventé. Les chiffres sont arrondis à l’impression.

Le journal brut est conservé, identique à la pièce jointe, sous `outputs/cloud_checks/2026-11-06/screening_console_41ccf8e9d2bb4f658e1f3843b15ec90f_full/`. Son SHA-256 est `582b8f2257cf1f38e776eba8a448445c46b66f141228837303a147995f01575f`. Le [rapport agrégé généré](../reports/encoder_screening_console_2026-11-06/README.md) décrit explicitement une observation console. Les six anciens checkpoints adaptés du Mac sont toujours présents ; ils constituent une expérience initiale distincte.

## Récupération avant toute relance

La disparition est rapportée ; nous n’avons pas identifié sa cause. Une réinitialisation de session est une possibilité. Nous devons d’abord regarder les versions déjà enregistrées du notebook et leurs fichiers **Output**. Une version peut disposer d’artefacts même si le répertoire de la session d’édition est vide. La documentation décrit une version de notebook comme un ensemble incluant code, sorties et métadonnées. [Documentation officielle Kaggle](https://www.kaggle.com/docs/notebooks)

Si une version enregistrée contient la campagne, récupérer ou rattacher ses fichiers avant de décider une nouvelle exécution. S’ils n’existent dans aucune version ni sauvegarde externe, le journal seul ne restaure pas les poids. Le fait de cloner le dépôt Git ne restaure pas ces sorties ignorées par Git.

## Sauvegarde pour les prochains runs

Lorsqu’elle est proposée dans les paramètres de session, activer la persistance des fichiers **avant** la fin de session. L’annonce officielle de Kaggle précise que ce réglage doit être actif avant l’arrêt pour conserver les fichiers. [Annonce Kaggle Staff sur la persistance](https://www.kaggle.com/discussions/product-feedback/355440)

Conserver aussi une version privée incluant les sorties et une copie téléchargée des artefacts nécessaires. Une archive ZIP laissée uniquement dans le même espace de travail n’est pas une sauvegarde indépendante. **Save & Run All** crée une nouvelle exécution depuis un état propre ; cette commande ne récupère pas les anciens fichiers disparus. La préparation du dépôt, de l’environnement et des sources doit figurer dans le notebook pour ce type d’exécution. [Documentation officielle Kaggle](https://www.kaggle.com/docs/notebooks)

La prochaine décision scientifique est de confirmer le réglage2e-5 sur les trois graines avec les journaux complets et une sauvegarde maîtrisée, après la recherche des artefacts existants. Cette décision ne lance aucun entraînement automatiquement. Le test officiel et le remplacement applicatif restent séparés.
