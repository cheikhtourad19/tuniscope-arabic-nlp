# Criblage encodeur Kaggle : observations du journal

**Preuve limitée au journal fourni.** Les valeurs sont celles imprimées, arrondies ; les fichiers de résultats, poids et prédictions ne sont pas reçus. Ce relevé ne remplace pas les artefacts originaux.

| Taux | TSAC Macro-F1 | MADAR Macro-F1 | Durée imprimée du run FULL |
| ---: | ---: | ---: | ---: |
| 1e-05 | 0.934457 | 0.947274 | 533.2 s |
| 2e-05 | 0.935433 | 0.954740 | 539.6 s |
| 5e-05 | 0.924712 | 0.953027 | 537.2 s |

Le journal comporte trois contrôles préalables vérifiés, six adaptations complètes et leurs messages de rechargement réussi, puis la synthèse finale. Le meilleur taux imprimé est **2e-5 pour chaque tâche**, graine42. La confirmation sur trois graines n’est pas exécutée dans cette campagne.

La somme des durées FULL imprimées est **1610,0 s (26 min 50 s)**, hors smokes, vérifications et exports ultérieurs. Cette somme n’est pas la durée totale de la session ni un benchmark répété.

À2e-5 : accuracy impriméeTSAC0.935738, MADAR0.957500 ; meilleure époque3 pour chaque tâche. À5e-5, TSAC retient époque1 et non la dernière. Les pertes de validation TSAC augmentent tandis que la perte train diminue ; cela invite à examiner surapprentissage et calibration, sans changer a posteriori le critère macro-F1.

La présence du tableau final indique que le programme a atteint sa synthèse après les contrôles. La disparition ultérieure du dossier est rapportée par nous, mais sa cause et l’existence d’une version sauvegardée Kaggle restent inconnues. Aucun poids, matrice de confusion ou journal complet précis n’est reconstruit depuis les chiffres imprimés.

Le journal brut et son relevé sont archivés sous `outputs/cloud_checks/2026-11-06/screening_console_41ccf8e9d2bb4f658e1f3843b15ec90f_full/`, exclu de Git. SHA-256 du journal : `582b8f2257cf1f38e776eba8a448445c46b66f141228837303a147995f01575f`.

L05/L07/L12 restent partiels : rechercher les sorties d’une version Kaggle enregistrée avant de décider une relance, puis confirmer le réglage retenu sur trois graines avec sauvegarde durable. Les anciens checkpoints initiaux du Mac sont toujours présents.
