# Sélection limitée du taux d’apprentissage de l’encodeur

**6 novembre 2026 — protocole fixé avant les nouveaux scores. Contributions partielles à L05, L07 et L12.** La première expérience à 2e-5 et sa revue assistée sont conservées. Ce document fixe une phase exploratoire distincte ; il n’annonce aucun nouveau résultat FULL ni configuration finale.

## Question et contrôle

Nous comparons trois taux d’apprentissage : **1e-5, 2e-5 et 5e-5**, dans cet ordre, pour l’adaptation complète de CAMeLBERT. Le taux détermine l’amplitude des mises à jour de l’optimiseur. Les autres choix restent ceux de `configs/camelbert_finetuning_v1.json` : base et révision épinglées, architecture CLS native, texte brut, longueur128, float32/eager, lot8 et accumulation2, AdamW/weightdecay0,01, plafond3époques et patience2, dropout du backbone désactivé et dropout0,1 de la tête. La normalisation et les labels ne sont pas modifiés après la revue qualitative.

Le plan `configs/camelbert_screening_v1.json` enregistre l’empreinte de cette configuration initiale. Aucun autre hyperparamètre n’est ajusté dans ce criblage. Les anciens runs ne sont ni modifiés ni repris comme des exécutions de ce nouveau plan.

## Données, graine et matériel

Chaque entraînement FULL utilise tout le train et toute la validation v2 : TSAC10053/1774, MADAR6388/800. Les fichiers sources sont vérifiés par empreinte et les mêmes identifiants sont employés pour tous les taux. Le test officiel reste exclu.

Le criblage utilise uniquement la **graine42**, fixée pour les trois taux. Le mode FULL désigne ici l’intégralité des exemples, pas trois graines : le champ `experiment_scope=screening` distingue explicitement ces résultats exploratoires des runs initiaux ou d’une confirmation finale.

Les trois candidats sont entraînés et vérifiés sur le **même appareil et le même environnement** au sein d’une campagne. Le réglage2e-5 est réexécuté dans la campagne, afin de ne pas confondre une différence de matériel ou de version de journal avec celle du taux. Les anciens scores MPS restent des résultats historiques. Nous ne combinons pas trois candidats exécutés sur des plateformes différentes pour choisir un gagnant.

Le code est commun CPU/MPS/CUDA et ne distribue pas automatiquement les tâches sur les deux GPU de Kaggle. Un seul GPU est utilisé. Les mentions de matériel et les durées sont archivées ; aucun gain de vitesse n’est promis sur la base du petit essai.

## Contrôle avant chaque entraînement complet

Pour chaque taux, un smoke apparié utilise100exemples par partition/tâche, deuxépoques et graine42. Les quatre modèles techniques sont rechargés et leurs prédictions, gradients, poids et métriques par époque sont vérifiés. Le FULL n’est autorisé que si ce smoke de même configuration et appareil est vérifié. Le contrôle apparié sert de garde technique ; les entraînements complets de criblage apprennent seulement les modèles adaptés.

La campagne complète produit **trois smokes appariés**, puis **six adaptations FULL** : trois taux × deux tâches × une graine. Les modèles sont successifs ; ils ne sont pas tous chargés en mémoire. Aucun témoin CLS gelé FULL supplémentaire n’est imposé par cette recherche.

## Sélection et traces

Pour chaque candidat, le meilleur checkpoint est celui du macro-F1 de validation le plus élevé ; en cas d’égalité d’époques, la première est conservée. Après rechargement et contrôles de tous les candidats, nous choisissons **séparément pour chaque tâche** le taux ayant le plus grand macro-F1 de validation. Une égalité exacte entre taux est départagée par le plus petit taux. L’accuracy, les labels discutables, les erreurs relues et le test ne servent pas de critères alternatifs.

Un score de smoke ne peut pas sélectionner le réglage final. Une campagne incomplète, un fichier altéré, un appareil incohérent, des identifiants différents ou un journal d’époques incomplet empêchent la production d’une sélection validée. La synthèse archive les empreintes des résultats et de leurs vérifications, les scores, les candidats et les taux proposés par tâche. Les courbes sont issues des mêmes runs contrôlés.

La sélection exploratoire ne garantit pas une amélioration sur de nouvelles données. La validation a déjà servi à plusieurs comparaisons ; nous explicitons cette limite et limitons la recherche à trois taux comme prévu au cahier (§10 et §13).

## Confirmation ultérieure

Après sélection, une phase distincte confirmera chaque réglage retenu sur les graines13/42/2026, avec journaux complets à chaque époque, sauvegarde, rechargement et cartes mises à jour. Les trois scores seront rapportés avec moyenne et écart-type, sans choisir une meilleure graine. Cette confirmation n’est pas lancée implicitement par le notebook de criblage.

Le test final, le bootstrap, les autres composants et le remplacement des modèles de l’application restent séparés. La création d’un protocole ou le succès d’un smoke ne clôt pas L05/L07/L12.
