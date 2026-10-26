# TuniScope — Journal des expériences

Les premiers entraînements classiques ont été exécutés le 17 octobre 2026. Les audits de données reproductibles et contrôles d'environnement sont également consignés ici.

Créer une entrée pour chaque expérience significative.

## 2026-10-26 — Preuves de la revue Bronze, sans nouvel entraînement

- Objectif : adapter la présentation au guide fourni et rattacher chaque affirmation technique à une preuve visible. Source : guide PDF de 16 pages, SHA-256 `a67a547bbd80cdd57aa24db4984a44112fcc7e9594d4f294c00ff5846c479540`. Travail individuel, aucune seconde annotation humaine inventée.
- Commande réellement exécutée : `sh scripts/project_uv.sh run --locked --offline python scripts/show_validation_results.py --verify --output outputs/bronze_preuves_2026-10-26`. Cinq familles historiques complètes vérifiées, avec rechargement des checkpoints et prédictions recalculées : caractères, mots, combinaison, lexiques et augmentation lexicale. Trois HTML et `summary.json` générés ; anciens runs inchangés. HEAD `2d6900a9fc5d7c2808ea7fb76b4d3b07a3942668`, modifications locales, aucun tag bronze à la vérification.
- Comparaison complète, macro-F1 TSAC/MADAR sur la validation v2 : majorité 0,343206/0,222222 ; lexique 0,720134/0,882974 ; mots 0,872023/0,917075 ; caractères 0,928898/0,949244 ; mots+caractères 0,923817/0,938384. Effectifs 1 774/800. Accuracy caractères 0,9289740698985344/0,95125. La variante règles+négation TSAC obtient 0,718898 contre 0,720134 sans négation ; une erreur corrigée et trois ajoutées. Aucun gain nouveau annoncé.
- Recalcul indépendant des slices caractères depuis leurs prédictions sauvegardées : TSAC AR n=785, F1 0,921349 ; LATIN n=932, 0,938445 ; MIXED n=46, 0,867308 ; UNKNOWN n=11, 0,611765 ; MADAR AR n=800, 0,949244. Les métriques globales de chaque slice correspondent exactement aux champs sauvegardés. Le vérificateur lexical couvre également ses slices : TSAC 0,733399/0,695892/0,771825/0,476190, MADAR AR 0,882974. Ces écritures ne sont pas des labels de langue ; FR/EN/Arabizi restent non identifiés par ce détecteur.
- `notebook_support.py --mode real` réexécuté : TSAC 11 827 × 17 et MADAR 7 188 × 17 en RAM ; zéro groupe partagé train/validation, bruts préservés, zéro ligne test sélectionnée. NFC/espaces change 917/144 textes TSAC et 18/3 MADAR train/validation. Vérification du classeur/registre assisté par `verify_assisted_error_review.py`, sans option d'écriture : 30 cas par tâche, 60 avis humains, 11 catégories humaines et 49 catégories/60 hypothèses IA conservées ; ajouts IA toujours non relus. Huit tests ciblés du lecteur de métriques passent ; pas d'annonce d'une relance de toute la suite.
- Présentation des erreurs : regroupement mécanique des catégories TSAC « avis mixte/ambigu » (9), « manque de contexte » (2) et « incertain » (1) en « ambiguïté/contexte » (12), pour six catégories affichées. Les registres d'origine ne changent pas. Deux cas réels sont référencés sans transfert de leur texte dans Canva : TSAC-01 POS/NEG et MADAR-01 TN/MSA. Le classeur local reste la preuve détaillée ; aucun nouveau CSV d'erreurs n'est revendiqué.
- Partitions : SHA-256 v2 TSAC `ac8a453de6115bea6b51255fe7badc1eab72e2649e0056a1ad918aa963730fa7`, MADAR `3d81826b70437aa37a286d2d8fded745b01295165395b26cf868f91f90204474`. Versionnement v2 dans `dd4d218d5574307fe57cd156ea5e587d220ab518` le 19 octobre ; ne pas confondre cette date avec le gel final de toutes les expériences.
- Support Canva adapté, 14 pages inspectées, révisions explicitement approuvées puis enregistrées. Preuves et trois tableaux dans `docs/33_preuves_revue_bronze.md` ; conducteur privé 14. Les scores sont clairement étiquetés validation, non test. Démo applicative, BM25, latence bout en bout, reproduction publiée, tag bronze, CSV détaillé, PDF et vidéo de secours restent ouverts. Aucun entraînement, modèle neuronal, test final, label ou source modifié, commit ou push par cette étape ; L02/L04/L07/L10/L11/L12 partiels.

## 2026-10-25 — Vérification du guide personnel execution.md

- Objectif : documenter la relance locale et la lecture des colonnes pour la démonstration ; pas une nouvelle expérience d'apprentissage ni une sélection de modèle.
- Contrôle : options de `run_baselines.py`, `show_validation_results.py` et `render_baseline_report.py` consultées ; `show_validation_results.py --run outputs/baselines/2026-10-17_grouped_v2_full --verify` exécuté dans l'environnement uv verrouillé et hors ligne.
- Résultat : six systèmes/variantes rechargés, prédictions et métriques recalculées. Caractères bruts : TSAC accuracy 0,928974 et macro-F1 0,928898 ; MADAR accuracy 0,951250 et macro-F1 0,949244. Références inchangées, aucun nouveau checkpoint appris ni score test.
- Preuve pédagogique : `explanation NLP/execution.md`, ignoré par Git. Les commandes de nouveaux runs y sont fournies pour notre répétition ; elles n'ont pas été exécutées par cette étape. La répétition complète et sa propre entrée d'expérience restent à faire.

## 2026-10-25 — Relecture des résultats et vérification d'une relance complète du notebook

- Objectif : afficher les métriques pour la présentation et vérifier les sauvegardes ; aucun nouveau choix de modèle ni nouvel entraînement lancé par l'étape de reporting. Révision de travail `2d6900a9fc5d7c2808ea7fb76b4d3b07a3942668`, état non commité.
- Relance observée dans le notebook sauvegardé : `outputs/baselines/notebooks/d99058b90e8447fa9cc676ffe1a472d3_characters_smoke/` puis `..._characters_full/`, full commencé le `2026-10-25T17:25:03.146141+00:00`. Configuration caractères v2 inchangée, Python 3.12.15, CPU, un thread, graine 42 et verrouillage existants. Six résultats par run : majorité brute, caractères bruts et normalisés sur TSAC et MADAR. Les deux dossiers ont passé séparément le vérificateur après rechargement.
- Résultats complets caractères bruts, identiques aux références historiques : TSAC accuracy 0,9289740698985344 et macro-F1 0,9288980669985036, 1 774 validations ; MADAR accuracy 0,95125 et macro-F1 0,9492441926133318, 800 validations. TSAC normalisé macro-F1 0,9288927289408286, même accuracy ; MADAR inchangé. Les durées d'apprentissage enregistrées du full pour caractères brut/normalisé sont respectivement TSAC 0,655/0,667 s et MADAR 0,834/0,853 s : elles n'incluent pas le temps total de la démonstration.
- Relecture comparative : les cinq runs historiques complets caractères/mots/combinaison/règles/augmentation ont passé leurs vérificateurs. `scripts/show_validation_results.py --verify --output outputs/presentation_validation_2026-10-25` produit trois pages HTML et un résumé de provenance ; trois PNG capturés dans le navigateur et inspectés. Le mode `--run` affiche aussi les métriques globales/par classe et matrices du nouveau full, sans réentraînement. Code et procédure : `docs/32_relecture_metriques_validation.md` ; huit tests synthétiques ciblés réussis.
- Incident : premier rendu interrompu avant écriture car le comparateur prenait les slices historiques supplémentaires pour une différence de métriques globales. Comparaison corrigée et test de non-régression ajouté ; métriques issues des prédictions inchangées. Le premier essai n'est pas présenté comme réussi.
- Limites : préservation de la copie interactive du notebook, pas de nettoyage automatique des sorties ni d'annonce d'une nouvelle réussite de toute la suite structurelle. Pas de run Kaggle, relance indépendante propre, test officiel, bootstrap, trois graines, poids neuronaux ou application. Corpus, labels, manifestes et anciens runs inchangés. Le support L11 est une présentation d'avancement, pas la soutenance finale ; livrables globaux non clôturés.

## 2026-10-19 — Vérification séquentielle des premiers notebooks

- Objectif : vérifier les nouveaux points d'entrée L03/L04/L12, pas améliorer les scores. Deux notebooks sans sorties réutilisent le chargeur, la normalisation, les configurations et le runner TF-IDF existants. Aucun nouvel algorithme ou réglage.
- Code : `scripts/notebook_support.py`, `scripts/check_notebooks.py`, `tests/test_notebooks.py` ; procédure et commandes dans `docs/31_notebooks_reproductibles.md`. Commit de départ inchangé `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité.
- Vérification fictive : deux notebooks exécutés cellule par cellule dans l'ordre ; tableau inventé 4 × 17 et contrôle scikit-learn réussis, sans lecture des corpus. C'est une vérification séquentielle Python, pas une exécution native Jupyter.
- Vérification réelle données : v2 explicitement, TSAC 10 053/1 774 train/validation (11 827 × 17 en RAM), MADAR 6 388/800 (7 188 × 17), zéro groupe partagé ; normalisation modifie 917/144 et 18/3 lignes respectivement. Bruts conservés, aucune ligne test sélectionnée, aucun fichier de textes préparés écrit.
- Trois smoke runs réels, graine 42, 100 lignes train et 100 validation par tâche ; configurations figées `baselines_grouped_v2.json`, `baselines_words_v2.json`, `baselines_combined_v2.json`. Mac ARM64, CPU, un thread par runner, Python 3.12.15 et dépendances inchangées de `uv.lock`. Les runs mots/combinaison ont été lancés en parallèle : leurs durées ne constituent pas une comparaison contrôlée de coût.
- Dossiers privés : `outputs/baselines/notebooks/a0b7f297741c4a57b2b67a43ea4274bb_characters_smoke/`, `.../1b6923e152c6474eabf226b994971498_words_smoke/`, `.../11872d5cc4dd45a78926b0c2764a0d8b_combined_smoke/`. Chaque `results.json` enregistre paramètres, empreintes, environnement, effectifs, ressources et références des sauvegardes.
- Macro-F1 **smoke uniquement**, brut/normalisé identiques : caractères TSAC 0,703325 et MADAR 0,449464 ; mots 0,546410 et 0,537970 ; combinaison 0,712644 et 0,534202. Majorités communes 0,342105 et 0,222222. Ces petits scores ne remplacent pas les résultats complets historiques ni le choix de référence.
- Chaque série de six résultats a réussi `verify_baseline_run.py` : alignement, rechargement, prédictions et métriques recalculées. 93 tests passent, dont neuf nouveaux contrôles fictifs. Le mode full est arrêté si le smoke échoue ; un nouveau full via notebook n'a pas été exécuté.
- Incident initial : destination de la passerelle refusée car hors du répertoire autorisé `outputs/baselines/`, avant entraînement ; destination corrigée sans modifier le runner historique, puis trois smoke runs réussis. Aucun résultat échoué présenté comme validé.
- Décision : conserver les deux notebooks comme premiers supports exécutables commentés, sans sorties partagées. Jupyter natif, Kaggle/chemins privés et reproductibilité indépendante restent non vérifiés. Aucun téléchargement neuronal, modification des sources/labels/manifestes ou évaluation test ; livrables globaux toujours partiels.

## 2026-10-19 — Ajout lexical et négation au modèle caractères

- Hypothèse et contrastes : tester si les voix du lexique aident le meilleur modèle caractères, malgré la faiblesse du lexique seul. Contraste principal caractères+lexique versus contrôle pour TSAC/MADAR ; contraste secondaire TSAC avec/sans négation dans le même bloc lexical. Contributions partielles L04/L07/L10/L12.
- Protocole/configuration fixés avant évaluation de ces variantes : `docs/30_caracteres_et_indices_lexicaux.md`, `configs/baselines_char_rules_v1.json`, version `character-rule-features-v1`. Manifestes v2, texte brut. Paramètres caractères et régression logistique inchangés par rapport à `baselines_grouped_v2.json` ; un thread CPU, C=1, lbfgs et max_iter=1000. Pas de recherche de paramètres après résultat.
- Représentation ajoutée : fractions de voix par classe puis indicateur de présence d'indice, trois colonnes TSAC/quatre MADAR. Poids fixe 1, sans normalisation globale, valeurs caractères inchangées. Le recours majoritaire des règles n'est pas injecté comme feature. La variante TSAC négation applique les mêmes fenêtres de trois mots aux mêmes ressources ; ses poids de classifieur sont réappris sur train, pas gelés à ceux de l'autre variante.
- Ressources : lexiques et contrôles reconstruits sur train uniquement avec `configs/rule_baselines_v1.json`, identiques aux checkpoints de règles précédents. Aucune annotation IA ou erreur de validation utilisée comme nouvelle étiquette. L'usage de ressources supervisées sur le train même du classifieur est une limite de surajustement explicitée.
- Code : `scripts/character_rule_models.py`, `scripts/run_character_rules.py`, `scripts/verify_character_rules.py`. Commit de départ `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité. Empreintes code/config/protocole au lancement, références historiques, manifestes, dépendances, checkpoints et prédictions enregistrées. Le protocole est ensuite enrichi du bilan, sans modifier la configuration exécutée.
- Runs : `outputs/baselines/2026-10-19_character_rules_v1_smoke/` puis `outputs/baselines/2026-10-19_character_rules_v1_full/`, complet commencé à `2026-10-19T20:50:47.497752+00:00`. Trois modèles TSAC (contrôle, lexical, lexical+négation) et deux MADAR (contrôle, lexical). Smoke sur 100 lignes par partition/tâche, cinq modèles vérifiés avant lancement complet.
- Environnement et données : Python 3.12.15, bibliothèques déjà verrouillées, Mac ARM64 CPU, aucun nouveau paquet/GPU. TSAC 10 053 train/1 774 validation ; MADAR 6 388/800. Les lexiques restent à 400/586 termes. Contrôles à 92 733/53 079 features, augmentations à 92 736/53 083. Aucun texte du test officiel sélectionné.
- Résultats globaux validation TSAC : contrôle macro-F1 0,928898 (126 erreurs), ajout lexical 0,925016 (133), lexical+négation 0,925580 (132). MADAR : contrôle 0,949244 (39), ajout lexical 0,905184 (71). Aucune ligne sans indice supprimée ni score recalculé avec labels ajustés.
- Comparaison appariée : TSAC lexical corrige 43 erreurs et en introduit 50 (93 prédictions différentes, écart F1 −0,003882) ; lexical+négation corrige 42 et en introduit 48 (90 différences, écart −0,003318). MADAR lexical corrige 13 erreurs et en introduit 45, avec une autre modification erroné→erroné (59 différences, écart F1 −0,044060). Négation versus lexical TSAC : 13 différences, sept corrigées et six introduites, mais contrôle toujours meilleur.
- Blocs nuls en validation : 781 TSAC et 126 MADAR sans indice lexical ; caractères nuls sur huit TSAC et zéro MADAR. L'ajout lexical laisse sept vecteurs totaux nuls TSAC contre huit au contrôle, sans en faire une amélioration globale. Toutes ces lignes restent évaluées.
- Ressources mesurées : ajustement complet contrôle/lexical/négation TSAC 1,101/1,436/1,480 s ; contrôle/lexical MADAR 1,082/1,229 s. Durées de prédiction et pics RSS cumulatifs au processus dans le rapport, sans revendication de latence applicative.
- Vérification : dix nouveaux tests fictifs, 84 tests Python réussis au total. Les contrôles reproduisent exactement les prédictions historiques. Vocabulaires/IDF/valeurs du bloc caractères identiques, ressources lexicales reconstruites depuis train, modèles rechargés, toutes prédictions/slices/mesures rejouées et macro-F1 recalculé depuis les matrices. Références caractères et règles historiques également revérifiées.
- Rapport public généré : `reports/character_rules_v1_2026-10-19/`, sans texte de corpus ni vocabulaire/lexique. Modèles augmentés et prédictions privés, anciennes expériences préservées. Décision : conserver caractères seuls comme référence initiale, archiver scientifiquement le résultat négatif sans retuning. Les notebooks, autres baselines requises, relecture humaine et évaluation finale restent à faire ; pas de clôture des livrables ou début de phase neuronale.

## 2026-10-17 — Baselines lexicales induites et portée locale de négation

- Hypothèse : un lexique d'indices appris depuis le train fournit une référence inspectable ; une fenêtre locale de négation pourrait modifier le sentiment, sans gain présupposé. Contributions partielles à L04/L07/L10/L12.
- Protocole fixé avant évaluation de cette configuration : `docs/29_baselines_regles_lexicales.md`, `configs/rule_baselines_v1.json`, version `train-lexicon-rules-v1`, manifestes v2. NFC/casefold, unités alphanumériques, marqueurs techniques et ponctuations de frontière ; émojis non représentés mais exemples conservés.
- Lexique : fréquence documentaire classe versus reste, lissage 1, minimum cinq documents dans la classe, log-rapport de fréquences au moins 1, au plus 200 termes par classe, longueur minimale deux. Les contrôles de négation sont exclus des indices et activés seulement s'ils sont attestés sur train. Vote par terme distinct, majorité train explicitement signalée si absence d'indice ou égalité. Même lexique TSAC pour la variante à fenêtre de trois mots, avec frontières, adversatifs et exception configurée.
- Code : `scripts/rule_baselines.py`, `scripts/run_rule_baselines.py`, `scripts/verify_rule_baselines.py`. Commit de départ `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité. Empreintes du code, config, protocole au moment du run, manifestes, environnement, lexiques et prédictions enregistrées. L'ajout ultérieur du bilan dans le protocole change son empreinte courante, pas la configuration exécutée.
- Environnement : Python 3.12.15 CPU macOS ARM64, dépendances existantes verrouillées, aucun GPU ou nouveau paquet. Construction déterministe ; la graine 42 sert au sous-échantillonnage smoke, pas à sélectionner un résultat final parmi plusieurs graines.
- Runs : `outputs/baselines/2026-10-17_rules_v1_smoke/` puis `outputs/baselines/2026-10-17_rules_v1_full/`, run complet commencé à `2026-10-17T21:41:30.715837+00:00`. Cinq sorties par run : majorité/lexique/lexique+négation TSAC et majorité/lexique MADAR. Le smoke à 100 lignes par partition ne retient aucun terme TSAC avec les seuils fixes ; recours intégral testé, sans abaissement a posteriori des seuils.
- Données complètes : TSAC 10 053 train, 1 774 validation ; MADAR 6 388 train, 800 validation. Aucun exemple test chargé. Lexiques privés : 400 termes TSAC (200 NEG et 200 POS), 586 MADAR (200 MSA, 200 TN, 186 EG), aucune annotation IA ajoutée aux labels.
- Résultats globaux validation : TSAC majorité macro-F1 0,343206 ; lexique 0,720134 (accuracy 0,744645 ; 453 erreurs) ; lexique+négation 0,718898 (accuracy 0,743517 ; 455 erreurs). MADAR majorité 0,222222 ; lexique 0,882974 (accuracy 0,891250 ; 87 erreurs). Tous les recours majoritaires sont inclus.
- Couverture : TSAC 993 lignes avec indice et 973 votes décisifs (54,8478 %), 801 recours ; MADAR 674 avec indice et 635 décisifs (79,3750 %), 165 recours. La couverture ne correspond pas à une probabilité ni à la fraction de sorties ayant un label valide.
- Ablation TSAC : 77 textes déclenchent une fenêtre de négation, neuf contiennent un indice inversé, quatre prédictions finales changent ; une erreur corrigée et trois introduites. Macro-F1 légèrement inférieur, aucune tentative de réglage après résultat. Les deux lexiques sont strictement identiques puisque le même checkpoint est utilisé.
- Comparaison caractères brute v2 : mêmes IDs/groupes/labels vérifiés. Le lexique TSAC corrige 76 erreurs des caractères mais en introduit 403 ; sa variante négation en introduit 405 pour les mêmes 76 corrigées. Le lexique MADAR corrige 13 erreurs et en introduit 61. Les caractères restent la référence initiale la plus forte ; ce contraste compare des systèmes différents, pas une ablation à représentation constante.
- Ressources : construction des lexiques TSAC/MADAR 0,1183/0,0688 s ; prédiction validation lexique TSAC 0,0639 s, variante négation 0,0651 s et MADAR 0,0351 s. Pics RSS cumulatifs et autres durées dans les métadonnées, sans revendication de benchmark de latence applicative.
- Vérification : 74 tests réussis (18 nouveaux fictifs), cinq systèmes smoke puis cinq complets revérifiés par reconstruction train-only et rechargement JSON. Toutes les prédictions, indicateurs, métriques, slices et comptes rejoués ; macro-F1 recalculé depuis les matrices. Référence caractères également revérifiée. Rapport public généré : `reports/rule_baselines_v1_2026-10-17/`.
- Suite : conserver ce résultat négatif, expliquer le recours et la portée, relire les propositions linguistiques et préenregistrer l'éventuel ajout des caractéristiques au classifieur caractères. Pas d'intégration déjà validée, pas de score test, pas de clôture des notebooks ou des livrables.

## 2026-10-17 — Première étude linguistique assistée de 50 textes du train

- Objectif : préparer les règles lexicales et de portée de négation à partir du train, indépendamment des erreurs de validation. Contributions partielles à L04/L10/L12.
- Configuration fixée avant lecture : `configs/linguistic_sample_v1.json`, manifestes v2, train exclusivement, graine 42. Quotas de 25 commentaires TSAC (13 POS, 12 NEG ; neuf AR, huit LATIN, huit MIXED) et 25 phrases MADAR MSA AR. Entre 4 et 50 unités alphanumériques et au plus 320 caractères ; un groupe et un texte NFC/espaces distincts par cas.
- Sélection : ordre SHA-256 de `42|id` dans les strates ; 20 cas enrichis par marqueurs candidats et 30 sans motif reconnu. La regex n'est pas une annotation de négation. Lot qualitatif non proportionnel, sans scores ni prédictions pour choisir les cas.
- Code/environnement : `scripts/export_linguistic_sample.py`, `scripts/render_linguistic_study.py`, Python 3.12 local, bibliothèque standard, aucune nouvelle dépendance. Commit de départ `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité. Empreintes du sélecteur, configuration, manifestes, lot, index et propositions dans les registres privés et le résumé généré.
- Résultats : 50 textes avec première rédaction IA des indices grammaticaux, portée de négation, cible, polarité et réserves pragmatiques. Zéro annotation validée humainement, aucun label de sentiment ajouté à MADAR. L'ironie n'est pas supposée absente quand elle n'est pas établie.
- Observations proposées : 15 négations TSAC et huit MSA au sens élargi du guide (grammaticales, réponses négatives et constructions privatives), sans en faire une vérité de référence. Deux cas dépistés reçoivent une proposition d'absence de négation ; cinq cas non dépistés une proposition de présence. Ces comptes ne sont ni des performances de détection ni des taux de corpus.
- Vérification : reconstitution identique du lot depuis le train v2, empreintes, quotas, identifiants/groupes/textes distincts, extraits observés et champs d'analyse vérifiés. Huit nouveaux tests fictifs ; 56 tests Python réussis au total. La cohérence des chaînes ne certifie pas la justesse linguistique des lectures IA.
- Sorties : notes détaillées dans `explanation NLP/08_etude_linguistique_50_textes_train.md` et registres sous `data/interim/`, tous ignorés par Git. Protocole public `docs/28_etude_linguistique_train.md` et rapport agrégé `reports/linguistic_study_train_v1/`, sans texte source ou référence individuelle.
- Suite : relecture des propositions et conception d'une règle candidate, puis mesure distincte de sa couverture et de son macro-F1 sur validation. Aucune baseline à règles, intégration syntaxique ou fin de livrable revendiquée. Corpus, labels, manifestes, modèles et scores restent inchangés ; aucun exemple de validation/test sélectionné.

## 2026-10-17 — Complément assisté de l'analyse qualitative des erreurs

- Objectif : documenter les catégories et hypothèses manquantes sur délégation explicite, sans les attribuer à une relecture humaine indépendante.
- Sources : lot privé figé de 30 erreurs TSAC et 30 MADAR, validation v2 de la référence caractères brute ; instantané humain `215d8f0fdce9`. Aucun test officiel chargé. Commit de départ `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité.
- Code : `scripts/complete_error_review.mjs` et `scripts/verify_assisted_error_review.py`, runtime documentaire Node/artifact-tool et Python local. Aucun entraînement ni dépendance ajoutée.
- Résultat : 60 avis humains et 11 catégories humaines conservés. Ajout de 49 catégories et 60 hypothèses IA dans une copie privée distincte. Le registre conserve l'auteur de chaque champ ; validation humaine des ajouts encore fausse. Le classeur original est inchangé.
- Comptes principaux : TSAC neuf catégories avis mixte/ambigu, sept Autre, cinq orthographe/Arabizi, trois émoji/symbole, deux manque de contexte, deux négation, un mélange de langues et un incertain. MADAR 15 orthographe/morphologie, 11 variétés proches/texte commun, trois lexique partagé et un texte très court. Ces catégories principales décrivent le lot non proportionnel, pas des causes prouvées ou des taux du corpus.
- Réserve : l'une des deux catégories de négation est un choix humain conservé malgré l'absence de négation explicite visible ; les lectures incertaines et l'ironie possible ne sont pas forcées. Aucun label ou score corrigé.
- Vérification : dix assertions fictives de complétion, réouverture du XLSX, contrôle indépendant XML des valeurs, validations, largeurs et limites des volets figés, compteurs à 30/30. L'export réinitialise seulement la position de défilement. La suite Python totale compte 48 tests réussis ; aucun essai dans Excel natif revendiqué.
- Provenance : empreinte de la copie complétée `84c8c4656440fb8c663c45a844fa32c5ceac833f0dcb4ce8c0dff05b53fdbf04` ; registre privé et rapport agrégé généré `reports/error_analysis_assisted_2026-10-17/` contenant les empreintes du lot, des propositions, scripts et classeurs. Aucun texte de corpus ou note libre dans le rapport public.
- Suite : relire les ajouts IA, notamment les réserves, puis poursuivre l'analyse linguistique et les règles sur ressources issues du train. L07/L10/L12 restent partiels. L'étude des 50 phrases, les futurs modèles, les notebooks et l'évaluation finale ne sont pas remplacés par ce lot.

## 2026-10-17 — Import partiel de la revue humaine des erreurs

- Objectif : conserver les réponses humaines effectivement saisies et distinguer la revue des labels de l'analyse des causes, sans inférer de décision à partir des blancs.
- Code : `scripts/import_error_review.mjs`, runtime documentaire Node/artifact-tool pour la lecture ; aucun entraînement ni nouvelle dépendance. Commit de départ `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité.
- Source : classeur privé de 30 cas TSAC et 30 MADAR de la référence caractères brute, validation v2. Empreinte du classeur commençant par `215d8f0fdce9` ; instantané immuable sous `data/interim/baseline_error_review_decisions_2026-10-17_215d8f0fdce9.json`, avec empreintes complètes de l'entrée, de l'index et du run source. Le XLSX n'a pas été réécrit.
- Résultats : 30 avis TSAC (23 plausibles, quatre discutables, deux incertains, un non interprétable) et 30 MADAR (24 plausibles, six discutables). Cinq catégories TSAC et six MADAR choisies ; aucune explication en G. Aucun cas E/F/G complet, aucune convention de remplissage automatique appliquée.
- Intégrité : cas, labels, prédictions et références vérifiés contre l'entrée figée, manifestes et run source toujours conformes. Une seule différence de texte de lecture : espace final supprimé, sans changement du reste du contenu ; écart enregistré, sources préservées.
- Vérification : huit contrôles fictifs de l'importateur réussis, rapprochement indépendant des réponses depuis le XML du XLSX et 44 tests Python réussis. Aucun texte test chargé et aucune métrique de classification recalculée après réétiquette.
- Rapport agrégé généré : `reports/error_review_progress_2026-10-17_215d8f0fdce9/`, sans commentaires protégés ni notes libres. L'échantillon reste qualitatif et la revue individuelle ne fournit ni accord interannotateurs ni taux réel de labels faux.
- Suite : compléter progressivement F et G avec indices et hypothèses, ou expliciter une incertitude. L07/L12 restent partiels ; un avis « plausible » n'explique pas encore pourquoi le modèle s'est trompé.

## 2026-10-17 — Préparation de la revue qualitative des erreurs classiques

- Objectif : préparer un diagnostic humain de la référence caractères brute v2, sans modifier ses prédictions ou labels. Contribution partielle à L07/L12, pas d'analyse humaine achevée.
- Run source : `outputs/baselines/2026-10-17_grouped_v2_full/`, vérifié par rechargement des six checkpoints et recalcul des métriques. Code : `scripts/export_baseline_error_review.py`, `scripts/build_error_review_workbook.mjs` ; provenance du run, manifestes, prédictions et extraction enregistrée dans les entrées privées. Commit de départ `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité.
- Données : validation v2 seulement. TSAC 126 désaccords dans 124 groupes ; MADAR 39 dans 34. Aucun nouveau modèle appris et aucun texte test chargé.
- Sélection : 30 cas distincts par tâche, groupes et textes NFC/espaces distincts. Cycle entre strates label/prédiction/écriture, plus petites d'abord ; ordre stable SHA256 de `42|id`. Échantillon qualitatif non proportionnel, inadapté à une estimation de prévalence des causes.
- Résultat : classeur privé avec feuilles TSAC et MADAR, 60 cas au total, sources et prédictions préremplies, toutes les cellules d'avis/catégorie/hypothèse vides. Exact raw JSON et index sans texte sous dossiers ignorés. Cinq chaînes TSAC contiennent littéralement `_x000D_` dans les sources ; l'affichage explicite `⟦x000D⟧` les signale sans changer les données ou entrées des modèles.
- Environnement : extraction Python 3.12.15 du projet ; création et rendu du tableur via Node et artifact-tool du runtime documentaire fourni. Aucune dépendance ajoutée à l'environnement de modélisation.
- Vérification : 44 tests locaux réussis, dont quatre tests fictifs du sélectionneur ; données source et prédictions alignées, unicité des cas/familles/textes, champs humains vides, compteur réactif aux saisies partielles/complètes puis restauré, listes de choix, rendu des deux feuilles et textes longs, réouverture et contrôles XML. Le rendu et l'export ne prouvent pas un essai dans Excel natif.
- Suite : revue humaine des champs E/F/G, sans déduire le sens des blancs. Les catégories restent des hypothèses, aucun accord interannotateurs annoncé, aucune réétiquette ou correction rétroactive de score. Voir `docs/26_preparation_analyse_erreurs_baselines.md` ; le travail linguistique des 50 phrases reste distinct.

## 2026-10-17 — TF-IDF mots + caractères réellement combinés

- Objectif : tester une combinaison entraînée des deux représentations, sans vote ni sélection après connaissance du label.
- Protocole/configuration fixés avant exécution : `docs/25_baseline_mots_caracteres_combines.md`, `configs/baselines_combined_v2.json`, version `baseline-combined-v2`. Branches identiques aux références mots 1–2 et caractères 3–5, poids 1:1, concaténation puis L2 globale ; régression logistique C=1, lbfgs, max_iter=1000, sans recherche de paramètres après les scores.
- Commit de départ : `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité ; empreintes du code, de la configuration, des manifestes et de `uv.lock` enregistrées.
- Runs : `outputs/baselines/2026-10-17_combined_v2_smoke/` (100 lignes par partition et tâche, graine 42) puis `outputs/baselines/2026-10-17_combined_v2_full/`, début à `2026-10-17T10:30:22.012068+00:00`. Six systèmes par run, majorité incluse.
- Données/environnement : TSAC 10 053 train / 1 774 validation et MADAR 6 388 / 800, manifestes v2 inchangés, aucun texte test chargé. Mac ARM64, CPU, un thread, Python 3.12.15, scikit-learn 1.9.1, aucune dépendance ajoutée.
- Résultats validation : TSAC combiné macro-F1 0,923817, accuracy 0,923901, 135 erreurs ; MADAR 0,938384, accuracy 0,942500, 46 erreurs. Brut/NFC-espaces : mêmes prédictions. Majorités inchangées : 0,343206 et 0,222222.
- Couverture/dimensions : TSAC 14 vecteurs nuls train / 7 validation ; MADAR 0/0, tous conservés. En brut, TSAC 10 296 mots + 92 733 caractères = 103 029 colonnes ; MADAR 7 070 + 53 079 = 60 149. Dimensions et paramètres entraînés par variante dans le rapport.
- Coût d'ajustement : TSAC brut/normalisé 0,761/0,691 s ; MADAR 0,965/0,937 s. Le pic mémoire est cumulatif au processus, pas isolé par modèle.
- Vérification : 40 tests locaux réussis ; six systèmes smoke et six complets vérifiés après rechargement, métriques recalculées. `scripts/compare_combined_baselines.py` vérifie les trois runs, mêmes IDs/groupes/labels, environnement/classifieur, et identité exacte des vocabulaires et IDF de chaque branche avec les références seules. Rapports générés : `reports/baselines_combined_v2_2026-10-17/` et `reports/three_representations_v2_2026-10-17/`.
- Écarts en brut : macro-F1 combiné − caractères de −0,005082 TSAC et −0,010861 MADAR ; combiné − mots de +0,051794 et +0,021308. Face aux caractères, le combiné corrige 21 erreurs TSAC mais en introduit 30 ; sur MADAR, 12 corrigées et 19 introduites.
- Décision : résultat négatif conservé, caractères retenus comme référence principale actuelle. Cette combinaison précise ne prouve pas que toute combinaison échouerait. Pas de conclusion de significativité ou de test final ; poursuivre analyses linguistiques/erreurs et règles. L04/L07/L12 restent partiels.

## 2026-10-17 — TF-IDF mots 1–2 et comparaison aux caractères v2

- Objectif : ajouter la référence mots du cahier et comparer deux représentations sur les mêmes partitions v2, sans toucher au test.
- Protocole/configuration écrits avant exécution : `docs/24_baseline_mots_et_comparaison.md`, `configs/baselines_words_v2.json`, version `baseline-words-v2`. Mots Unicode avec éléments d'une lettre et chiffres Arabizi, ponctuation/émojis non représentés ; n-grammes 1–2, min_df=2, sublinear_tf=True, lowercase=False, L2, régression logistique C=1, lbfgs, max_iter=1000.
- Commit de départ : `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité. Configuration complète et empreintes du code, des manifestes et de `uv.lock` dans les résultats.
- Runs : `outputs/baselines/2026-10-17_words_v2_smoke/` (100 lignes train/validation par tâche, graine 42) puis `outputs/baselines/2026-10-17_words_v2_full/`, début à `2026-10-17T09:58:04.078058+00:00`. Six systèmes par run, incluant la majorité.
- Données : TSAC 10 053 train / 1 774 validation ; MADAR 6 388 / 800, sources et labels inchangés. Vocabulaire et IDF ajustés sur train seulement ; aucune ligne test chargée.
- Environnement : Python 3.12.15, scikit-learn 1.9.1, Mac ARM64, CPU, un thread ; aucun nouveau paquet.
- Résultats mots : TSAC macro-F1 0,872023, accuracy 0,872041, 227 erreurs ; MADAR macro-F1 0,917075, accuracy 0,922500, 62 erreurs. Brut et normalisé donnent les mêmes prédictions pour chaque corpus. Majorités inchangées : 0,343206 et 0,222222.
- Couverture : vecteurs mots nuls TSAC 744 train / 172 validation ; MADAR 46 / 11. Aucun retrait de ces lignes. Ces comptes agrègent segmentation et vocabulaire retenu, pas seulement l'absence de mots.
- Coût : ajustement TSAC brut/normalisé 0,109/0,101 s, MADAR 0,120/0,119 s ; vocabulaires 10 296 et 7 070, solveur 31 et 47 itérations. Mémoire cumulée, tailles/checksums des checkpoints et métriques par classe/écriture consignées dans le rapport généré.
- Vérification : 31 tests locaux réussis, six résultats smoke et six complets vérifiés indépendamment, checkpoints rechargés sur toutes les prédictions. Le run caractères v2 historique reste vérifiable. Rapports mots dans `reports/baselines_words_v2_2026-10-17/`.
- Comparaison : `scripts/compare_baseline_runs.py` vérifie les runs, mêmes empreintes des manifestes, environnement, classifieur et alignement des IDs/groupes/labels ; recalcule les écarts depuis les prédictions. En brut, écart macro-F1 mots − caractères de −0,056875 TSAC et −0,032169 MADAR. Les mots corrigent 41 erreurs TSAC et 15 MADAR de la référence caractères, mais font respectivement 142 et 38 erreurs sur des cas que les caractères réussissent. Rapport généré : `reports/word_vs_character_v2_2026-10-17/`.
- Décision : conserver les caractères comme référence principale actuelle, sans annoncer de supériorité générale ou significative. Tester séparément la combinaison et analyser les erreurs ; L04/L07/L12 restent partiels.

## 2026-10-17 — Revue humaine et manifestes v2

- Objectif : corriger le partitionnement des variantes TSAC à répétitions indépendamment des résultats des modèles.
- Décisions : les 12 cellules sont explicitement « Même sens », zéro cellule vide ou décision inférée. L'importateur compare chaque référence et texte à l'entrée figée, préserve le classeur annoté et écrit un registre privé avec empreintes.
- Code : `scripts/import_repetition_review.mjs`, `scripts/build_manifests_v2.py` ; données source et v1 vérifiées, aucun test texte chargé.
- Méthode : conserver les 374 contraintes internes v1 et les 49 familles du filtre de répétitions dans une seule partition. Seules les neuf familles initialement croisées ont été relues ; les autres sont des contraintes candidates conservatrices.
- Résultats : 11 441 groupes, 225 multi-lignes couvrant 611 références, huit mixtes par label ; zéro famille répétée traversant train/validation. Même graine 42 et quotas par label ; 1 476 références changent de partition dans chaque sens. Effectifs TSAC 10 053 train / 1 774 validation, MADAR inchangé ; exclusions et test inchangés. Aucune modification des textes ou labels.
- Vérification : quatre tests supplémentaires, soit 24 tests locaux réussis ; reconstruction déterministe et identité des copies contrôlées. Preuve détaillée : `docs/23_manifestes_v2_et_baselines.md`.

## 2026-10-17 — Baselines complètes sur groupes v2

- Commit de départ : `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité ; empreintes du code, de la configuration, des manifestes et de `uv.lock` enregistrées.
- Configuration : `configs/baselines_grouped_v2.json`, version `baseline-grouped-v2`, manifestes v2, graine 42. Hyperparamètres inchangés par rapport au run initial ; aucun choix guidé par les nouveaux scores.
- Runs privés : `outputs/baselines/2026-10-17_grouped_v2_smoke/` (100 lignes par partition/tâche) puis `outputs/baselines/2026-10-17_grouped_v2_full/` (complet), démarré à `2026-10-17T09:49:44.965487+00:00`.
- Environnement : Mac ARM64, Python 3.12.15, scikit-learn 1.9.1, un thread CPU. Ni modèle neuronal ni test officiel chargé.
- Résultats validation : TSAC majorité macro-F1 0,343206 ; caractères brut 0,928898, normalisé 0,928893, accuracy commune 0,928974, 126 erreurs par variante et deux prédictions différentes. MADAR majorité 0,222222 ; caractères brut/normalisé 0,949244, accuracy 0,951250, 39 erreurs et mêmes prédictions.
- Coût d'ajustement complet : TSAC caractères brut environ 0,676 s, normalisé 0,706 s ; MADAR 0,830 s et 0,848 s. Tailles, vocabulaires, itérations et pic mémoire cumulatif du processus dans le rapport généré.
- Vérification : six résultats smoke puis six résultats complets contrôlés, checkpoints rechargés, prédictions alignées, métriques recalculées. Le run historique v1 a aussi été revérifié avec le choix de version rétrocompatible. Rapport agrégé : `reports/baselines_grouped_v2_2026-10-17/`.
- Interprétation : pas de gain établi pour NFC/espaces dans cette configuration. Les exemples TSAC changent entre v1 et v2 : la différence de score n'est pas une estimation isolée de l'effet des groupes corrigés. L02/L04/L07/L12 restent partiels ; poursuivre les comparaisons classiques sur v2 explicite et l'analyse linguistique/d'erreurs.

## 2026-10-17 — Préparation de la revue des variantes à répétitions

- Objectif : matérialiser la revue de toutes les familles candidates train/validation signalées par l'audit de réduction des lettres répétées, indépendamment des prédictions des modèles.
- Données : TSAC, manifestes v1, train et validation seulement ; neuf familles, 21 références source (11 train et 10 validation), 12 comparaisons croisées. Aucun test officiel chargé.
- Code : `scripts/export_repetition_review.py`, `scripts/build_repetition_review_workbook.mjs`. Corpus et manifeste vérifiés par le chargeur ; empreintes du manifeste, du script d'extraction et de l'entrée privée enregistrées dans les index.
- Environnement : Python et Node du runtime documentaire fourni pour l'export du classeur ; aucun paquet supplémentaire installé dans l'environnement de modélisation.
- Résultat : classeur privé exporté et rouvert, textes et références vérifiés, 12 décisions vides, listes de choix et compteur testés, rendu inspecté. Aucune décision sémantique n'est déduite automatiquement.
- Suite : compléter les 12 décisions humaines puis examiner une éventuelle révision versionnée des groupes. Aucun corpus, manifeste, label, modèle ou score n'a été modifié par cette opération ; L02 reste partiel.

## 2026-10-17 — Smoke tests des premières baselines

- Commit de départ : `384f8a5` ; nouveaux scripts et configuration non commités, empreintes enregistrées dans chaque dossier.
- Configuration : `configs/baselines_initial.json`, graine d'échantillonnage 42, 100 lignes train et 100 validation par tâche depuis les manifestes v1 ; aucune ligne test.
- Runs privés : `outputs/baselines/2026-10-17_smoke_v1/` puis `2026-10-17_smoke_v2/`. Chaque run vérifie majorité et TF-IDF caractères brut/normalisé pour les deux corpus.
- Matériel : Mac ARM64, Python 3.12.15, scikit-learn 1.9.1, un thread CPU.
- Résultat : les six systèmes de v2 et leurs checkpoints rechargés ont passé la vérification indépendante des prédictions et métriques. Le premier run avait produit des avertissements pour des sous-groupes à classe unique ; le calcul de balanced accuracy a été rendu explicite à partir de la matrice à classes fixes avant v2. Aucun paramètre appris n'a été sélectionné sur ces scores techniques.
- Suite : autoriser l'exécution complète de la même configuration ; ne pas présenter les scores du sous-échantillon comme des résultats finaux.

## 2026-10-17 — Premières baselines complètes et ablation minimale

- Commit de départ : `384f8a5aba14650b75ebd7a47a558433b56553ce`, état non commité ; snapshots de configuration et empreintes du code, des manifestes v1 et de `uv.lock` dans les résultats.
- Objectif : établir la référence majoritaire et la première baseline caractères, puis comparer raw-v0 à norm-nfc-space-v1 sans changer les exemples.
- Run : `outputs/baselines/2026-10-17_initial_v1/`, démarré à `2026-10-17T08:11:46.346204+00:00` ; configuration `configs/baselines_initial.json`, préenregistrement `docs/19_protocole_ablation_normalisation.md`.
- Données : TSAC 10 053 train / 1 774 validation ; MADAR 6 388 / 800. Le test officiel n'a pas été chargé. Le vocabulaire TF-IDF est ajusté sur train uniquement.
- Modèles : majorité ; TF-IDF caractères 3–5, min_df=2, lowercase=False, sublinear_tf=True ; régression logistique C=1, solver=lbfgs, max_iter=1000, sans pondération de classe. Le solveur utilisé est déterministe avec un thread ; la graine 42 concerne les sous-échantillons et la configuration, pas trois entraînements stochastiques finaux.
- Matériel/environnement : Mac ARM64, CPU, un thread ; Python 3.12.15, scikit-learn 1.9.1 ; versions complètes, temps, mémoire cumulée et tailles des checkpoints dans le rapport généré.
- Résultats : TSAC majorité macro-F1 0,343206 ; TF-IDF brut et normalisé 0,931145, accuracy 0,931229. MADAR majorité 0,222222 ; TF-IDF brut et normalisé 0,949244, accuracy 0,951250. Les deux variantes donnent les mêmes prédictions : 122 erreurs TSAC et 39 MADAR.
- Vérification : six artefacts rechargés, empreintes et identifiants contrôlés ; accuracy et macro-F1 recalculés depuis les matrices de confusion par `scripts/verify_baseline_run.py`. Rapport agrégé généré dans `reports/baselines_initial_2026-10-17/`. Vingt tests locaux réussissent.
- Limites/suite : scores de validation provisoires, aucun résultat final/test ou bootstrap. Revoir les neuf groupes candidats liés aux répétitions : leur conservation graphique ne prouve pas leur indépendance train/validation. Compléter baselines mots/règles, comparaisons et analyse d'erreurs ; L04/L07/L12 restent partiels.

## 2026-10-17 — Installation et contrôle de l'environnement CPU des baselines

- Commit Git de départ : `384f8a5` ; les fichiers de cet environnement ne sont pas encore commités.
- Objectif : construire un environnement local reproductible et vérifier les bibliothèques avant les premiers essais de modèle sur corpus.
- Données : exemples inventés pour le contrôle de bibliothèques ; conversion pandas des manifestes v1 TSAC/MADAR sur train/validation pour vérifier le chargeur. Aucun test officiel chargé et aucun score sur corpus.
- Configuration : `.python-version`, `pyproject.toml`, `uv.lock` et export `requirements.txt` ; uv 0.13.1, Python 3.12.15.
- Matériel et paquets : Mac ARM64, CPU ; NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, threadpoolctl 3.7.0.
- Résultats : contrôle fictif Unicode/DataFrame, TF-IDF creux et minuscule classifieur réussi ; 18 tests de données réussis ; dix paquets compatibles selon uv. DataFrames réels train/validation de 11 827 × 17 (TSAC) et 7 188 × 17 (MADAR), brut préservé. L'audit NFC/espaces sous Python 3.12 reproduit les comptes précédents.
- Observation : l'analyseur TF-IDF de caractères harmonise déjà certaines suites d'espaces ; réserve ajoutée au protocole d'ablation avant tout score réel.
- Suite : exécuter le contrôle sur Kaggle/Colab, puis préparer la première baseline. Le verrouillage comprend des distributions Linux compatibles, mais aucune exécution Linux n'a été validée. L12 reste partiel.

## Modèle d'entrée

```markdown
## AAAA-MM-JJ — nom court de l'expérience

- Commit Git :
- Objectif :
- Corpus et version du manifeste :
- Graine du partitionnement :
- Modèle et révision :
- Fichier de configuration :
- Matériel et environnement :
- Résumé des résultats :
- Erreurs et observations :
- Décision ou prochaine action :
```

## 2026-10-04 — Audit initial des sources (intégrité et structure)

- Commit Git de départ : `2d9ed45` ; le script de cet audit n'était pas encore commité lors de l'exécution.
- Objectif : vérifier les archives, leurs copies extraites, l'encodage et les effectifs bruts sans modifier les corpus.
- Corpus et version du manifeste : MADAR v1.1 et TSAC, archives identifiées dans `SOURCE_REGISTER.md` ; aucun manifeste créé.
- Graine du partitionnement : sans objet ; aucune nouvelle partition.
- Modèle et révision : sans objet.
- Fichier de configuration : `scripts/audit_structure.py`, Python standard 3.9.6.
- Matériel et environnement : Mac arm64 ; aucun GPU utilisé.
- Résumé des résultats : archives conformes aux SHA-256 enregistrés, ZIP intègres, 34 fichiers MADAR et 6 fichiers TSAC identiques après extraction ; huit fichiers utiles en UTF-8. Quatre lignes vides dans le train TSAC, aucune dans son test ; aucune anomalie structurelle dans les quatre TSV MADAR.
- Erreurs et observations : deux noms de fichiers macOS `Icon` vides perdent leur retour chariot final lors de l'extraction ; cas traité explicitement dans le script. Doublons et chevauchements non encore vérifiés.
- Décision ou prochaine action : ne modifier aucun original ; effectuer les contrôles de groupes, doublons et conflits avant de figer les manifestes. Détails et empreintes dans `docs/03_data_audit_structure.md`.

## 2026-10-04 — Audit des groupes MADAR et des doublons exacts TSAC

- Commit Git vérifié lors de la dernière relance : `7010dba` (script versionné à cette révision ; rapport encore en cours de rédaction).
- Objectif : vérifier le regroupement des traductions MADAR et mesurer répétitions, conflits POS/NEG et recouvrements train–test TSAC sans modifier les sources.
- Corpus et version du manifeste : MADAR v1.1 Corpus-26 et TSAC original ; aucun manifeste créé.
- Graine du partitionnement : sans objet ; partitions officielles observées sans changement.
- Modèle et révision : sans objet.
- Fichier de configuration : `scripts/audit_duplicates.py`, Python standard 3.9.6.
- Matériel et environnement : Mac arm64 ; aucun GPU utilisé.
- Résumé des résultats : 2 000 groupes MADAR complets et cohérents, mais quatre chaînes identiques entre partitions ; TSAC comporte 846 groupes de doublons exacts, dont 316 présents dans train et test, et 13 groupes à labels POS/NEG contradictoires. Comptage principal confirmé par `awk` indépendant.
- Erreurs et observations : les chiffres sont agrégés ; aucun texte protégé n'a été publié. Les quasi-doublons et la revue humaine des cas restent à faire.
- Décision ou prochaine action : ne supprimer ni relabelliser aucune ligne ; examiner les cas privés et choisir une politique de décontamination avant les manifestes. Détails dans `docs/04_data_audit_groups_and_duplicates.md`.

## 2026-10-04 — Dépistage des quasi-doublons TSAC train–test

- Commit Git de départ : `6c9c7e8` ; le script de dépistage n'était pas encore commité lors de l'exécution.
- Objectif : repérer, sans décision automatique, des commentaires très similaires entre le train et le test officiel TSAC.
- Corpus et version du manifeste : TSAC original ; aucun manifeste créé.
- Graine du partitionnement : sans objet ; aucune partition créée ou modifiée.
- Modèle et révision : sans objet ; méthode heuristique de trigrammes de caractères et `difflib.SequenceMatcher` de Python 3.9.6.
- Fichier de configuration : seuils inscrits dans `scripts/screen_near_duplicates.py` ; index de revue locaux sous `data/interim/`, exclus de Git.
- Matériel et environnement : Mac arm64 ; aucun GPU ni paquet externe utilisé.
- Résumé des résultats : 324 formes identiques train–test après NFKC/espaces regroupés (467 lignes test) ; 124 paires supplémentaires candidates, concernant 98 lignes test. Deux paires candidates peuvent rapprocher des labels opposés.
- Erreurs et observations : dépistage conservateur et non exhaustif ; 139 formes test très courtes exclues de la recherche de quasi-doublons, mais non des contrôles exacts. Aucune paire candidate n'a été validée humainement.
- Décision ou prochaine action : revue privée des cas et politique de décontamination avant les manifestes. Voir `docs/05_data_audit_near_duplicate_screen.md`.
