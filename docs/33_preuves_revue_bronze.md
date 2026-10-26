# Revue Bronze : correspondance entre affirmations et preuves

**Date :** 26 octobre 2026. Contributions partielles à L02, L04, L07, L10, L11 et L12. Ce document indexe des preuves existantes et leur relecture ; il ne déclare ni la conformité complète Bronze ni l'ouverture de l'évaluation finale.

## Cadre retenu

Le guide institutionnel de revue Bronze fourni le 26 octobre demande six blocs en 15 minutes, puis 10 minutes de questions. Nous appliquons les exigences communes au projet TuniScope, consacré à la classification de variété et de sentiment, sans lui attribuer les composants de traduction ou de segmentation des autres sujets du guide.

Le travail est individuel : un seul auteur assure les fonctions données, règles, représentations, évaluation et intégration. Les propositions IA sont déclarées et ne comptent pas comme un second annotateur humain. Aucun kappa interannotateurs ne peut être annoncé sur cette revue individuelle.

**Frontière majeure :** nos résultats sont calculés sur la validation v2. Le guide attend des mesures sur test fixé. Nous n'avons pas rebaptisé la validation « test », ni évalué le test officiel pour la seule préparation de slides. Son ouverture exige le gel explicite des configurations et du contraste à évaluer.

## Registre des preuves

| Affirmation | Preuve à ouvrir | Portée et limite |
| --- | --- | --- |
| Corpus acquis et documentés | `SOURCE_REGISTER.md`, cartes `docs/12_carte_donnees_tsac.md` et `docs/13_carte_donnees_madar.md` | Conditions d'accès contrôlées ; cartes encore partielles, droits des dérivés à vérifier. Aucun lien privé d'acquisition dans Canva. |
| Qualité contrôlée | `docs/04_data_audit_groups_and_duplicates.md`, `docs/07_audit_union_exclusions_tsac.md` | 846 groupes exacts TSAC, 316 entre train et test, 13 conflits ; 1 842 exclusions train selon l'union des motifs, pas 1 842 labels incorrects. |
| Partitions et exclusions traçables | `data/manifests/v2/summary.json`, CSV de manifestes et exclusions | TSAC 10 053/1 774/3 400 ; MADAR 6 388/800/800. Aucun original ou label modifié par cette relecture. |
| Chargement réel vérifié | `scripts/notebook_support.py --mode real` | V2 : DataFrames TSAC 11 827 × 17 et MADAR 7 188 × 17 ; zéro groupe train/validation partagé ; zéro ligne test sélectionnée. Ce n'est pas une API sur texte libre. |
| Normalisation conservatrice | `scripts/normalize_text.py:14`, document 16 | NFC et espaces en copie. La transformation conserve notamment négation, emojis et répétitions ; elle n'identifie pas la langue. |
| Classifieurs réellement entraînés | `scripts/run_baselines.py:129`, `scripts/baseline_models.py`, configurations v2 | Le pipeline est ajusté sur train et prédit la validation ; code source cliquable dans le support. |
| Résultats retrouvables | `outputs/baselines/*_full/results.json` et `EXPERIMENTS.md` | Chaque chiffre est associé à une méthode, une variante, une partition et un effectif. |
| Modèles rechargés | `scripts/verify_baseline_run.py`, `verify_rule_baselines.py`, `verify_character_rules.py` | Cinq familles historiques vérifiées le 26 octobre ; ce contrôle n'entraîne pas un nouveau modèle. |
| Erreurs documentées | `reports/error_analysis_assisted_2026-10-17/summary.json`, registre et classeur privés | 30 cas par tâche, 60 avis humains ; 49 catégories et 60 hypothèses IA restent à relire. Vérificateur XML réexécuté sans réécrire les classeurs. |
| Relecture et supports agrégés | `outputs/bronze_preuves_2026-10-26/summary.json` et trois HTML | Nouvelle génération après vérification ; uniquement résultats agrégés. Les aperçus des slides ne remplacent pas la démo demandée par le guide. |

## Tableau 1 : résultats par composant, validation uniquement

La métrique est le **macro-F1**, à maximiser. Les scores test de toutes ces lignes sont **non évalués**.

| Tâche | Méthode | Référence triviale : majorité | Score de la méthode | N validation |
| --- | --- | ---: | ---: | ---: |
| TSAC | Lexique | 0,343206 | 0,720134 | 1 774 |
| TSAC | TF-IDF mots | 0,343206 | 0,872023 | 1 774 |
| TSAC | TF-IDF caractères | 0,343206 | 0,928898 | 1 774 |
| TSAC | Mots et caractères | 0,343206 | 0,923817 | 1 774 |
| MADAR | Lexique | 0,222222 | 0,882974 | 800 |
| MADAR | TF-IDF mots | 0,222222 | 0,917075 | 800 |
| MADAR | TF-IDF caractères | 0,222222 | 0,949244 | 800 |
| MADAR | Mots et caractères | 0,222222 | 0,938384 | 800 |

Ces méthodes utilisent les mêmes exemples v2 et le texte brut. La règle avec négation TSAC obtient 0,718898 contre 0,720134 sans négation. L'ablation caractères brut/NFC-espaces donne 0,928898/0,928893 sur TSAC et 0,949244/0,949244 sur MADAR. Les ajouts lexicaux aux caractères restent des résultats négatifs conservés, pas de nouvelles méthodes sélectionnées.

Sources : `outputs/baselines/2026-10-17_grouped_v2_full/`, `2026-10-17_words_v2_full/`, `2026-10-17_combined_v2_full/`, `2026-10-17_rules_v1_full/` et `2026-10-19_character_rules_v1_full/`. Rapports agrégés correspondants sous `reports/`. La vérification complète est lancée par `show_validation_results.py --verify --output <nouveau_dossier>` ; le dossier doit être nouveau.

## Tableau 2 : résultats par écriture, validation uniquement

Les lignes décrivent la **présence de caractères Unicode**, pas une annotation de langue. Nous ne transformons pas LATIN en Arabizi, français ou anglais, et MIXED ne prouve pas le code-switching. Les colonnes FR/EN/Arabizi du guide restent non renseignables avec cette seule détection.

| Tâche | Écriture | N | Macro-F1 caractères | Macro-F1 lexique |
| --- | --- | ---: | ---: | ---: |
| TSAC | AR | 785 | 0,921349 | 0,733399 |
| TSAC | LATIN | 932 | 0,938445 | 0,695892 |
| TSAC | MIXED | 46 | 0,867308 | 0,771825 |
| TSAC | UNKNOWN | 11 | 0,611765 | 0,476190 |
| MADAR | AR | 800 | 0,949244 | 0,882974 |

Les effectifs des slices somment aux tailles de validation. UNKNOWN contient seulement 11 cas TSAC, dont un NEG : ne pas généraliser à partir de ce sous-ensemble. Les métriques caractères ont été recalculées indépendamment depuis les prédictions sauvegardées le 26 octobre ; les règles ont également passé leur vérificateur dédié, incluant leurs slices.

## Tableau 3 : catégories des erreurs de validation examinées

Ce lot qualitatif n'est pas proportionnel et n'est pas un échantillon de test. Nous conservons les catégories humaines et proposées par IA. Pour limiter la présentation à six catégories TSAC, nous regroupons **Avis mixte / ambigu (9)**, **Manque de contexte (2)** et **Incertain (1)** sous **Ambiguïté / contexte (12)**. C'est un regroupement de présentation, pas une nouvelle annotation ni une cause confirmée.

| Tâche | Catégorie présentée | Nombre sur 30 |
| --- | --- | ---: |
| TSAC | Ambiguïté / contexte | 12 |
| TSAC | Autre | 7 |
| TSAC | Orthographe / Arabizi | 5 |
| TSAC | Emoji / symbole | 3 |
| TSAC | Négation proposée | 2 |
| TSAC | Mélange de langues | 1 |
| MADAR | Orthographe / morphologie | 15 |
| MADAR | Variétés proches / texte commun | 11 |
| MADAR | Lexique partagé | 3 |
| MADAR | Texte très court | 1 |

La catégorie de négation conserve une réserve déjà documentée : l'un des cas humains ne montre pas de négation explicite évidente. Ces comptes ne mesurent pas la fréquence des causes dans le corpus entier.

Deux cas réels sont indexés pour une lecture locale : **TSAC-01, POS attendu / NEG prédit**, et **MADAR-01, TN attendu / MSA prédit**. Leurs textes ne sont pas transférés dans Canva ; nous ouvrons les entrées et les avis dans le classeur privé `tuniscope_analyse_erreurs_baseline_assistee_2026-10-17.xlsx`. Le CSV détaillé demandé par le guide n'a pas été exporté par cette modification de présentation : le classeur et le registre existants ne doivent pas être présentés comme un nouveau CSV livré.

## Partitions versionnées : date et empreintes

La première version de manifestes est présente dans le commit `384f8a5aba14650b75ebd7a47a558433b56553ce`, daté du 16 octobre 2026. Les manifestes v2 actuels sont versionnés dans `dd4d218d5574307fe57cd156ea5e587d220ab518`, daté du 19 octobre 2026. Cette date de versionnement ne doit pas être présentée comme le gel final de toutes les expériences.

| Fichier v2 | SHA-256 vérifié le 26 octobre |
| --- | --- |
| `tsac.csv` | `ac8a453de6115bea6b51255fe7badc1eab72e2649e0056a1ad918aa963730fa7` |
| `madar.csv` | `3d81826b70437aa37a286d2d8fded745b01295165395b26cf868f91f90204474` |

HEAD observé : `2d6900a9fc5d7c2808ea7fb76b4d3b07a3942668`, avec modifications locales. Aucun tag `bronze` n'existe à la vérification. La demande d'éditer la présentation ne crée pas de tag, commit ou push et ne prouve pas une relance depuis un tag.

## Attendus restant ouverts

- Démo d'interface/API sur une entrée improvisée, gestion hors périmètre et mesure de latence de bout en bout.
- Recherche BM25 documentaire et sortie sourcée intégrées.
- Sélection et gel du contraste, puis évaluation autorisée du test officiel sans réglage sur ses résultats.
- Reproduction bibliographique : résultat choisi, protocole comparable, valeur publiée et valeur obtenue. Aucun résultat n'est déclaré reproduit ici.
- Relecture des ajouts IA et finalisation des cartes ; un second annotateur humain indépendant serait nécessaire pour un kappa interannotateurs.
- CSV détaillé des erreurs, tag `bronze` sur un état présenté vérifié, PDF des slides et vidéo de secours de deux minutes.

Ces limites sont explicites, sans comparaison de retard calendaire. Le travail solo ne fournit pas de dispense implicite aux exigences scientifiques du guide. Aucun nouvel entraînement, poids neuronal, seuil ajusté sur test ou application n'est ajouté pour embellir l'état annoncé.
