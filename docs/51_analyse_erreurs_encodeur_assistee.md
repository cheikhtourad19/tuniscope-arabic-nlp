# Analyse assistée des erreurs de CAMeLBERT

**6 novembre 2026 — contributions partielles à L07 et L10.** À notre demande explicite, nous complétons les 60 cas du formulaire préparé en [doc50](50_preparation_revue_erreurs_encodeur.md) avec une assistance IA déclarée. La revue assistée est documentée ; elle ne constitue pas une annotation humaine indépendante et ne clôt pas les livrables globaux.

## Méthode et provenance

La sélection reste celle de la validation FULL v2, texte brut, graine 42 : 30 erreurs TSAC et 30 erreurs MADAR, avec groupes et textes distincts. Les prédictions TF-IDF et des graines 13/2026 sont affichées comme contexte. Aucune nouvelle sélection, inférence ou mise à jour des modèles n’est effectuée.

Le classeur source ne contenait aucune réponse humaine au moment du contrôle. Nous conservons son fichier et une copie instantanée identiques, puis produisons une copie séparée `TuniScope_analyse_erreurs_encodeur_assistee.xlsx`. Les 60 avis sur les labels, 60 catégories et 60 hypothèses sont donc tous d’origine IA. Un registre privé consigne l’origine de chaque champ, les réserves, les réponses sources et les empreintes. L’outil préserve les réponses existantes lorsqu’elles sont présentes ; cette propriété est aussi contrôlée avec des exemples fictifs.

La méthode repose sur une lecture du texte et de ses indices linguistiques. Elle ne mesure ni les mécanismes internes du réseau ni une cause démontrée de chaque erreur. Les noms propres, monnaies et sujets évoqués ne sont pas traités comme des preuves suffisantes de dialecte. Une graphie partagée entre variétés ne rend pas automatiquement faux un label de provenance MADAR.

## Résultats descriptifs

Les tableaux et catégories détaillées sont [générés depuis le registre](../reports/encoder_errors_assisted_2026-11-06/README.md).

| Corpus | Cas documentés | Label plausible selon l’IA | Label discutable selon l’IA | Avis incertain | Avis humains |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 30 | 24 | 3 | 3 | 0 |
| MADAR | 30 | 18 | 0 | 12 | 0 |

Pour TSAC, les hypothèses proposées concernent surtout les graphies Arabizi, les avis mixtes, les expressions locales et la portée de la négation. Ces catégories organisent la lecture de cet échantillon ; elles ne constituent pas des fréquences de causes dans tout le corpus. Le commentaire entièrement français est signalé comme tel, sans être automatiquement décrit comme du mélange de langues.

Trois cas TSAC sont placés en réserve prioritaire parce qu’une lecture littérale semble contredire le label source. L’ironie, le contexte absent ou l’ambiguïté de transcription peuvent expliquer le désaccord : nous ne concluons pas à trois erreurs d’annotation établies. Aucun label proposé ne remplace celui utilisé pour l’évaluation.

Pour MADAR, plusieurs formulations sont brèves et partagées avec le standard ou d’autres variétés. Douze avis restent incertains. Cette observation rappelle la différence entre la provenance d’une traduction et la présence d’un indice dialectal exclusif dans chaque phrase. Les autres avis jugent le label plausible en relevant des indices compatibles, sans prétendre qu’ils sont toujours exclusifs.

## Vérifications

Le script `scripts/complete_encoder_error_review.mjs` complète uniquement les champs autorisés dans une copie, ajoute les mentions visibles d’assistance et adapte les hauteurs des lignes aux explications. Les catégories et avis restent dans les listes prévues. Les hypothèses ajoutées portent le préfixe `IA —`.

Après export, le classeur est réimporté et contrôlé : 60 cas documentés, 180 champs ajoutés, textes, labels, prédictions, références et autres graines préservés. Un contrôle OOXML indépendant compare les 480 cellules sources, les largeurs de colonnes, les listes de choix, les volets figés et les formules des compteurs. Les compteurs indiquent 30 cas documentés par onglet ; ils ne représentent pas une validation humaine. Le scan ne détecte aucune erreur de formule. Les deux onglets et les explications les plus longues sont inspectés visuellement.

SHA-256 du classeur assisté : `ef4ec8f770db759c5d281c76d8f8959ef6f1a2ead0be830d9e2452706940bdc0`. L’original conserve `683e0b3ff5b58be7634f491cc9a8a8f0e4d31a2a2d775c9d856de9d4971a255e`.

Les classeurs, instantané, propositions par cas, registre et preuves XML restent sous le dossier privé `outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4/encoder_error_review_2026-11-06/`, exclu de Git. Le dépôt contient les scripts, cette méthodologie et les agrégats, sans reproduire les textes protégés.

## Interprétation et portée

Cette analyse fournit un premier diagnostic qualitatif de l’encodeur initial. Les incertitudes restent visibles ; aucune obligation de réannotation exhaustive n’est ajoutée à cette étape déléguée. Une relecture humaine ultérieure, en particulier des trois réserves prioritaires, pourrait renforcer les conclusions et devra être distinguée des propositions actuelles.

Les métriques restent calculées avec les labels et partitions figés. Il n’y a ni suppression d’exemples de validation, ni transfert vers le train, ni modification des sources, ni évaluation du test officiel. Les hypothèses peuvent orienter des contrôles de robustesse et des comparaisons ultérieures, avec un protocole séparé.

Les autres composants appris, la configuration finale, les expériences finales et les exigences d’évaluation restent à traiter. L07 et L10 restent partiels ; aucune mesure d’accord interannotateurs ni validation humaine des ajouts IA n’est revendiquée.
