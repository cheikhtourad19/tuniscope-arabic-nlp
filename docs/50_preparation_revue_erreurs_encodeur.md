# Préparation de la revue qualitative des erreurs de l’encodeur

**6 novembre 2026 — contributions partielles à L05, L07 et L10.** La sélection et le formulaire sont préparés et vérifiés. Les avis humains, catégories linguistiques et hypothèses restent à recueillir ; cette préparation ne constitue pas une analyse d’erreurs terminée.

## Objectif

Nous cherchons à comprendre les désaccords entre les prédictions de CAMeLBERT adapté et les labels de validation, en présentant aussi la prédiction TF-IDF caractères. La [comparaison quantitative sur les trois graines](../reports/encoder_vs_tfidf_2026-11-05/README.md) reste la référence exhaustive pour les effectifs. Aucun score global, label, texte source, partition ou modèle applicatif n’est modifié par cette revue.

## Sources et référence qualitative

Nous réutilisons le run encodeur FULL `2026-11-05_cls_paired_mps_full` et la référence classique FULL `2026-10-17_grouped_v2_full`, représentation `raw-v0`, manifestes v2. Les prédictions sauvegardées sont contrôlées par leurs empreintes. Les identifiants, groupes et labels sont alignés avec les sources de validation ; les métriques des trois encodeurs adaptés sont recalculées. La preuve antérieure de rechargement encodeur est reliée aux résultats par SHA-256. Aucune nouvelle inférence ni aucun entraînement n’est nécessaire.

La **graine 42** est choisie comme référence qualitative fixe, conformément à l’annonce précédant la sélection des cas. Ce choix n’est pas celui de la meilleure graine : les évaluations quantitatives conservent 13, 42 et 2026. Les prédictions des deux autres graines apparaissent dans le contexte de chaque cas ; elles ne constituent pas un vote humain ni une mesure de confiance calibrée.

## Sélection

Nous sélectionnons 30 erreurs de validation par tâche parmi les 122 erreurs TSAC et les 38 erreurs MADAR de cette graine. Les critères sont purement techniques : comparaison avec TF-IDF, confusion entre labels et système d’écriture Unicode. L’ordre interne utilise SHA-256 de `42|id`. Une rotation des strates, des plus petites aux plus grandes, conserve des groupes et des textes normalisés NFC/espaces distincts. Cette normalisation sert uniquement à éviter des répétitions dans la revue ; le texte affiché reste celui de la source.

| Corpus | Erreurs candidates | Cas retenus | Erreurs communes | Erreurs introduites | Écriture des cas retenus |
| --- | ---: | ---: | ---: | ---: | --- |
| TSAC | 122 | 30 | 15 | 15 | 15 arabe, 15 latin |
| MADAR | 38 | 30 | 11 | 19 | 30 arabe |

Chaque tâche comporte 30 groupes distincts. « Commune » signifie que les deux modèles se trompent selon le label source, éventuellement avec des labels prédits différents. « Introduite » signifie que TF-IDF retrouve le label source et que l’encodeur se trompe. Ces termes ne décrivent pas la cause linguistique de l’erreur.

Cette sélection qualitative ne permet pas d’estimer la fréquence des difficultés dans toute la population. Les textes corrigés par CAMeLBERT ne font pas partie de ces 60 erreurs ; leurs effectifs restent dans le rapport quantitatif. Aucune ligne de test officiel n’est sélectionnée.

## Formulaire et intervention humaine

Le classeur privé `TuniScope_revue_erreurs_encodeur.xlsx` contient les onglets TSAC et MADAR. Il est conservé dans `outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4/encoder_error_review_2026-11-06/`, exclu de Git avec le payload contenant les textes.

Les champs à renseigner sont séparés des sources :

- **F — Avis sur le label :** label plausible, discutable, incertain ou texte non interprétable.
- **G — Difficulté supposée :** une catégorie choisie dans la liste, dont « Incertain ».
- **H — Indice / explication courte :** élément du texte ou contexte manquant qui soutient l’hypothèse. Une incertitude peut être explicitement écrite.

Les champs humains sont entièrement vides à la livraison. Les labels du corpus et les prédictions sont visibles : il s’agit d’une revue de diagnostic avec un seul annotateur humain, pas d’une annotation aveugle ni d’un accord interannotateurs. L’avis sur un label ne le remplace pas automatiquement. Toute révision éventuelle relèverait d’un protocole séparé et versionné ; nous préservons les comparaisons historiques.

Le compteur n’incrémente un cas que lorsque les trois champs F–H sont renseignés. Une case vide reste en attente. Les colonnes I–K donnent la comparaison, la référence à la ligne source et les autres graines. Les textes ressemblant à une formule sont affichés comme texte ; d’éventuelles séquences littérales `_xHHHH_` sont représentées par `⟦xHHHH⟧`, avec la chaîne brute exacte conservée dans le payload privé.

## Scripts et contrôles

- `scripts/export_encoder_error_review.py` : lecture des prédictions vérifiées, sélection et payload privé immuable.
- `scripts/build_encoder_error_review_workbook.mjs` : création du classeur, listes, mise en forme, compteurs, relecture du XLSX exporté.
- `tests/test_export_encoder_error_review.py` : trois tests ciblés, couvrant déterminisme, diversité, déduplication et rejet des partitions de test/prédictions correctes.

Le contrôle réel confirme 60 cas et 180 cellules humaines vides, les données sources identiques après export/réimport, les listes déroulantes et les volets figés. Le compteur est testé avec une saisie partielle puis complète, avant effacement de toutes les valeurs de test. Les deux onglets et leurs textes les plus longs sont inspectés visuellement. Un contrôle OOXML indépendant vérifie les cellules vides, l’absence de formules dans les lignes de cas, les validations et les volets. Le scan ne trouve aucune erreur de formule.

Les relevés privés sont `workbook_verification.json` et `independent_ooxml_checks.json`. Le SHA-256 initial du classeur est `683e0b3ff5b58be7634f491cc9a8a8f0e4d31a2a2d775c9d856de9d4971a255e` ; il changera normalement lors de la saisie humaine. Les scripts refusent d’écraser les fichiers existants.

## Travail restant

Nous devons remplir les deux onglets, importer les réponses avec contrôle des sources, puis produire les agrégats et interprétations en distinguant les constats des hypothèses. L’analyse des erreurs des composants finaux, les autres éléments d’évaluation, les runs finaux et le test restent ouverts. Cette revue de l’encodeur ne remplace pas celles des autres composants appris ni les exigences globales de L07/L10.
