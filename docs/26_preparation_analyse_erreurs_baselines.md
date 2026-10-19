# TuniScope — Préparation de l'analyse humaine des erreurs classiques

**Date :** 17 octobre 2026. **Livrables :** L07 (analyse d'erreurs) et L12 (traçabilité), contributions partielles. **État :** 60 avis humains enregistrés ; une copie complétée avec assistance IA documente les 60 cas. Les ajouts IA restent à relire ; voir la [synthèse du lot](27_analyse_erreurs_classiques_assistee.md).

## Référence et périmètre

Nous examinons la baseline caractères 3–5 / régression logistique sur textes bruts, retenue comme référence principale après la [comparaison des représentations](25_baseline_mots_caracteres_combines.md). Les exemples viennent exclusivement de la validation des manifestes v2. Nous conservons les prédictions du run déjà exécuté : aucun nouveau modèle n'est entraîné et aucun texte du test officiel n'est chargé.

Une « erreur » désigne ici un désaccord avec le label d'origine, pas la preuve que le modèle a nécessairement tort du point de vue linguistique. La revue peut signaler un label discutable, un texte ambigu ou une compétence linguistique insuffisante. Elle ne réétiquette ni les fichiers source ni les manifestes et ne recalcule pas les scores publiés sur des labels ajustés après coup.

Le cahier demande au moins 30 erreurs distinctes par composant appris. Ce premier lot porte sur deux classifieurs classiques : sentiment et variété linguistique. Il ne remplace pas les futures analyses des modèles adaptés ni l'analyse finale sur le protocole gelé. Il ne clôture pas non plus le travail séparé sur 50 phrases pour la syntaxe, la négation et la pragmatique.

## Sélection reproductible

`scripts/export_baseline_error_review.py` vérifie le run, les checkpoints et les métriques, aligne les prédictions sur les sources sélectionnées par manifeste, puis extrait les désaccords. Le script choisit 30 cas par tâche avec au plus une référence par groupe et un texte distinct après NFC/espaces.

| Corpus | Désaccords disponibles | Groupes concernés | Cas retenus | Groupes retenus |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 126 | 124 | 30 | 30 |
| MADAR | 39 | 34 | 30 | 30 |

Nous répartissons la sélection entre les combinaisons `(label source, prédiction, écriture)`. Les strates les moins nombreuses sont visitées en premier, puis une sélection cyclique prend un cas par strate encore disponible. L'ordre interne repose sur SHA-256 de `42|id`, sans dépendance à l'ordre des lignes. Nous écartons les références déjà couvertes par groupe ou par texte normalisé. Le lot couvre les six strates observées pour TSAC et les cinq pour MADAR, mais n'est **pas un échantillon proportionnel** : les fréquences de causes relevées dans ces 30 cas ne permettront pas d'estimer leur prévalence dans toutes les erreurs.

Le sous-lot TSAC comporte 11 désaccords NEG→POS et 19 POS→NEG ; les écritures sont 11 AR, 11 LATIN, six MIXED et deux UNKNOWN. Ces valeurs décrivent le script Unicode, pas une langue. MADAR comporte huit MSA→TN, deux TN→EG, un TN→MSA, 17 EG→TN et deux EG→MSA. Les groupes de variantes et phrases parallèles ne garantissent pas l'indépendance d'auteurs ou conversations absents des sources.

## Support de revue et consignes

Le classeur privé `tuniscope_revue_erreurs_baseline_2026-10-17.xlsx` se trouve sous le dossier `outputs/` de cette conversation, ignoré par Git. Deux feuilles, `TSAC` et `MADAR`, comportent chacune 30 cas. Chaque ligne présente le texte, le label d'origine, la prédiction et la référence source. Trois colonnes jaunes sont à renseigner :

1. **Avis sur le label source** : « Label plausible », « Label discutable », « Incertain » ou « Non interprétable ».
2. **Difficulté principale** : catégorie proposée à titre d'hypothèse, ou « Incertain ». Les listes sont adaptées aux deux tâches.
3. **Indice / explication courte** : un indice observé et une hypothèse, ou la raison explicite de notre incertitude. Une cause n'est pas établie par le seul choix d'une catégorie.

Les cellules sont initialement vides. Une cellule laissée vide signifie **à revoir**, sans reprendre la convention d'un ancien lot de doublons. Le compteur exige les trois champs renseignés ; une réponse incertaine honnêtement expliquée peut être enregistrée, mais ne constitue pas une cause linguistique résolue. Nous ne supposons pas une compétence suffisante pour tous les cas égyptiens. Les labels et prédictions sont visibles : cette procédure est un diagnostic guidé, pas une annotation indépendante ou aveugle ni un accord interannotateurs.

Les fichiers d'origine `train_pos.txt` et `train_neg.txt` peuvent figurer dans les références TSAC : la validation a été extraite de ces fichiers par le manifeste. Cela ne signifie pas que les textes sélectionnés ont été utilisés pour ajuster le modèle. Leurs identifiants de validation ont été vérifiés.

## Fidélité du texte et vérification du classeur

Les chaînes brutes exactes sont conservées dans l'entrée privée JSON. La vérification visuelle a repéré cinq commentaires TSAC contenant littéralement la séquence `_x000D_` dans le corpus. Comme Excel peut interpréter les séquences `_xHHHH_`, la colonne **Texte (lecture)** affiche réversiblement ces motifs sous la forme `⟦xHHHH⟧`, par exemple `⟦x000D⟧`. Ce marquage explicite est réservé à l'affichage. Il ne retire ni ne normalise les motifs dans les sources, les entrées des modèles ou les manifestes ; son effet sur les prédictions n'est pas mesuré ici. Les autres textes sont préservés, avec protection des chaînes pouvant commencer une formule.

`scripts/build_error_review_workbook.mjs` génère le support, les listes de choix et le compteur. Un test vérifie qu'une ligne partiellement renseignée ne compte pas comme complète ; les valeurs techniques sont effacées avant livraison. Le fichier exporté est rouvert et ses textes de lecture, labels, prédictions, références, champs vides et absence de formules dans les données sont contrôlés. Les en-têtes et colonnes d'identification sont figés pour la navigation. Les deux feuilles et les textes les plus longs ont été rendus pour inspection visuelle. Les contrôles XML complémentaires portent sur les valeurs, les validations, les volets figés et les formules de compteur.

Le script d'extraction, les empreintes des manifestes/prédictions et l'index privé sans texte permettent de retrouver chaque cas. Le classeur, les textes JSON, index et aperçus restent hors de Git. La suite locale compte **44 tests réussis**, dont quatre nouveaux tests de sélection sur données fictives. Ces tests ne remplacent pas les jugements humains encore manquants ni un essai dans l'application Excel native.

## Reproduction et suite

Avec les corpus autorisés et les sorties privées du run caractères disponibles, l'export refuse d'écraser un lot existant. Le constructeur refuse d'écraser un classeur modifié ou annoté ; son option de régénération technique ne fonctionne que si l'empreinte correspond au support vierge qu'il a lui-même enregistré. La génération du XLSX utilise le runtime documentaire décrit dans l'environnement de travail, pas une nouvelle dépendance de modélisation.

Après revue humaine, nous vérifierons les cellules source et enregistrerons les décisions séparément, puis produirons les catégories agrégées et discuterons leurs limites. Une revue de validation peut guider les hypothèses, mais les vocabulaires et lexiques induits restent à construire sur train seulement. Les textes de ce lot ne sont pas automatiquement ajoutés au train ni à un lexique pour améliorer les scores sur ces mêmes exemples.

## Instantané après la saisie humaine, avant complément IA

Nous avons enregistré un avis explicite sur chacun des 30 labels TSAC et des 30 labels MADAR. Le [rapport d'avancement généré](../reports/error_review_progress_2026-10-17_215d8f0fdce9/README.md) distingue la revue des labels de l'analyse des causes. TSAC compte 23 labels plausibles, quatre discutables, deux avis incertains et un cas jugé non interprétable. MADAR compte 24 labels plausibles et six discutables. « Discutable » reste notre avis individuel, pas une correction établie du corpus.

Nous avons choisi cinq catégories de difficulté TSAC et six MADAR. Les 60 explications courtes sont encore vides : aucun cas ne possède donc encore les trois champs E/F/G complets. Les cinq catégories TSAC sont réparties entre émoji/symbole, manque de contexte, avis mixte/ambigu (deux cas) et négation ; les six MADAR signalent des variétés proches ou textes communs. Ces indications restent des hypothèses sans explication rédigée et sans estimation de prévalence.

`scripts/import_error_review.mjs` lit le classeur sans le réécrire, vérifie les choix autorisés et la correspondance des cas, labels, prédictions et références avec l'entrée figée, puis sauvegarde un instantané privé nommé par l'empreinte du classeur. Les cellules vides restent explicitement vides. Un seul écart de texte de lecture a été trouvé : suppression d'un espace final, sans autre différence de contenu. Cet écart est tracé, pas répercuté dans les sources. Aucune étiquette, prédiction, donnée brute ou partition n'est modifiée.

L'instantané contient les réponses exactes et les empreintes du classeur, de l'entrée, de l'index et du run source. Les réponses ont aussi été comparées à la représentation XML du XLSX. Huit contrôles fictifs de l'importateur vérifient notamment les réponses partielles, les choix invalides, les modifications de contenu et l'absence d'inférence des blancs. Les 44 tests Python existants passent toujours. Le rapport public ne contient que des comptes et empreintes, sans texte de corpus, référence individuelle ni note libre.

À cet instantané, F et G restaient à compléter. Nous avons ensuite délégué explicitement ce complément à l'IA, sans lui déléguer une correction des labels ou une nouvelle décision humaine. La [synthèse suivante](27_analyse_erreurs_classiques_assistee.md) distingue les 11 catégories humaines conservées des 49 catégories et 60 hypothèses ajoutées par IA. L'original et son instantané partiel restent conservés. **L07 reste incomplet** pour les notebooks, les analyses des futurs modèles et l'évaluation finale. L12 reste également partiel, notamment pour la relance indépendante.
