# TuniScope — Dépistage des quasi-doublons TSAC

**Livrables concernés :** L02 (contrôle des données) et L03 (script reproductible), contributions partielles.

**Date :** 4 octobre 2026. **État :** dépistage automatique, sans confirmation humaine ni modification des corpus.

## Objectif et périmètre

Après le [contrôle des doublons exacts](04_data_audit_groups_and_duplicates.md), cette passe cherche des commentaires TSAC **similaires mais non identiques** entre le train officiel et le test officiel. Une faible différence de caractères peut signaler une copie modifiée, mais peut aussi inverser le sens, par exemple si une négation change. Chaque résultat est donc un *candidat à revoir*, jamais une suppression automatique. L'inspection du test sert uniquement à vérifier la contamination des données ; aucun modèle, prompt ou seuil de performance n'est réglé avec ses labels.

## Méthode reproductible

`python3 scripts/screen_near_duplicates.py` lit les fichiers originaux sans les modifier et n'affiche que des agrégats. Sa forme de **recherche temporaire** applique Unicode NFKC et regroupe les espaces. Elle ne remplace pas le texte brut, ne définit pas la future normalisation d'entraînement et n'est jamais écrite dans un jeu de données public.

- Les formes devenues exactement identiques après cette opération sont comptées séparément.
- Pour les textes de 8 à 19 caractères, le script recherche une clé commune après au plus une suppression de caractère par chaîne, puis exige un ratio `SequenceMatcher` d'au moins 0,90.
- À partir de 20 caractères, il récupère des candidats par trigrammes de caractères informatifs (présents dans au plus 800 formes train), exige un rapport de longueurs d'au moins 0,80, une similarité de Jaccard des trigrammes d'au moins 0,65, puis un ratio `SequenceMatcher` d'au moins 0,92.
- Les formes de moins de 8 caractères sont exclues **uniquement de la recherche de quasi-doublons** ; elles restent incluses dans les contrôles d'égalité exacte.

Ces seuils sont des choix prudents pour constituer une liste de revue, non une mesure de rappel démontrée. La recherche est limitée aux paires train–test ; elle ne couvre pas encore les quasi-doublons à l'intérieur du train ou du test. Une paire très proche peut être manquée par les filtres de candidats.

## Résultats

| Mesure | Effectif |
| --- | ---: |
| Formes train distinctes après NFKC et espaces regroupés | 12 355 |
| Formes test distinctes après la même opération | 3 231 |
| Formes identiques entre train et test après cette opération | 324, concernant 467 lignes test |
| Référence sans cette opération : textes strictement identiques | 316 groupes, concernant 457 lignes test |
| Paires **candidates** similaires mais non identiques | 124 |
| Formes test impliquées dans ces candidats | 94, concernant 98 lignes test |
| Formes train impliquées dans ces candidats | 114 |
| Paires candidates avec au moins une combinaison possible de labels opposés | 2 |
| Formes test de moins de 8 caractères non dépistées comme quasi-doublons | 139 |

Les 124 paires ne sont pas 124 erreurs prouvées. Les 98 lignes test candidates ne sont pas ajoutées aux 467 lignes d'égalité normalisée pour calculer un nouveau « taux de contamination » avant revue. Aucun exemple n'a été retiré, relabellisé ou déplacé.

## Index de revue privés

Deux index locaux, exclus de Git, ne contiennent **que des noms de fichiers, numéros de lignes, catégories et scores**, sans commentaire ni phrase de corpus :

- `data/interim/exact_review_2026-10-04.tsv` : 333 références de cas, soit 4 croisements textuels MADAR, 316 recouvrements exacts train–test TSAC et 13 conflits de labels TSAC. Une catégorie peut recouvrir une autre.
- `data/interim/tsac_near_review_2026-10-04.tsv` : 124 paires candidates TSAC, classées par similarité décroissante.

Les index sont des outils de navigation privée, **pas** des manifestes de données ni des jugements d'annotation. Ils peuvent être recréés avec les options `--review-index` des deux scripts ; chacun refuse d'écraser un index existant et n'accepte qu'un chemin sous `data/interim/`.

## Suite et limites

La revue humaine doit commencer par les 13 groupes à labels contradictoires et les quatre croisements MADAR, puis examiner en priorité les deux paires quasi identiques dont les labels peuvent s'opposer. Une règle déterministe, à décider et à documenter, pourra ensuite traiter les copies exactes du train sans toucher au test officiel. Les quasi-doublons ne seront exclus qu'après vérification du sens et de leur relation réelle. Les cartes de données, manifestes et effectifs exploitables restent à produire ; L02 et L03 ne sont pas terminés.
