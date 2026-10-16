# TuniScope — Audit de l'union des exclusions candidates TSAC

**Livrables concernés :** L02 (contrôles, provenance et traçabilité des données), contribution partielle ; L03 (future préparation reproductible), protocole et script d'audit seulement.

**Date :** 16 octobre 2026. **État :** calcul vérifié en lecture seule ; aucun jeu préparé, manifeste de partition ou fichier source modifié.

## Objectif

Une même ligne d'entraînement TSAC peut être un doublon exact du test, appartenir à une paire proche que nous avons jugée équivalente et se répéter dans le train. Additionner les effectifs de ces catégories compterait cette ligne plusieurs fois. Nous calculons donc l'**union des références de lignes** à exclure : une référence `(fichier, numéro de ligne)` apparaît une seule fois dans la liste finale, avec tous ses motifs conservés.

Le script `python3 scripts/audit_tsac_exclusion_union.py` lit les quatre fichiers TSAC originaux, les index privés figés et les décisions de revue. Il vérifie les empreintes des index et du classeur finalisé, ainsi que la cohérence des références. Il ne consulte le test officiel que pour repérer les recouvrements ; il ne l'altère pas et n'entraîne aucun modèle. La sortie publique ne contient que des agrégats. Le relevé ligne par ligne, sans texte de corpus, se trouve dans `data/interim/tsac_exclusion_candidates_2026-10-16_v2.json`, exclu de Git. Une première projection privée, remplacée par cette version après un contrôle indépendant, ne traitait que les doublons internes strictement identiques.

## Ordre des opérations simulées

1. Écarter du train les lignes vides, les textes identiques au test au caractère près, puis les formes identiques après Unicode NFKC et regroupement des espaces. Cette clé est une clé **d'audit**, non la normalisation définitive du modèle.
2. Ajouter les occurrences train associées aux 124 paires proches jugées équivalentes et aux 152 paires courtes jugées de même sens. Les 7 paires courtes jugées de sens différent ne déclenchent aucune exclusion. Les cas non linguistiques revus sont traités côté train seulement ; le cas court privé no 158 ne justifie pas de retirer le côté test contenant un emoji.
3. Sur les lignes train restantes, regrouper les textes strictement identiques et conserver un représentant déterministe par groupe. Pour les deux groupes restants où POS et NEG coexistent, nous gardons une occurrence portant l'étiquette choisie lors de notre revue, **sans modifier l'étiquette source**. Les autres groupes gardent la première référence ordonnée par `(fichier, numéro de ligne)`.
4. Rechercher encore les textes distincts au caractère près mais identiques après la clé NFKC/espaces. Les sept groupes restants portent chacun la même étiquette ; nous gardons un représentant déterministe par groupe. Ce contrôle supplémentaire a été ajouté après qu'une vérification indépendante a révélé sept doublons que la première projection laissait dans le train.

## Résultats vérifiés

| Étape | Lignes train écartées, distinctes | Lignes train restantes |
| --- | ---: | ---: |
| Fichiers officiels avant traitement, y compris 4 lignes vides | 0 | 13 669 |
| Contrôles train–test et invalidité | 1 261 | 12 408 |
| Déduplication exacte interne au train restant | 574 supplémentaires | 11 834 |
| Déduplication interne après NFKC/espaces | 7 supplémentaires | 11 827 |
| **Total provisoire** | **1 842** | **11 827** |

Les quatre fichiers sources contiennent aussi **3 400 lignes test**, inchangées. Les 1 842 références candidates sont uniques et appartiennent toutes au train. Parmi elles, **395 lignes ont plusieurs motifs**. Le nombre de lignes train retenues après la simulation de toutes les règles est `13 669 − 1 842 = 11 827` ; ce n'est pas encore l'effectif d'un manifeste produit.

| Motif | Lignes concernées avant union |
| --- | ---: |
| Identité brute avec le test | 963 |
| Identité après NFKC/espaces, sans identité brute | 20 |
| Paire proche revue parmi les 124 | 193 |
| Paire courte revue « Même sens » | 476 |
| Texte non linguistique revu côté train | 4 |
| Ligne train vide | 4 |
| Doublon exact interne après les contrôles précédents | 574 |
| Doublon interne après NFKC/espaces, après déduplication brute | 7 |

La somme de cette colonne **n'est pas** le total des suppressions : les motifs se recouvrent. Après la première étape, 471 groupes de doublons exacts subsistent dans le train, dont deux groupes à étiquettes contradictoires déjà revus. Une vérification indépendante du relevé privé et des fichiers sources a confirmé **1 842 références uniques**, aucun fichier test marqué, aucun texte de corpus dans le relevé, et zéro doublon interne ou recouvrement train–test après la clé NFKC/espaces dans la simulation.

## Limites et suite

Les jugements de proximité sont ceux d'une seule personne. Les 77 décisions courtes déduites de notre consigne « cellule vide = même sens » sont distinguées des décisions explicitement saisies. La méthode de recherche de candidats peut manquer des reformulations ; elle ne prouve pas l'absence de toute fuite sémantique. Les quasi-doublons **à l'intérieur du train** n'ont pas encore été examinés pour former les groupes de la future validation. Le test officiel comporte ses propres doublons et au moins une anomalie non linguistique déjà signalée ; le score principal devra les conserver et les commenter, ou présenter une analyse diagnostique séparée.

La prochaine étape est de contrôler les familles de variantes à l'intérieur du train, puis de figer la règle de partition et la graine. Ensuite seulement nous pourrons écrire les manifestes train/validation/test, appliquer les exclusions à une copie de travail, compter les effectifs réels et remplir les cartes de données. **L02 et L03 ne sont donc pas encore achevés.**
