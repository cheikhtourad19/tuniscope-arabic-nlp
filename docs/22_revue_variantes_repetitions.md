# TuniScope — Préparation de la revue des variantes à lettres répétées

**Livrables concernés :** L02 (groupes, décisions humaines et contrôle des fuites), contribution partielle ; L04/L07 pour la qualification des premiers scores. **Date :** 17 octobre 2026. **État :** 12 décisions humaines explicites enregistrées et révision des manifestes vérifiée.

## Périmètre

L'[audit des répétitions](18_audit_tatweel_repetitions.md) a identifié neuf familles TSAC qui deviennent exactement identiques après la règle candidate de réduction des suites d'au moins trois lettres identiques à deux. Nous avons repris **toutes** ces familles à partir des manifestes v1 et des textes source, indépendamment des scores et des prédictions des [premières baselines](21_premieres_baselines.md). Aucune ligne du test officiel n'est chargée.

Les neuf familles concernent **11 lignes train et 10 lignes validation**, soit 21 textes distincts par référence. Certaines familles comprennent plusieurs variantes : leur comparaison exhaustive entre les deux partitions produit **12 paires**, et non neuf. Le classeur comporte une ligne par paire et une colonne de famille pour retrouver ces relations.

## Support et vérification

`scripts/export_repetition_review.py` génère les références et les textes dans un fichier privé sous `outputs/`, ainsi qu'un index sans texte dans `data/interim/`. `scripts/build_repetition_review_workbook.mjs` construit le classeur privé `tuniscope_repetition_review_2026-10-17.xlsx`, sous le dossier de sortie de cette conversation. Chaque ligne présente les deux textes bruts, les références source, une décision vide et une note vide. Les prédictions du modèle ne sont pas fournies. Les noms `train_*.txt` dans les références désignent les fichiers d'origine : la validation en a été extraite par le manifeste.

Les choix proposés sont « Même sens », « Sens différent », « Incertain » et « Non linguistique ». **Toute cellule vide reste en attente** ; la consigne donnée lors d'une ancienne revue ne s'applique pas automatiquement à ce nouveau lot. Nous demandons une décision distincte pour chacune des 12 paires, même si plusieurs partagent une famille.

Nous avons vérifié le nombre de familles/paires, l'unicité des références de paires, la conformité des textes aux sources, l'absence de décisions et notes préremplies, la liste de validation, le compteur de décisions et la réouverture du fichier exporté. Le rendu visuel a été inspecté, y compris les commentaires en arabe. Le classeur, son aperçu et les relevés contenant des références ou textes privés restent ignorés par Git. Ce support ne modifie ni les sources, ni les labels, ni les manifestes, ni les checkpoints.

## Résultat de la revue

Nous avons renseigné les 12 comparaisons avec « Même sens ». Aucune cellule n'est vide ou incertaine et aucune décision n'a été déduite automatiquement. `scripts/import_repetition_review.mjs` a vérifié les numéros de cas, familles, textes et références contre l'entrée originale, rejeté toute formule dans les cellules de revue et enregistré les décisions dans un registre privé sous `data/interim/`. Les empreintes du classeur, de l'entrée, de l'index et du manifeste assurent leur traçabilité. Le classeur annoté n'a pas été réécrit.

Les douze paires soutiennent donc le maintien de leurs neuf familles dans une seule partition. La [révision v2 des manifestes](23_manifestes_v2_et_baselines.md) applique cette contrainte, contrôle les autres familles candidates de répétitions et relance les baselines sans changer leurs hyperparamètres. Les sources, étiquettes et lignes du test officiel restent inchangées.

## Portée de la décision

La collision après transformation n'est pas une preuve suffisante d'équivalence de sens : notre jugement porte sur les douze paires relues seulement. Les autres familles rapprochées par le filtre constituent des contraintes conservatrices de partitionnement, pas de nouvelles annotations sémantiques. Aucun commentaire n'a été supprimé ou réétiqueté par cette revue.

Il s'agit d'une revue individuelle, pas d'un accord interannotateurs indépendant. **L02 reste incomplet et la recherche heuristique ne garantit pas l'absence de toute parenté résiduelle.** Les premiers scores v1 sont conservés comme historique et ne remplacent pas les nouvelles mesures sur v2.
