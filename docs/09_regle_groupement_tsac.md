# TuniScope — Règle de regroupement des candidats TSAC

**Livrables concernés :** L02 (protocole de séparation et contrôle des fuites), contribution partielle ; L03 (audit reproductible), contribution partielle.

**Version :** `tsac-candidate-components-v1`, 16 octobre 2026. **État :** règle fixée et vérifiée en lecture seule ; aucun manifeste ni jeu de données préparé.

## Décision

Nous utilisons les 374 liens candidats repérés **uniquement pour définir des groupes indivisibles lors du futur partage train/validation**. Les 203 liens du filtre principal et les 171 liens entre formes très courtes sont tous inclus. Deux lignes directement liées vont dans le même groupe ; si A est liée à B et B à C, les trois vont dans le même groupe. C'est une composante connexe du graphe des liens.

Cette règle est volontairement prudente contre les fuites : deux variantes proches ne pourront pas se retrouver de part et d'autre du partage. Elle peut aussi réunir des commentaires de sens différent, surtout parmi les formes courtes. **Le groupe n'est pas une classe de sens, une correction d'étiquette ou une instruction de suppression.** Les étiquettes POS et NEG d'origine restent inchangées. Les sept groupes mêlant les deux étiquettes restent ensemble ; ils ne sont pas automatiquement considérés comme erronés.

La règle s'applique seulement aux **11 827 lignes train projetées après les exclusions candidates v2**. Elle n'utilise pas les lignes du test officiel et ne modifie ni les archives ni les fichiers originaux. Chaque ligne isolée est un groupe à elle seule. Pour le futur manifeste, un identifiant de groupe stable sera dérivé des références source des membres, pas du texte du corpus.

## Vérification reproductible

La commande `python3 scripts/audit_tsac_candidate_groups.py` relit l'index privé et le relevé des exclusions, vérifie leurs empreintes, l'unicité et la validité des références, puis recalcule les groupes sans écrire de fichier. Les deux entrées ligne par ligne restent dans `data/interim/`, ignoré par Git ; le script n'affiche que des agrégats.

| Contrôle | Résultat vérifié |
| --- | ---: |
| Liens candidats uniques | 374 |
| Lignes train retenues dans la projection | 11 827 |
| POS / NEG dans cette projection | 6 182 / 5 645 |
| Groupes au total | 11 484 |
| Groupes de plusieurs lignes | 209 |
| Lignes dans ces groupes | 552 |
| Taille du plus grand groupe | 10 |
| Groupes aux étiquettes mixtes | 7 (25 lignes) |
| Lignes test consultées par ce contrôle | 0 |

## Ce qui reste à faire

Nous devons encore fixer la proportion de validation et une graine, attribuer **chaque groupe entier** à train ou validation, puis vérifier la répartition POS/NEG et l'absence de groupes partagés. Les effectifs ci-dessus ne sont pas ceux d'une partition créée. Après ce contrôle, nous pourrons construire les manifestes et les cartes de données. L02 et L03 restent donc incomplets.
