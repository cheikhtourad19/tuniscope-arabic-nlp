# TuniScope — Carte de données TSAC

**Livrable :** L02, carte de données propre au corpus TSAC. **Version :** 0.1, 16 octobre 2026. **État :** fondée sur les sources et manifestes v1 vérifiés ; composition par écriture et copie de travail encore à compléter.

## Identification et usage

Nous utilisons le [Tunisian Sentiment Analysis Corpus (TSAC)](https://github.com/fbougares/TSAC) pour une classification binaire du sentiment de commentaires tunisiens : `POS` et `NEG`. Une observation correspond à une ligne d'un des quatre fichiers `train_pos.txt`, `train_neg.txt`, `test_pos.txt` et `test_neg.txt`. L'étiquette d'origine provient du nom du fichier, et non d'un jugement que nous aurions produit pour chaque ligne.

L'archive du dépôt officiel a été acquise le 2 octobre 2026. Sa version locale et son empreinte SHA-256 sont consignées dans le [registre des sources](../SOURCE_REGISTER.md). Le dépôt affiche une licence LGPL-3.0 ; nous devons confirmer les obligations applicables à d'éventuelles données dérivées avant publication. Les commentaires originaux et les classeurs de revue restent hors de Git. Nous ne présentons pas le corpus comme représentatif de tous les avis tunisiens contemporains.

## Structure et effectifs

Les quatre fichiers sont des textes UTF-8 : une ligne physique correspond à un commentaire candidat. Le train officiel contient 13 669 lignes physiques, dont quatre vides ; le test officiel contient 3 400 lignes. Le [manifeste v1](11_manifestes_donnees_v1.md) conserve les références `(fichier, numéro de ligne)` et sépare les lignes retenues des exclusions, sans republier les commentaires.

| État | POS | NEG | Total |
| --- | ---: | ---: | ---: |
| Train officiel brut | 7 154 | 6 515 | 13 669 |
| Exclusions du train de travail | 972 | 870 | 1 842 |
| Train de travail v1 | 5 255 | 4 798 | 10 053 |
| Validation v1 extraite du train | 927 | 847 | 1 774 |
| Test officiel figé | 1 700 | 1 700 | 3 400 |

La validation est issue uniquement du train officiel, avec une cible de 15 % et la graine 42. Les 374 liens de similarité candidats internes sont utilisés comme **contrainte de groupement** : aucune famille repérée n'est partagée entre train et validation. Les liens n'établissent pas à eux seuls que deux textes ont le même sens. Le test officiel conserve toutes ses lignes et étiquettes, y compris les anomalies connues.

## Préparation, contrôle et annotations

Les 1 842 exclusions concernent uniquement le train de travail. Le relevé conserve séparément les motifs : ligne vide ou non linguistique, chevauchement exact ou léger avec le test, paire train–test jugée de même sens lors de notre revue, doublon interne exact ou équivalent après la clé d'audit NFKC/espaces. Les motifs peuvent se cumuler ; leur somme n'est pas le nombre de lignes exclues. Les conflits POS/NEG encore présents dans le train après décontamination ont été résolus en gardant une occurrence portant le label choisi lors de notre revue, sans réécrire les fichiers sources.

La revue des cas ambigus a été réalisée par **une seule personne**. Douze groupes à étiquettes exactes contradictoires ont reçu un choix POS/NEG ; un treizième cas a été déclaré non linguistique. Nous avons jugé 124 paires longues de même sens et, parmi 160 paires courtes, 152 de même sens, sept de sens différent et une non linguistique. Pour les paires courtes de même sens, 75 décisions ont été saisies explicitement et 77 ont été déduites de notre consigne selon laquelle une cellule laissée vide signifiait « même sens ». Cette provenance reste tracée dans un registre privé. Il ne s'agit **pas** d'une double annotation indépendante ni d'une mesure d'accord interannotateurs.

La clé NFKC/espaces utilisée pour détecter les doublons n'est **pas** la normalisation finale destinée aux modèles. Aucune table `text_raw`/`text_norm` et aucun DataFrame d'entraînement ne sont encore matérialisés. Le script et le protocole qui chargeront le texte depuis les originaux suivant le manifeste restent à construire.

## Limites et contrôles restants

Le corpus provient de commentaires de réseaux sociaux ; son ancienneté, son domaine et ses graphies informelles peuvent limiter la généralisation. L'absence d'identifiants d'auteur ou de conversation exploitables empêche de prouver l'indépendance de ces unités entre partitions. La recherche de quasi-doublons est heuristique : une reformulation non repérée peut subsister. Le test officiel comprend ses propres doublons et au moins un cas non linguistique ; nous conserverons le score principal sur ce test, et distinguerons toute analyse diagnostique nettoyée.

La répartition vérifiée par écriture (`AR`, `LATIN`, `MIXED`) et par langue n'est pas encore disponible ; nous ne l'inventons pas. Elle sera ajoutée après définition d'une règle de détection et chargement des textes. Nous devons également confirmer la citation bibliographique exacte et les droits de publication des dérivés avant la remise finale. **La présente carte est une contribution partielle à L02, pas sa clôture.**
