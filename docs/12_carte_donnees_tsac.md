# TuniScope — Carte de données TSAC

**Livrable :** L02, carte de données propre au corpus TSAC. **Version :** 0.3, 17 octobre 2026. **État :** manifestes v2 et regroupement des variantes vérifiés ; composition linguistique et contrôles finaux encore partiels.

## Identification et usage

Nous utilisons le [Tunisian Sentiment Analysis Corpus (TSAC)](https://github.com/fbougares/TSAC) pour une classification binaire du sentiment de commentaires tunisiens : `POS` et `NEG`. Une observation correspond à une ligne d'un des quatre fichiers `train_pos.txt`, `train_neg.txt`, `test_pos.txt` et `test_neg.txt`. L'étiquette d'origine provient du nom du fichier, et non d'un jugement que nous aurions produit pour chaque ligne.

L'archive du dépôt officiel a été acquise le 2 octobre 2026. Sa version locale et son empreinte SHA-256 sont consignées dans le [registre des sources](../SOURCE_REGISTER.md). Le dépôt affiche une licence LGPL-3.0 ; nous devons confirmer les obligations applicables à d'éventuelles données dérivées avant publication. Les commentaires originaux et les classeurs de revue restent hors de Git. Nous ne présentons pas le corpus comme représentatif de tous les avis tunisiens contemporains.

## Structure et effectifs

Les quatre fichiers sont des textes UTF-8 : une ligne physique correspond à un commentaire candidat. Le train officiel contient 13 669 lignes physiques, dont quatre vides ; le test officiel contient 3 400 lignes. Le [manifeste v1](11_manifestes_donnees_v1.md) conserve les références `(fichier, numéro de ligne)` et sépare les lignes retenues des exclusions, sans republier les commentaires.

| État | POS | NEG | Total |
| --- | ---: | ---: | ---: |
| Train officiel brut | 7 154 | 6 515 | 13 669 |
| Exclusions du train de travail | 972 | 870 | 1 842 |
| Train de travail v2 | 5 255 | 4 798 | 10 053 |
| Validation v2 extraite du train | 927 | 847 | 1 774 |
| Test officiel figé | 1 700 | 1 700 | 3 400 |

La validation est issue uniquement du train officiel, avec une cible de 15 % et la graine 42. Les 374 liens de similarité candidats internes sont utilisés comme **contrainte de groupement** : aucune famille repérée n'est partagée entre train et validation. Les liens n'établissent pas à eux seuls que deux textes ont le même sens. Le test officiel conserve toutes ses lignes et étiquettes, y compris les anomalies connues.

La [révision v2](23_manifestes_v2_et_baselines.md) ajoute les contraintes des familles à lettres répétées : 12 paires traversant les partitions v1 ont été explicitement jugées de même sens lors de notre revue individuelle. Les 49 familles détectées par le filtre restent chacune dans une partition, sans prétendre que toutes ont été relues. Les effectifs du tableau restent ceux de v1, mais 1 476 lignes sont échangées dans chaque sens. Les textes, étiquettes et exclusions sont identiques. Les groupes mixtes par label ne sont pas corrigés automatiquement.

## Préparation, contrôle et annotations

Les 1 842 exclusions concernent uniquement le train de travail. Le relevé conserve séparément les motifs : ligne vide ou non linguistique, chevauchement exact ou léger avec le test, paire train–test jugée de même sens lors de notre revue, doublon interne exact ou équivalent après la clé d'audit NFKC/espaces. Les motifs peuvent se cumuler ; leur somme n'est pas le nombre de lignes exclues. Les conflits POS/NEG encore présents dans le train après décontamination ont été résolus en gardant une occurrence portant le label choisi lors de notre revue, sans réécrire les fichiers sources.

La revue des cas ambigus a été réalisée par **une seule personne**. Douze groupes à étiquettes exactes contradictoires ont reçu un choix POS/NEG ; un treizième cas a été déclaré non linguistique. Nous avons jugé 124 paires longues de même sens et, parmi 160 paires courtes, 152 de même sens, sept de sens différent et une non linguistique. Pour les paires courtes de même sens, 75 décisions ont été saisies explicitement et 77 ont été déduites de notre consigne selon laquelle une cellule laissée vide signifiait « même sens ». Cette provenance reste tracée dans un registre privé. Il ne s'agit **pas** d'une double annotation indépendante ni d'une mesure d'accord interannotateurs.

La clé NFKC/espaces utilisée pour détecter les doublons n'est **pas** la normalisation destinée aux modèles. Un [chargeur raw-v0 en mémoire](14_chargement_donnees_en_memoire.md) fournit d'abord `text_raw` et `text_norm` identiques. La première [normalisation minimale `norm-nfc-space-v1`](16_normalisation_minimale_v1.md) a ensuite été auditée sur train/validation en RAM : 874 lignes train et 187 lignes validation changent, surtout pour les espaces ; aucun texte brut n'est réécrit. Un [audit séparé des URL et mentions](17_audit_url_mentions.md) ne reconnaît qu'une URL dans le train TSAC retenu et aucune mention ; le remplacement reste une variante candidate. La conversion en DataFrame pandas a été vérifiée le 17 octobre dans l'[environnement local du projet](20_environnement_python.md).

L'[audit du tatweel et des répétitions](18_audit_tatweel_repetitions.md) trouve huit lignes train avec tatweel mais 2 066 avec au moins une suite de trois lettres identiques. Réduire ces suites créerait neuf groupes exacts traversant train et validation ; nous conservons donc les répétitions dans la variante de référence.

Ces comptes d'audit décrivent **les affectations v1**. Dans v2, NFC/espaces modifie 917 textes train et 144 validation en mémoire ; les répétitions restent préservées. Le filtre de répétitions n'entraîne plus de chevauchement train/validation sous v2, car il sert au groupement et non à la transformation des entrées des modèles.

## Limites et contrôles restants

Le corpus provient de commentaires de réseaux sociaux ; son ancienneté, son domaine et ses graphies informelles peuvent limiter la généralisation. L'absence d'identifiants d'auteur ou de conversation exploitables empêche de prouver l'indépendance de ces unités entre partitions. La recherche de quasi-doublons est heuristique : une reformulation non repérée peut subsister. Le test officiel comprend ses propres doublons et au moins un cas non linguistique ; nous conserverons le score principal sur ce test, et distinguerons toute analyse diagnostique nettoyée.

L'[audit des textes retenus](15_audit_ecriture_longueur.md) compte dans le train **4 440 `AR`, 5 330 `LATIN`, 262 `MIXED` et 21 `UNKNOWN`** ; la validation compte respectivement **771, 957, 43 et 3**. Ces valeurs décrivent l'écriture, **pas la langue**. Les 24 lignes sans lettre arabe ou latine dans train/validation ne sont pas automatiquement invalides : certains émojis peuvent porter le sentiment. Les longueurs P50/P95 et les écarts entre POS/NEG figurent dans l'audit. La composition linguistique reste inconnue ; nous ne l'inventons pas. Nous devons également confirmer la citation bibliographique exacte et les droits de publication des dérivés avant la remise finale. **La présente carte est une contribution partielle à L02, pas sa clôture.**

Les comptes d'écriture ci-dessus sont historiques v1. Avec les nouvelles affectations v2, train contient **4 426 `AR`, 5 355 `LATIN`, 259 `MIXED`, 13 `UNKNOWN`** ; validation contient **785, 932, 46 et 11** respectivement. Les 24 références `UNKNOWN` sont inchangées au total, seulement réparties autrement. Le script de détection est identique et ne constitue toujours pas une identification de langue.
