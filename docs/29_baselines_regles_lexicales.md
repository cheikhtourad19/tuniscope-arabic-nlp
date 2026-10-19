# TuniScope — Protocole initial des baselines lexicales à règles

**Livrables :** L04, L07, L10 et L12, contributions partielles. **Configuration :** `configs/rule_baselines_v1.json`, fixée avant la première évaluation de ces règles.

## Question et périmètre

Nous construisons des références simples et inspectables : vote lexical pour le sentiment TSAC et les variétés MADAR, puis même lexique de sentiment avec ou sans portée locale de négation. Il s'agit d'une baseline hybride à lexique statistiquement induit, pas d'un dictionnaire expert ni d'une nouvelle architecture neuronale. Nous ne supposons pas que les mots sélectionnés expriment intrinsèquement une émotion ou une variété.

Les ressources sont dérivées exclusivement du train complet des manifestes v2. L'étude qualitative de 50 textes motive la règle, mais ses propositions IA ne sont pas traitées comme un jeu de référence validé. Les erreurs de validation précédemment examinées ne sont pas ajoutées au lexique. La validation sert seulement à mesurer cette configuration figée ; aucune recherche de paramètres ne sera engagée pour masquer un résultat négatif. Le test officiel reste fermé.

## Tokenisation et construction du lexique

La représentation `rule-nfc-casefold-v1` applique NFC puis casefold, reconnaît les unités alphanumériques Unicode (dont Arabizi) et conserve certaines ponctuations comme frontières. Les séquences techniques littérales `_xHHHH_` sont des frontières, pas des indices lexicaux. Les sources ne sont pas modifiées. Les émojis ne sont pas représentés dans ce premier lexique : les cas concernés restent dans l'évaluation et peuvent aboutir au recours majoritaire.

Pour chaque terme et classe, nous comptons le nombre de documents train qui contiennent le terme, une seule fois par document. Nous comparons sa fréquence dans la classe à sa fréquence dans les autres classes réunies, avec lissage additif 1 : logarithme du rapport `(df_classe+1)/(N_classe+2)` sur `(df_autres+1)/(N_autres+2)`. Nous ne parlons pas de log-odds : ce sont des rapports de fréquences lissées.

Un terme d'au moins deux caractères est associé à la classe de rapport maximal s'il apparaît dans au moins cinq documents de cette classe et si son log-rapport atteint 1. Nous retenons au plus 200 termes par classe, triés par rapport décroissant puis par forme lexicale. Les marqueurs et frontières discursives configurés sont exclus des indices. Les lexiques, dont les éventuels noms propres ou termes de domaine, restent privés. Leur interprétabilité est limitée par les corrélations et les labels source.

## Décision et couverture

Chaque indice distinct donne une voix à sa classe. Le score est un nombre de voix, pas une probabilité calibrée. Un mot répété ne multiplie pas sa contribution ; le même terme rencontré avec deux orientations de négation peut contribuer une fois à chaque polarité.

Nous considérons la règle décisive seulement si une classe gagne strictement et possède au moins une voix. Sans indice ou en cas d'égalité au sommet, nous utilisons la classe majoritaire calculée sur le train, avec un indicateur explicite de recours. Les éventuelles égalités de majorité suivent l'ordre des labels fixé dans la configuration.

Nous publions le macro-F1 sur **tous** les exemples, recours inclus. Nous distinguons la couverture lexicale (au moins un indice) de la couverture décisive (sans recours). Une performance calculée seulement sur les cas décisifs reste secondaire et ne remplace jamais le score global.

## Ablation de la négation

Nous conservons strictement le même lexique TSAC pour le vote seul et sa variante avec négation. Seuls les marqueurs, ruptures de proposition et exceptions effectivement attestés dans le train sont activés. Aucun patron générique de type ma…ch n'est ajouté dans cette version, pour ne pas transformer mécaniquement des noms en négations.

La règle inverse la polarité des indices situés dans les trois unités suivantes. Une ponctuation de frontière, un saut de ligne, un marqueur technique ou un adversatif met fin à cette fenêtre. Deux négations dans une fenêtre inversent la parité ; c'est une convention heuristique, pas une analyse fiable de la concordance négative tunisienne. La locution configurée « لا بد » ne déclenche pas l'inversion. Les marqueurs eux-mêmes ne donnent pas de voix.

Cette règle ignore notamment les cibles multiples, l'ironie, les suffixes attachés non reconnus et les longues dépendances. Elle peut dégrader les scores, en particulier si un mot est déjà associé statistiquement à une classe dans des contextes niés. Nous conserverons le résultat sans réétiqueter les données ni modifier la règle après observation des erreurs.

## Vérification prévue

Avant le run complet : tests fictifs de construction train-only, filtrage des indices, recours, portée, frontières, exception et fidélité JSON, puis smoke test sur 100 lignes par partition et tâche. Le run complet enregistrera lexiques, identifiants, empreintes, paramètres, coûts et prédictions hors Git. Un vérificateur reconstruira les lexiques depuis le train, rechargera les sorties et recalculera les métriques.

Les comparaisons avec les caractères exigent les mêmes identifiants, groupes et labels de validation v2. Elles comparent des systèmes différents, non un effet causal à représentation constante. La comparaison lexique seul versus négation est appariée sur les mêmes textes et ressources. L'intégration de ces règles au meilleur classifieur, les notebooks, la relecture linguistique et l'évaluation finale restent distincts et ne sont pas achevés par ce protocole.

## Exécution et bilan du 17 octobre 2026

Le smoke test sur 100 lignes par partition et tâche a produit cinq sorties vérifiées. À cette taille réduite, aucun terme TSAC ne satisfait les seuils fixés : ses deux variantes lexicales reviennent entièrement à la majorité, ce qui vérifie le traitement du lexique vide sans démontrer la qualité du système. Les tests fictifs couvrent séparément les décisions lexicales et les inversions. Nous n'avons pas abaissé les seuils après ce constat.

Le run complet a ensuite construit 400 termes TSAC (200 par classe) depuis les 10 053 lignes du train, et 586 termes MADAR (200 MSA, 200 TN, 186 EG) depuis 6 388 lignes. Les validations restent respectivement à 1 774 et 800 exemples. Les lexiques sont reconstitués depuis le train, les modèles JSON rechargés et toutes les prédictions rejouées ; matrices, macro-F1, slices et couverture sont recalculés. Le run caractères v2 servant de référence est également revérifié. La suite Python compte 74 tests réussis, dont 18 nouveaux contrôles sur données fictives.

Le [rapport de résultats généré](../reports/rule_baselines_v1_2026-10-17/README.md) fournit le tableau des scores globaux, couvertures et erreurs appariées, ainsi que les empreintes et coûts dans son résumé JSON. Le lexique améliore la référence majoritaire sur les deux tâches mais reste moins performant que la référence caractères. Il décide sans recours sur 973/1 774 commentaires TSAC et 635/800 phrases MADAR ; les autres exemples restent intégralement dans les métriques.

L'ablation de négation conserve exactement les mêmes 400 indices TSAC. Elle déclenche une fenêtre sur 77 commentaires, inverse au moins un indice sur neuf et modifie finalement quatre labels prédits : une erreur est corrigée, trois sont introduites. Son macro-F1 est légèrement inférieur à celui du lexique seul. Ces résultats ne démontrent pas que la négation serait inutile en général : ils concernent cette heuristique et ce lexique, où les associations ont déjà été apprises dans leurs contextes d'origine. Aucune règle n'est modifiée a posteriori pour inverser ce résultat.

Les commandes exécutées sont :

```sh
sh scripts/project_uv.sh run --locked python -m unittest discover -s tests
sh scripts/project_uv.sh run --locked python scripts/run_rule_baselines.py --mode smoke --output outputs/baselines/2026-10-17_rules_v1_smoke
sh scripts/project_uv.sh run --locked python scripts/verify_rule_baselines.py outputs/baselines/2026-10-17_rules_v1_smoke
sh scripts/project_uv.sh run --locked python scripts/run_rule_baselines.py --mode full --output outputs/baselines/2026-10-17_rules_v1_full
sh scripts/project_uv.sh run --locked python scripts/verify_rule_baselines.py outputs/baselines/2026-10-17_rules_v1_full --report reports/rule_baselines_v1_2026-10-17
```

Les générateurs refusent d'écraser les dossiers existants : une reproduction doit choisir un nouvel identifiant de sortie. L'empreinte du protocole enregistrée par le run désigne sa version avant l'ajout de ce bilan, pas le document enrichi après exécution.

Les baselines lexicales initiales et l'ablation de négation sont réalisées et vérifiées. L'ajout contrôlé d'indices lexicaux ou de négation au meilleur classifieur caractères reste à tester avec un protocole distinct. Ce résultat n'achève ni les notebooks L04/L07, ni le rapport L10, ni la relance indépendante L12 ; il ne clôture pas non plus la relecture des propositions linguistiques.
