# TuniScope — Analyse qualitative des erreurs classiques avec assistance IA

**Date :** 17 octobre 2026. **Livrables :** L07, L10 et L12, contributions partielles.

## Objet et origine des jugements

Nous avons renseigné les avis sur les 60 labels du lot décrit dans le [protocole de revue](26_preparation_analyse_erreurs_baselines.md), puis délégué à l'IA les catégories et explications manquantes. Nous conservons les 60 avis et les 11 catégories déjà choisis, sans les présenter comme une annotation indépendante. L'IA ajoute 49 catégories principales et 60 explications courtes, chacune explicitement marquée « IA ». Ces propositions restent à relire et ne deviennent pas des jugements humains par le seul fait de leur délégation.

Le [rapport agrégé généré](../reports/error_analysis_assisted_2026-10-17/README.md) contient les comptes vérifiés. Les textes, références individuelles, notes et registres détaillés restent dans les dossiers privés ignorés par Git. La copie complétée n'écrase pas notre classeur original. Les labels source, les prédictions, les partitions et les scores sont inchangés.

## Lecture des observations

Pour TSAC, les descriptions font notamment ressortir des avis mixtes, des expressions figurées, des graphies Arabizi et des signes non verbaux. Ces signes ne sont pas automatiquement invalides : un émoji peut exprimer une appréciation. À l'inverse, un fragment décoratif peut rester non interprétable. La catégorie « Autre » contient plusieurs phénomènes différents ; elle ne représente pas une cause unique.

Pour MADAR, les hypothèses portent principalement sur des indices morphologiques et des formulations communes aux variétés. La provenance dialectale de la traduction ne garantit pas que le texte isolé présente un indice distinctif. Un futur, un démonstratif ou une forme familière peut fournir un indice utile, mais sa présence ne suffit pas à expliquer le comportement du classifieur sans analyse de ses caractéristiques.

Nous conservons une catégorie humaine « Négation » bien qu'aucune négation explicite ne soit visible dans le fragment correspondant. La note signale cette réserve au lieu de modifier silencieusement notre choix. Le total de deux cas classés dans cette catégorie ne signifie donc pas deux négations confirmées. Certaines lectures restent incertaines, notamment pour les expressions isolées ou la possibilité d'ironie. Une explication rédigée n'est pas une incertitude résolue.

## Validité scientifique

Le lot couvre 30 erreurs de validation par tâche et des groupes distincts, mais il n'est pas proportionnel aux erreurs totales. Les comptes décrivent ce lot, pas la fréquence des causes dans le corpus. Les labels et prédictions étaient visibles. Nous n'avons ni second annotateur humain indépendant ni accord interannotateurs à annoncer.

Les explications sont des hypothèses linguistiques, non des attributions causales établies. Nous n'avons pas démontré qu'un mot donné a provoqué une décision, ni mesuré ici l'effet du marqueur technique présent dans certaines chaînes. Aucun entraînement, nouvelle prédiction ou correction rétroactive de métrique n'est effectué.

La validation peut guider notre diagnostic et la sélection des expériences. Elle ne doit pas servir à construire un lexique induit, à ajouter ces exemples au train ou à concevoir des exceptions par identifiant. Les ressources apprises doivent rester issues du train. Le test officiel n'est pas chargé pour cette analyse.

## Reproduction et contrôles

1. Préserver l'instantané humain créé par `scripts/import_error_review.mjs` et l'entrée figée du lot.
2. Conserver les propositions IA dans un fichier privé, avec leur provenance distincte. `scripts/complete_error_review.mjs` vérifie les empreintes avant de compléter une copie et refuse tout écrasement d'une sortie existante.
3. Vérifier les champs conservés, les nouvelles valeurs, les compteurs et les contrôles natifs par `scripts/verify_assisted_error_review.py --report`. Le rapport public est généré depuis le registre vérifié, sans recopier les textes protégés.

Le classeur complété a été rendu et rouvert. Une lecture indépendante de son XML confirme les valeurs, les listes de choix, les largeurs et les limites des volets figés ; les deux compteurs indiquent 30 cas documentés. La position de défilement est réinitialisée par l'export. Nous ne revendiquons pas d'essai dans l'application Excel native. Dix assertions fictives de complétion et quatre tests de contrôles XML accompagnent les vérifications réelles ; la suite Python totale compte 48 tests réussis.

## État des livrables et suite

Le premier lot de 30 cas par classifieur classique est documenté avec assistance IA, mais la relecture de ces propositions reste à faire, en priorité sur les réserves et expressions incertaines. Nous pouvons préparer cette relecture sans recommencer le travail déjà saisi. Les notebook(s) d'évaluation, les analyses des modèles adaptés et l'évaluation finale avec bootstrap restent manquants pour L07. Cette synthèse fournit de la matière pour L10, sans constituer le rapport final. L12 reste partiel, notamment pour la relance indépendante.

L'étude linguistique de 50 phrases et les baselines à règles constituent des travaux distincts encore ouverts dans la roadmap. Cette revue ne les remplace pas et n'autorise pas automatiquement un passage au fine-tuning neuronal.
