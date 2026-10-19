# TuniScope — Ajout d'indices lexicaux au classifieur caractères

**Date :** 19 octobre 2026. **Livrables :** L04, L07, L10 et L12, contributions partielles. **État initial :** protocole fixé avant l'évaluation de cette nouvelle configuration.

## Question et contrastes fixés

La baseline caractères brute v2 reste notre meilleure référence initiale. Le lexique seul est moins performant et sa correction par négation ne l'améliore pas. Nous testons maintenant si ces informations peuvent néanmoins aider le classifieur caractères lorsqu'il apprend leurs poids, sans remplacer directement sa décision par celle des règles.

La configuration `configs/baselines_char_rules_v1.json` fixe trois systèmes TSAC : caractères seuls (contrôle), caractères avec indices lexicaux, caractères avec les mêmes indices corrigés par négation. MADAR comporte le contrôle caractères et l'ajout lexical, sans inversion de polarité dialectale. Nous n'ajoutons pas de recherche de poids ou d'hyperparamètres après observation des scores.

Le contraste principal de cette exploration est caractères+lexique versus contrôle, sur chaque tâche. Le contraste secondaire TSAC compare les deux versions augmentées, pour isoler l'heuristique de négation. Ce ne sont pas encore les contrastes finaux LLM/encodeur à confirmer sur trois graines et test gelé.

## Ressources et représentation

Les partitions, ordre des exemples et labels sont ceux des manifestes v2. Le bloc caractères utilise exactement les paramètres de `configs/baselines_grouped_v2.json` : texte brut, caractères 3–5, min_df=2, TF sous-linéaire, IDF et L2, sans passage en minuscules. La régression logistique conserve C=1, lbfgs, max_iter=1000, sans pondération des classes et avec un seul thread.

Les lexiques et contrôles sont reconstruits uniquement depuis le même train selon la configuration de la [baseline à règles](29_baselines_regles_lexicales.md). Nous n'utilisons ni les notes de revue de validation, ni ses labels pour construire ces ressources. Les annotations linguistiques proposées par IA ne deviennent pas des labels d'apprentissage.

Pour chaque texte, le bloc lexical contient une fraction de voix par classe, suivie d'un indicateur binaire de présence d'au moins un indice. Les fractions sont les voix de la classe divisées par le total des voix ; sans voix, elles valent zéro. Les colonnes suivent l'ordre des classes configuré. TSAC ajoute donc trois colonnes, MADAR quatre. Ces nombres ne sont pas des probabilités calibrées et la classe majoritaire de recours n'est pas injectée comme une réponse attendue.

La variante TSAC avec négation recalcule les mêmes colonnes en appliquant la fenêtre de trois mots de la baseline à règles. La taille, le lexique et l'échelle du bloc restent identiques. Aucune colonne supplémentaire de déclenchement ou de nombre de négations n'est ajoutée : nous isolons ici l'effet du changement des voix.

Le bloc lexical a un poids fixe de 1. Nous concaténons les colonnes sans renormalisation globale : les valeurs du bloc caractères restent inchangées. Les nouveaux coefficients sont appris sur train avec le même C. Ce choix ajoute des dimensions et modifie la géométrie globale ; il ne faut pas confondre cette expérience avec une comparaison où toutes les représentations auraient la même norme totale.

## Apprentissage, validation et limites

Chaque système ajuste son vocabulaire/IDF et son classifieur uniquement sur train. Les lexiques sont supervisés par les labels source du train puis appliqués à ce même train ; leurs associations peuvent donc surajuster ces données. Aucune performance train n'est présentée comme mesure de généralisation. La validation sert à mesurer les configurations fixées, sans filtrer les exemples où les règles sont absentes ou incertaines. Le test officiel reste fermé.

Les sources, labels et modèles historiques sont conservés. Une absence d'indice lexical produit un bloc nul, mais le modèle dispose toujours des caractères ; nous ne forçons alors ni abstention ni classe majoritaire. Les risques d'effets de domaine, noms propres, fenêtres de négation imparfaites et ironie restent ceux des règles initiales.

## Vérifications prévues et preuve attendue

Nous exécutons d'abord les tests fictifs et un smoke test sur 100 lignes par partition et tâche. Le contrôle doit reproduire le modèle caractères historique correspondant, en particulier son vocabulaire, ses IDF et ses prédictions. Les branches caractères doivent aussi être identiques entre contrôle et variantes augmentées. Les modèles de règles sont reconstruits depuis le train et comparés aux ressources de l'expérience précédente, sur les mêmes lignes.

Les checkpoints, prédictions et empreintes restent privés. Un vérificateur rejoue toutes les prédictions après rechargement, contrôle la forme des matrices, les ressources, paramètres, identifiants/groupes/labels et les métriques, et recalcule le macro-F1 depuis la matrice de confusion. Le rapport public généré ne contient que des agrégats.

Les métriques principales restent le macro-F1 sur toute la validation et les métriques par classe/slice. Nous rapportons les erreurs corrigées et introduites entre systèmes appariés, ainsi que le nombre de blocs lexicaux nuls et les coûts. Aucun écart observé sur cette validation ne prouvera à lui seul un gain significatif ou final. Sans gain, nous conserverons le résultat et la référence plus simple, sans modifier a posteriori le protocole pour inverser la conclusion.

## Exécution et décision du 19 octobre 2026

Les 84 tests Python passent, dont dix nouveaux contrôles fictifs pour les dimensions, la fidélité du bloc caractères, l'absence de renormalisation globale, les règles train-only, les colonnes nulles, l'indépendance des transformations aux labels de validation et le rechargement. Le smoke test a été exécuté sur 100 lignes par partition/tâche ; les cinq modèles ont été vérifiés avant le lancement complet.

Le run complet porte sur 10 053 lignes train et 1 774 validation TSAC, et 6 388 train et 800 validation MADAR. Les contrôles reproduisent exactement les prédictions historiques brutes v2. Les cinq branches caractères possèdent les mêmes vocabulaires/IDF que leurs références ; les lexiques reconstruits sont identiques aux modèles de règles précédents. Les modèles TSAC augmentés ajoutent trois colonnes aux 92 733 caractères, MADAR quatre aux 53 079. Les lexiques et vocabulaires restent privés.

Le [rapport généré](../reports/character_rules_v1_2026-10-19/README.md) fournit les scores, erreurs appariées, nombres de blocs nuls, empreintes et coûts. Les ajouts n'améliorent pas la référence sur cette validation : TSAC passe de 126 erreurs à 133 avec lexique et 132 avec négation ; MADAR passe de 39 à 71 erreurs avec lexique. Les différences de macro-F1 sont négatives sur les contrastes principaux.

Sur TSAC, le lexique corrige 43 erreurs du contrôle mais en introduit 50 ; sa variante de négation en corrige 42 et en introduit 48. Sur MADAR, 13 sont corrigées et 45 introduites ; une autre prédiction change entre deux classes erronées, d'où 59 différences pour 58 bascules correct/incorrect. La variante TSAC avec négation fait légèrement mieux que l'ajout lexical simple (sept erreurs corrigées, six introduites), sans dépasser les caractères seuls.

Nous conservons donc les caractères seuls comme référence initiale plus simple. Les variantes sont sauvegardées comme expériences négatives, pas intégrées comme amélioration retenue. La redondance ou le surajustement des indices pourraient contribuer aux dégradations, mais ces mécanismes n'ont pas été isolés par cette expérience. Nous ne modifions pas les poids ou seuils après le résultat et ne prétendons pas que tout lexique ou toute négation seraient inutiles en général.

Les modèles sont rechargés et toutes les prédictions sont rejouées ; matrices et macro-F1 sont vérifiés. Les contrôles portent aussi sur les valeurs numériques des blocs concaténés, l'identité des ressources train et l'alignement des exemples. Le test officiel, les labels, les sources et les manifestes ne sont pas modifiés. Les champs de provenance enregistrent l'empreinte du protocole avant l'ajout de ce bilan, pas celle du document enrichi.

Commandes exécutées :

```sh
sh scripts/project_uv.sh run --locked python -m unittest discover -s tests
sh scripts/project_uv.sh run --locked python scripts/run_character_rules.py --mode smoke --output outputs/baselines/2026-10-19_character_rules_v1_smoke
sh scripts/project_uv.sh run --locked python scripts/verify_character_rules.py outputs/baselines/2026-10-19_character_rules_v1_smoke
sh scripts/project_uv.sh run --locked python scripts/run_character_rules.py --mode full --output outputs/baselines/2026-10-19_character_rules_v1_full
sh scripts/project_uv.sh run --locked python scripts/verify_character_rules.py outputs/baselines/2026-10-19_character_rules_v1_full --report reports/character_rules_v1_2026-10-19
```

Une reproduction doit utiliser un nouvel identifiant de sortie : les anciens dossiers ne sont pas écrasés. Les notebooks, les autres références requises (notamment modèle de langage et BiGRU), la relecture humaine, la confirmation du contraste final et l'évaluation finale restent nécessaires. Cette expérience termine le premier essai contrôlé d'ajout lexical, pas les livrables complets L04/L07/L10/L12 ni une phase neuronale.
