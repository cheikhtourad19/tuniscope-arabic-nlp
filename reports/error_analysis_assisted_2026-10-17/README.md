# TuniScope — Analyse qualitative assistée des erreurs classiques

**Livrables :** contribution à L07, L10 et L12. Lot initial documenté, livrables complets non achevés.

## Origine des champs

Nous conservons 60 avis humains sur les labels et les 11 catégories déjà choisies. Sur délégation explicite, l'IA a proposé les 49 catégories manquantes et les 60 hypothèses courtes. Ces ajouts ne sont pas encore validés humainement. Le classeur original est conservé.

| Corpus | Cas documentés | Avis humains | Catégories humaines | Catégories IA | Hypothèses IA |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 30 | 30 | 5 | 25 | 30 |
| MADAR | 30 | 30 | 6 | 24 | 30 |

## Catégories principales du lot

Une seule catégorie principale par cas. Les catégories humaines et IA sont agrégées ici, mais leur origine est distinguée dans le registre privé. Ce sont des descriptions proposées, pas des causes prouvées.

| Corpus | Catégorie | Nombre |
| --- | --- | ---: |
| TSAC | Émoji / symbole | 3 |
| TSAC | Mélange de langues | 1 |
| TSAC | Autre | 7 |
| TSAC | Avis mixte / ambigu | 9 |
| TSAC | Orthographe / Arabizi | 5 |
| TSAC | Manque de contexte | 2 |
| TSAC | Incertain | 1 |
| TSAC | Négation | 2 |
| MADAR | Variétés proches / texte commun | 11 |
| MADAR | Orthographe / morphologie | 15 |
| MADAR | Lexique partagé | 3 |
| MADAR | Texte très court | 1 |

## Interprétation et réserves

Pour TSAC, nous relevons notamment la polarité mixte ou ambiguë, les graphies Arabizi, les expressions figurées et les signes non verbaux. La catégorie Autre regroupe des phénomènes différents, détaillés seulement dans les notes privées. Leur regroupement n'est pas un mécanisme linguistique unique.

Pour MADAR, nous examinons les indices morphologiques et le vocabulaire partagé. Une phrase courte peut manquer d'indices distinctifs. Un label issu de la provenance du corpus ne garantit pas que la variété soit identifiable dans chaque texte isolé.

Une catégorie humaine de négation est conservée alors qu'aucune négation explicite n'est visible dans le fragment correspondant. Sa note signale la réserve. Le compte de deux cas de négation ne signifie donc pas deux négations linguistiquement confirmées.

Certaines lectures restent incertaines, y compris lorsqu'une hypothèse est renseignée. Nous n'inventons ni contexte, ni intention ironique certaine, ni correction du label. L'absence de catégorie ironie/sarcasme comme catégorie principale n'établit pas une absence d'ironie.

Le lot est qualitatif et non proportionnel, choisi parmi les erreurs de validation de la référence caractères brute v2. Ces comptes ne mesurent pas la prévalence des causes dans le corpus. Les labels et prédictions étaient visibles. Cette procédure ne fournit ni seconde annotation humaine indépendante, ni kappa, ni explication causale des poids du modèle.

## Vérification et suite

Le XLSX complété a été rendu, rouvert et contrôlé. Une lecture indépendante du XML vérifie les champs préservés, les nouvelles valeurs, les listes de choix, les largeurs, les volets et les deux compteurs à 30. Aucun essai dans l'application Excel native n'est revendiqué. La position de défilement enregistrée est réinitialisée par l'export, sans changer les limites des volets figés. Aucun label, score, source, manifeste ou modèle n'est modifié. Aucun texte du test officiel n'est chargé.

La relecture des propositions IA reste à faire, en priorité pour les réserves et expressions incertaines. L07 reste partiel : notebooks, modèles ultérieurs, évaluation finale et bootstrap restent à réaliser. Le travail syntaxique sur 50 phrases est distinct. Aucun texte du lot ne doit être ajouté au train ni à un lexique induit depuis la validation.
