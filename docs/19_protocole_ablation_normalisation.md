# TuniScope — Protocole préenregistré : texte brut ou normalisé

**Livrables concernés :** L01 (protocole expérimental), L02 (versions des données), L04 (future baseline) et L07 (future évaluation) ; **contributions partielles uniquement**. **Date :** 17 octobre 2026. **État :** protocole fixé avant tout score ; aucun modèle entraîné et aucun résultat d'ablation disponible.

**Suivi :** la première exécution a depuis été réalisée le 17 octobre ; ses résultats et limites figurent dans les [premières baselines](21_premieres_baselines.md). Le protocole ci-dessous conserve les choix formulés avant ce run.

## Question et principe

Nous voulons savoir si la normalisation prudente aide réellement la classification, plutôt que de supposer qu'un texte « plus propre » donne un meilleur modèle. Une **ablation** consiste ici à ne changer qu'un facteur expérimental : la représentation du texte. La comparaison porte sur `raw-v0` (texte source intact) contre [`norm-nfc-space-v1`](16_normalisation_minimale_v1.md) (Unicode NFC et espaces harmonisés). Les décisions d'[URL/mentions](17_audit_url_mentions.md) et de [tatweel/répétitions](18_audit_tatweel_repetitions.md) ne sont pas ajoutées à cette comparaison principale.

## Ce qui reste identique

Pour chaque tâche, les deux essais utiliseront exactement les mêmes références de lignes, étiquettes, groupes et partitions des [manifestes v1](11_manifestes_donnees_v1.md). TSAC conserve 10 053 lignes train et 1 774 lignes validation ; MADAR, 6 388 et 800. Le test officiel n'est pas consulté pour cette comparaison. Nous garderons la même famille de modèle, les mêmes hyperparamètres, le même code d'évaluation et la même métrique ; seule la colonne d'entrée sera différente (`text_raw` ou `text_norm`).

Comme première comparaison contrôlée, nous prévoyons une baseline de **TF-IDF sur n-grammes de caractères 3 à 5**, suivie d'une **régression logistique**. TF-IDF transforme les séquences de caractères en nombres pondérés ; la régression logistique apprend ensuite une frontière entre les étiquettes. Les paramètres de départ seront fixés dans une configuration versionnée **avant la première exécution** : `analyzer=char`, `ngram_range=(3,5)`, `lowercase=False`, `min_df=2`, `sublinear_tf=True`, `C=1.0`, `class_weight=None`, `max_iter=1000`, `solver=lbfgs`. Ces valeurs sont un plan initial, pas des paramètres déjà testés. La version de la bibliothèque et toute adaptation nécessaire à sa compatibilité devront être enregistrées avant le run.

Chaque vectoriseur TF-IDF sera **ajusté uniquement sur le train** de sa variante ; la validation sera seulement transformée par le vectoriseur déjà ajusté. Le classifieur sera appris sur le train, puis évalué sur la même validation figée pour les deux variantes. Aucune sélection de vocabulaire, de paramètre ou de règle ne se fera à partir du test.

**Précision avant le premier run, 17 octobre 2026 :** l'[environnement installé](20_environnement_python.md) fixe scikit-learn 1.9.1. Son analyseur de caractères harmonise déjà les suites d'espaces à l'intérieur du vectoriseur, ce qui est confirmé par le code installé et un contrôle synthétique. Nous conservons le même analyseur dans les deux branches : l'expérience mesure donc l'apport supplémentaire de NFC et du retrait des espaces périphériques. Une différence faible ou nulle ne démontrerait pas que la normalisation est inutile pour d'autres modèles. Toute autre transformation interne rencontrée devra être décrite avant d'interpréter les scores.

## Mesures et décision

La mesure principale sera le **macro-F1 de validation** : moyenne non pondérée des F1 de chaque classe, utile quand les classes n'ont pas la même taille. Nous enregistrerons aussi l'accuracy, la précision/rappel/F1 par classe, la matrice de confusion, les effectifs par écriture (`AR`, `LATIN`, `MIXED`, `UNKNOWN`), le temps et la mémoire si mesurables. Les résultats par écriture seront descriptifs lorsque les groupes sont petits ; `LATIN` ne signifie pas « français ».

Nous calculerons pour chaque tâche `Δ = macro-F1(normalisé) − macro-F1(brut)` sur **les mêmes exemples**. Si Δ est positif, cela soutiendra un avantage de la normalisation pour cette baseline précise ; s'il est négatif, elle peut avoir retiré ou modifié un indice utile ; s'il est très proche de zéro, nous ne conclurons pas à un gain pratique sans analyse complémentaire. Nous publierons **les deux scores et leurs erreurs**, même si une variante est retenue pour la suite. Une seule comparaison déterministe ne constitue ni une preuve de significativité ni le résultat final des trois graines prévus plus tard.

L'audit précédent a trouvé 874 textes TSAC train et 187 validation modifiés par v1, contre 18 et 3 dans MADAR. Un effet très faible sur MADAR est donc plausible, mais **ce n'est pas un résultat de modèle**. Nous ne créons ni tableau de scores fictif ni entrée d'expérience avant une exécution réelle. Une fois l'environnement fixé, chaque run réel sera inscrit dans `EXPERIMENTS.md` avec configuration, graines si applicables, empreintes des manifestes, commit, matériel et résultats.

## Vérification et limites

Avant l'entraînement, nous revérifierons les empreintes des sources/manifestes, l'identité des références des deux variantes, l'absence de groupe train/validation partagé et le fait que `text_raw` n'a pas changé. Après le run, nous vérifierons que les prédictions des deux variantes sont alignées par `id` et que les métriques sont calculées par le code, jamais saisies manuellement.

Ce protocole compare seulement une normalisation minimale sur deux corpus donnés. Il ne répond pas encore à l'effet sur l'Arabizi à lui seul, ni aux transformations plus fortes, ni au test final. **L01, L02, L04 et L07 ne sont pas achevés par ce document.**
