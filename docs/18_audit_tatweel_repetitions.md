# TuniScope — Audit du tatweel et des répétitions de lettres

**Livrables concernés :** L02 (qualité, groupes et prévention des fuites), contribution partielle ; L03 (audit reproductible), contribution partielle. **Phase :** S3. **Date :** 16 octobre 2026.

## Question et méthode

Le *tatweel* (`ـ`) est un caractère d'allongement graphique de l'écriture arabe. Il faut le distinguer d'une **répétition de lettres** comme trois occurrences consécutives de la même lettre : dans des commentaires, cette répétition peut exprimer une insistance ou un sentiment. Nous avons audité ces deux phénomènes **séparément** après la normalisation NFC/espaces v1, sur les seules lignes train et validation des manifestes v1. Le test officiel n'a pas été ouvert.

Le script `python3 scripts/audit_elongation_policy.py` repère le tatweel et les suites d'au moins trois lettres Unicode identiques (arabes ou latines, en excluant le tatweel, les chiffres et la ponctuation). Il simule deux règles candidates indépendantes : supprimer le tatweel, ou ramener chaque suite d'au moins trois lettres identiques à deux. Il n'écrit pas de texte et n'applique aucune de ces règles au dataset. La détection par caractères identiques peut manquer des allongements avec diacritiques intercalés ; elle n'est pas une analyse linguistique complète.

## Résultats

| Corpus / partition | Lignes avec tatweel | Lignes avec ≥ 3 lettres identiques | Changées si tatweel supprimé | Changées si répétition réduite |
| --- | ---: | ---: | ---: | ---: |
| TSAC train | 8 | 2 066 | 8 | 2 066 |
| TSAC validation | 2 | 363 | 2 | 363 |
| MADAR train | 1 | 1 | 1 | 1 |
| MADAR validation | 0 | 0 | 0 | 0 |

Dans TSAC train, les suites de lettres concernent 957 lignes POS et 1 109 NEG ; dans la validation, 188 POS et 175 NEG. Supprimer le tatweel ne crée aucun nouveau groupe exact train/validation dans le périmètre audité. **Réduire les répétitions crée neuf groupes de textes exactement identiques entre train et validation TSAC**, ainsi que 40 groupes de doublons au sein d'une partition. Dans MADAR, aucune des deux règles candidates ne crée de recouvrement exact train/validation ; les 82 groupes déjà répétés au sein d'une partition restent présents.

## Décision

Nous **conservons le tatweel et les répétitions** dans `norm-nfc-space-v1`, notre première variante normalisée de référence. Le bénéfice potentiel du retrait du tatweel est faible dans ces données retenues ; une ablation ultérieure peut le mesurer sans modifier le protocole de base. La réduction automatique des lettres répétées est particulièrement risquée ici : elle effacerait une possible intensité sentimentale et introduirait neuf fuites exactes dans les partitions actuelles. Nous ne l'adoptons pas. Une adoption future exigerait un nouveau regroupement/versionnement des partitions avant entraînement et une nouvelle justification scientifique.

Les **18 tests locaux** passent après correction d'un détail technique révélé par les tests : Unicode classe lui-même le tatweel parmi les lettres, mais notre règle doit explicitement l'exclure des répétitions. Ces observations ne concernent ni le test officiel ni les performances d'un modèle. Les décisions d'ablation `raw-v0` versus `norm-nfc-space-v1` restent à évaluer sur validation ; **L02, L03 et S3 ne sont pas clos.**

**Précision du 17 octobre :** les neuf collisions créées par cette règle sont aussi des signaux de parenté possible entre variantes brutes déjà réparties entre train et validation. Conserver les répétitions n'établit pas leur indépendance ; une revue sémantique reste nécessaire. Nous signalons cette limite dans les [premières baselines](21_premieres_baselines.md), sans déclarer ces paires de même sens ni modifier le split d'après les scores.
