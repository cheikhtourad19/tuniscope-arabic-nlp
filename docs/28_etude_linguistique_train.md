# TuniScope — Protocole de l'étude linguistique sur le train

**Livrables :** contributions à L04, L10 et L12. La préparation du lot ne suffit pas à terminer l'étude linguistique.

## Objectif

Nous distinguons cette étude des 60 erreurs de validation : nous examinons ici des textes du train pour étudier les mots porteurs de polarité, la portée de la négation, la cible et les ambiguïtés. Nous pouvons ensuite proposer des règles sans induire un lexique depuis la validation. Aucun nouveau modèle neuronal ou analyseur syntaxique n'est installé.

Nous choisissons une annotation légère des indices pertinents, plutôt qu'une prétendue analyse syntaxique complète. Dans ce contexte, POS au sens *part of speech* désigne une catégorie grammaticale, par exemple nom, verbe ou adjectif ; il ne faut pas le confondre avec le label de sentiment POS. Une proposition IA sera attribuée comme telle et séparée de sa validation humaine.

## Sélection fixée avant lecture du lot

La configuration `configs/linguistic_sample_v1.json` fixe 50 textes des manifestes v2, exclusivement du train : 25 commentaires TSAC et 25 phrases MSA de MADAR. TSAC constitue notre volet d'usage tunisien, sans prétendre que tous ses mots soient exclusivement tunisiens. MADAR fournit le volet standard. Le choix équilibré entre ces deux domaines est qualitatif, pas représentatif des corpus.

Pour TSAC, nous retenons 13 labels source POS et 12 NEG, répartis entre neuf textes AR, huit LATIN et huit MIXED. Cette répartition concerne les écritures Unicode, pas une identification certaine des langues. Pour MSA, nous retenons 25 textes AR. Nous exigeons 4 à 50 unités alphanumériques et au plus 320 caractères, pour privilégier des textes suffisamment développés pour l'étude ; ce filtre ne supprime aucune ligne des datasets d'entraînement.

Nous enrichissons le lot en marqueurs **candidats** de négation : deux cas par combinaison script/label TSAC et huit cas MSA. Les autres places n'ont pas de marqueur reconnu par ce dépistage. La regex, limitée et imparfaite, figure dans le sélecteur. Un nom ou une formule peut correspondre à son motif sans constituer une négation, et une vraie négation peut ne pas être détectée. Les comptes de candidats ne sont donc pas des annotations grammaticales.

Dans chaque strate, nous ordonnons les identifiants par SHA-256 de `42|id`, puis conservons au plus un cas par groupe et texte NFC/espaces. Une strate insuffisante déclenche une erreur, sans remplacement silencieux ni sélection selon les scores des modèles. Le script ne charge ni prédiction ni texte de validation/test.

## Champs d'analyse et critères

Pour chaque cas, nous devons conserver :

1. Le texte et sa référence privée, le label source et le split.
2. Les mots ou expressions porteurs de polarité, ou leur absence explicite.
3. Quelques indices grammaticaux : extrait exact et rôle proposé (nom, verbe, adjectif, particule, etc.), sans prétendre étiqueter tous les tokens.
4. La négation : présente, absente ou incertaine ; marqueur observé et portée proposée. Nier une action ne transforme pas automatiquement l'ensemble du texte en sentiment négatif.
5. La cible de l'appréciation, ou une cible non identifiable. Un label de sentiment n'est pas créé pour MADAR, qui n'en fournit pas.
6. Les ambiguïtés et l'ironie éventuelle, sans inventer de contexte ni d'intention.
7. L'origine humaine ou IA des annotations et leur état de relecture. Un champ rédigé par IA ne compte pas comme une seconde annotation humaine.

Les labels du corpus ne sont pas réécrits. Les annotations linguistiques constituent un registre d'étude distinct, pas un nouveau jeu de référence validé. Les termes issus de ce lot peuvent motiver une règle candidate sur train, mais son utilité devra être mesurée sur validation avec couverture et comparaison à la baseline. Aucune règle par identifiant de cas n'est autorisée.

## Vérifications et travail restant

Le sélecteur doit vérifier le split, les empreintes des sources via le chargeur, les quotas, l'unicité des identifiants/groupes/textes et la reproductibilité. Les textes et notes restent sous `data/interim/` ou `explanation NLP/`, ignorés par Git. Les documents publics ne contiennent que le protocole et des résultats agrégés.

L'étude des 50 textes ne sera déclarée terminée qu'après rédaction et vérification des champs d'analyse, avec les incertitudes et la provenance explicites. Le test effectif d'une règle de négation, la baseline à règles, le notebook et la discussion dans le rapport restent distincts et nécessaires pour achever les contributions correspondantes à L04/L10/L12.

## Première rédaction assistée et observations

Les 50 cas ont maintenant une proposition IA pour un extrait grammatical, la négation et sa portée, la cible, la lecture de polarité et les réserves pragmatiques. Nous avons choisi des rôles légers (nom, verbe, adjectif, emploi adverbial, particule ou expression), pas un étiquetage exhaustif ni une annotation grammaticale de référence. Les champs d'ironie indiquent généralement une intention non établie, plutôt qu'une absence certaine d'ironie. Aucun cas n'est encore validé humainement.

Le [rapport généré](../reports/linguistic_study_train_v1/README.md) décrit les comptes du lot. Parmi les 20 cas dépistés par les marqueurs, deux reçoivent une proposition IA d'absence de négation ; cinq autres textes sans marqueur reconnu reçoivent une proposition de négation. Les statuts proposés comprennent la négation grammaticale, les réponses négatives et certaines constructions privatives ; ils ne se limitent pas à une particule précédant un adjectif. Ces observations ne sont pas des mesures de précision/rappel validées.

Nous observons notamment qu'une négation peut nier une propriété défavorable, décrire un passé révolu ou concerner un autre interlocuteur que la cible de l'éloge. Les réponses négatives isolées ne portent pas nécessairement sur toute la phrase suivante. Des constructions interrogatives et des locutions peuvent ressembler à des négations. Enfin, certaines écritures MIXED contiennent des restes techniques plutôt qu'un mélange linguistique réel.

Ces constats motivent une règle candidate de portée locale avec limites de proposition et exceptions explicites, **sans en démontrer le gain**. Son lexique éventuel restera issu du train. Nous n'intégrerons pas cette règle au meilleur modèle appris sans comparaison contrôlée.

`scripts/render_linguistic_study.py` reconstitue la sélection depuis les sources, vérifie les empreintes, les 50 identifiants, les extraits exacts et les champs obligatoires. Il génère les notes détaillées privées et le rapport agrégé public, avec provenance des propositions. La sélection figée conserve son état initial de zéro annotation ; le rapport distingue ce snapshot de ses 50 textes maintenant documentés par IA. Huit nouveaux tests fictifs accompagnent ces contrôles, pour 56 tests Python réussis au total. Ces tests valident la cohérence technique, pas la justesse des interprétations linguistiques.

Une première rédaction des 50 analyses est donc disponible. La relecture et les réserves restent ouvertes ; aucune double annotation indépendante, évaluation de règle, baseline à règles ou clôture de L04/L10/L12 n'est revendiquée.
