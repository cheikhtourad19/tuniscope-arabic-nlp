# Baseline de langage à n-grammes de caractères

**Date :** 4 novembre 2026. **Contributions partielles : L04, L07, L10 et L12.** Configuration `configs/character_lm_v1.json`, fixée avant les premières mesures de cette référence.

## Objectif et portée

Nous complétons la référence « modèle de langage » de la section 8 du cahier des charges. Elle prédit une distribution du caractère suivant ; elle n'attribue pas de label POS/NEG ou MSA/TN/EG. Nous entraînons un modèle par corpus, en réunissant ses classes dans le train, et évaluons la validation v2 du même corpus. Les labels ne conditionnent pas le modèle.

Nous comparons un unigramme sans contexte à un trigramme utilisant les deux caractères précédents. Cette référence statistique fonctionne sur CPU, ne requiert pas de poids préentraînés et ne constitue pas un démarrage de la phase neuronale. Nous ne lui attribuons ni compréhension du sens ni capacité de dialogue.

## Protocole fixé avant exécution

- Manifestes v2 explicites, train et validation existants ; aucun accès au test officiel.
- Une seule représentation `raw-v0`, sans normalisation, découpe en mots, troncature ou filtrage d'exemples.
- Un caractère correspond à un **point de code Unicode**, pas nécessairement à un graphème visible : une marque diacritique peut être une unité distincte.
- Vocabulaire constitué de tous les caractères du train, avec symboles techniques distincts `UNK` et `EOS`. Les deux ordres partagent exactement le même vocabulaire et les mêmes cibles de validation au sein d'un corpus et d'un mode.
- Un caractère absent du train devient `UNK` à l'évaluation. Nous mesurons le nombre et le taux de ces unités, sans étendre le vocabulaire avec la validation.
- Chaque document recommence avec `n−1` marqueurs `BOS`. Il se termine par une cible `EOS`. Les contextes ne traversent jamais la frontière entre deux textes.
- `BOS` n'est pas une cible et n'entre pas dans le dénominateur du vocabulaire prédit. Les marqueurs techniques ont des identifiants distincts des caractères littéraux.
- Lissage additif `alpha = 0,1`, même valeur pour les deux ordres, sans recherche après lecture des résultats. Un contexte jamais observé donne une distribution uniforme sur le vocabulaire des cibles.
- Modèles déterministes : pas d'initialisation aléatoire ni de trois graines artificiellement présentées comme trois expériences indépendantes.
- Smoke sur 100 lignes par partition/corpus, sélection déterministe par SHA-256 des identifiants ; vérification avant le run complet. Le vocabulaire du smoke vient de son seul train et ne doit pas être comparé directement à celui du full.

## Probabilités et métriques

Pour un contexte `h` et une cible `c`, la probabilité est :

```text
P(c | h) = (compte(h,c) + alpha) / (compte(h) + alpha × V)
```

`V` est le nombre de caractères train distincts plus `UNK` et `EOS`. Les comptes sont appris uniquement sur le train. Nous utilisons les logarithmes pour éviter le sous-dépassement numérique des produits de petites probabilités.

Nous enregistrons la somme des logarithmes négatifs `NLL`, le nombre de cibles `N`, puis :

```text
entropie croisée = NLL / N              (nats par cible)
bits par cible  = NLL / (N × ln(2))
perplexité      = exp(NLL / N)
```

`N` inclut un `EOS` par document et exclut les `BOS`. Nous agrégeons les sommes et les effectifs avant d'exponentier ; nous ne moyennons pas les perplexités par phrase. Une perplexité plus faible indique une meilleure prévision des cibles selon ce protocole ; ce n'est pas une accuracy, un pourcentage ou une mesure du sentiment.

Les deux ordres sont comparables **dans un même corpus**, avec le même vocabulaire, les mêmes exemples et le même traitement des frontières. Nous ne classons pas TSAC et MADAR selon leur perplexité : domaines, textes et vocabulaires diffèrent. Nous ne comparons pas cette perplexité à celle d'un tokenizer Transformer ou d'un autre prétraitement.

## Vérification et conservation des preuves

Les tests fictifs vérifient les probabilités à la main, leur somme à un, les frontières de document, les inconnus sans fuite, la sérialisation et l'agrégation de perplexité. Les modèles JSON et les scores par ligne restent dans `outputs/`, ignoré par Git : un vocabulaire et des comptes sont des dérivés des corpus.

Le vérificateur recharge chaque modèle, reconstruit ses comptes depuis le train autorisé, vérifie les empreintes et rejoue toute la validation. Il recalcule séparément la formule et l'agrégation des scores ; un simple fichier de résultats écrit n'est pas considéré comme preuve suffisante. Les sorties publiques sont limitées à des statistiques agrégées et à un rapport généré par le code.

La clôture de L04 exige encore les autres composants manquants, notamment la BiGRU. Cette étape ne termine pas les expériences neuronales, les trois graines de leurs runs finaux, la relecture humaine des analyses ou le test figé.

## Référence

Jurafsky et Martin, *Speech and Language Processing*, chapitre 3, [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf), sections sur l'évaluation par perplexité et le lissage. Nous adaptons les unités aux caractères et fixons ci-dessus nos conventions de frontière et de vocabulaire ; il ne s'agit pas d'une reproduction d'un résultat publié sur TSAC ou MADAR.

## Exécution et résultats vérifiés du 4 novembre 2026

Les 12 tests fictifs réussissent, y compris un calcul manuel des probabilités, l'absence d'apprentissage à l'évaluation, le rechargement et le rejet de scores altérés même après mise à jour de leur empreinte. Le smoke réel a ensuite été exécuté et vérifié sur 100 lignes de train et 100 de validation par corpus. Le lancement full impose ce contrôle préalable.

Le run complet `outputs/language_models/2026-11-04_char_lm_v1_full/` comprend quatre modèles JSON, quatre fichiers privés de scores par document et un `results.json` de provenance. Le vérificateur a reconstruit les comptes depuis le train v2, rechargé les quatre checkpoints et rejoué les 2 574 documents de validation pour chacun des deux ordres. Les sources, les manifestes et le test officiel restent inchangés.

| Corpus | Unigramme : perplexité ↓ | Trigramme : perplexité ↓ | Train | Validation |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 41,790861 | 11,128590 | 10 053 | 1 774 |
| MADAR | 23,802216 | 9,222734 | 6 388 | 800 |

Dans cette configuration, les deux caractères précédents améliorent la prévision par rapport à la fréquence sans contexte. Nous n'en déduisons pas un gain de classification ou de compréhension. Les vocabulaires prédits contiennent respectivement 381 et 61 unités, marqueurs `UNK` et `EOS` compris. La validation TSAC comporte 27 caractères inconnus sur 70 150 ; MADAR en comporte 2 sur 25 497. Les dénominateurs de perplexité, incluant EOS, sont 71 924 et 26 297.

Les apprentissages par comptage durent localement de 0,10 à 0,24 seconde par modèle ; ces durées ne sont ni des temps d'époque neuronale ni des latences de l'application. L'empreinte du protocole enregistrée par le run désigne la version avant cet ajout du bilan.

Le [rapport agrégé généré](../reports/character_lm_v1_2026-11-04/README.md) et son `summary.json` contiennent les valeurs recalculées et leur provenance. Aucun vocabulaire, n-gramme ou texte source n'est publié dans ce rapport. Cette référence n'est pas ajoutée au sélecteur de classification de l'application, car elle ne retourne pas de label de tâche.

### Commandes de reproduction

Choisir de nouveaux noms de dossiers lors d'une relance ; les dossiers existants ne sont jamais écrasés.

```sh
sh scripts/project_uv.sh run --locked --offline python -m unittest discover -s tests -p test_character_lm.py
sh scripts/project_uv.sh run --locked --offline python scripts/run_character_lm.py --mode smoke --output outputs/language_models/NOUVEAU_SMOKE
sh scripts/project_uv.sh run --locked --offline python scripts/verify_character_lm.py outputs/language_models/NOUVEAU_SMOKE
sh scripts/project_uv.sh run --locked --offline python scripts/run_character_lm.py --mode full --smoke-run outputs/language_models/NOUVEAU_SMOKE --output outputs/language_models/NOUVEAU_FULL
sh scripts/project_uv.sh run --locked --offline python scripts/verify_character_lm.py outputs/language_models/NOUVEAU_FULL --report reports/NOUVEAU_RAPPORT_LM
```

Nous avons réalisé et vérifié la référence statistique de langage initiale. La BiGRU reste à réaliser après démarrage de sa phase neuronale ; les autres preuves ouvertes de L04/L07/L12 ne sont pas clôturées par ce résultat.
