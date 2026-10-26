# Relecture des métriques et reproduction locale des résultats de validation

**Date :** 25 octobre 2026. **Contributions partielles :** L04, L07 et L12. Les tableaux agrégés soutiennent également la présentation d'avancement L11. Nous ne rapportons ici ni nouvelle méthode ni évaluation sur le test officiel.

## Objectif et périmètre

Nous rendons les résultats consultables indépendamment du résumé du notebook. Le lecteur `scripts/show_validation_results.py` recalcule les métriques depuis les prédictions sauvegardées et les compare aux valeurs du run. Avec `--verify`, il appelle d'abord les vérificateurs existants, qui rechargent les modèles locaux et reproduisent leurs prédictions de validation.

Le mode `--run` lit un run TF-IDF v2 local, complet ou smoke, et affiche accuracy, balanced accuracy, macro-F1, précision/rappel/F1 par classe et matrice de confusion. Le mode smoke est explicitement signalé comme un test technique. Ce lecteur n'effectue aucun nouvel apprentissage.

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/show_validation_results.py --verify --run outputs/baselines/notebooks/d99058b90e8447fa9cc676ffe1a472d3_characters_full
```

La passerelle du notebook enregistre le chemin de chaque nouveau run dans `runs`. Pour relire un autre entraînement, remplacer le chemin ci-dessus par celui indiqué par son exécution, sans choisir silencieusement le dossier le plus récent.

## Relance complète observée et vérifiée

Le notebook sauvegardé référence un nouveau smoke puis un full caractères sous le préfixe `d99058b90e8447fa9cc676ffe1a472d3`. Le full a commencé le `2026-10-25T17:25:03.146141+00:00`, avec Python 3.12.15, CPU, manifestes v2, mêmes configurations et verrouillage. Ses métadonnées référencent le commit `2d6900a9fc5d7c2808ea7fb76b4d3b07a3942668` et un état de travail non commité. Le rechargement des six systèmes du full et du smoke a été vérifié séparément. Les sorties du notebook et ses modifications locales ont été préservées.

| Modèle caractères brut | Validation | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| TSAC | 1 774 | 0,9289740699 | 0,9288980670 |
| MADAR | 800 | 0,9512500000 | 0,9492441926 |

Ces résultats reproduisent la référence complète historique, sans changement des exemples, labels, partitions ou paramètres. Une même accuracy peut coexister avec un macro-F1 différent : sur TSAC normalisé, le nombre total d'erreurs reste 126 mais leur répartition change et le macro-F1 devient 0,9288927289.

La preuve porte sur cette exécution locale sauvegardée et sa vérification. Elle n'établit pas une relance indépendante dans une installation vierge, une exécution Kaggle ou la reproductibilité de toutes les familles via l'interface notebook.

## Comparaison agrégée et captures

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/show_validation_results.py --verify --output outputs/presentation_validation_replay_02
```

Le dossier de destination doit être nouveau et situé sous `outputs/`. Cette commande relit les cinq références historiques fixées dans le script : caractères, mots, combinaison, règles et augmentation caractères/règles. Elle ne remplace pas ces références par un nouveau run du notebook. Elle produit trois HTML autonomes et `summary.json`, avec valeurs numériques, empreintes des résultats sources, empreinte du verrouillage, date et statut de vérification. Sans `--verify`, les pages annoncent seulement le recalcul des métriques sauvegardées, pas le rechargement des modèles.

La génération vérifiée du 25 octobre est conservée sous `outputs/presentation_validation_2026-10-25/` :

- `01_comparaison.html` : accuracy et macro-F1 de cinq systèmes sur les deux tâches ;
- `02_classes_confusions.html` : matrices et métriques par classe du modèle caractères brut ;
- `03_ablations.html` : normalisation et ajout lexical ;
- trois captures PNG du rendu dans le navigateur, inspectées visuellement.

Les pages ne contiennent ni texte de corpus, ni vocabulaire appris, ni identifiant de ligne privée. Elles peuvent être ouvertes directement dans un navigateur, sans serveur. Les PNG sont des captures de ces pages, non des métriques saisies dans une image. Les modèles et prédictions détaillées restent dans les répertoires ignorés par Git.

## Contrôles et limites

Huit tests synthétiques ciblés vérifient la sélection non ambiguë d'une variante, le recalcul malgré les champs de slices supplémentaires, le refus d'une accuracy incohérente, l'échappement HTML, les axes des matrices, l'absence de fausse annonce de rechargement et le refus d'un run annonçant un accès test. Les cinq familles historiques ont passé leurs vérificateurs ; le nouveau full et son smoke ont également été rechargés.

Une première génération a été arrêtée parce que le lecteur comparait tout le dictionnaire historique, comprenant aussi les slices par écriture, aux seules métriques globales recalculées. La comparaison a été corrigée pour contrôler chaque métrique globale, sans assimiler les slices supplémentaires à une incohérence. Aucun résultat modèle n'a été modifié pour résoudre cet incident.

Nous ne revendiquons pas une nouvelle réussite de toute la suite de tests : les contrôles structurels des notebooks attendent des fichiers sans sorties, tandis que la copie de travail conserve volontairement l'exécution interactive. Avant partage public, la préparation d'une copie sans sorties et la vérification native/cloud restent à effectuer explicitement. Le test final, les expériences à trois graines et le bootstrap ne sont pas couverts par cette relecture.
