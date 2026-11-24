# Audit Gold — Robustesse et incertitude sur la validation

**22 novembre 2026 · L07, L10 et L12.** Ce complément à l’[étude de la quantité de données et de l’abstention](76_avancement_gold_validation.md) réunit un essai de robustesse contrôlé, une comparaison avec intervalle d’incertitude et un inventaire des erreurs à examiner.

## Robustesse de deux classifieurs sur TSAC

Nous rechargeons le TF-IDF caractères brut et CAMeLBERT confirmé, graine 42, déjà retenue pour la démonstration. Les deux checkpoints reproduisent leurs 1 774 prédictions originales. Nous calculons ensuite de nouvelles prédictions sur trois variantes des mêmes avis. Pour chaque variante, le texte original et sa version modifiée partagent le label TSAC de référence. La duplication de lettre simule une faute mécanique. Elle peut modifier la lecture de certains mots et ne représente pas toute la diversité des graphies réelles.

| Modification | Avis modifiés | TF-IDF · décisions changées | TF-IDF · écart de macro-F1 | CAMeLBERT · décisions changées | CAMeLBERT · écart de macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalisation NFC et espaces | 144 | 0 | 0,00 pt | 0 | 0,00 pt |
| Triplement des espaces ordinaires | 1 639 | 0 | 0,00 pt | 0 | 0,00 pt |
| Duplication d’une lettre dans un mot | 1 645 | 40 | −1,12 pt | 83 | −1,64 pt |

Les écarts sont calculés sur les 1 774 avis, y compris ceux que la règle n’a pas modifiés. La duplication de lettre introduit 30 erreurs et en corrige 10 pour TF-IDF, contre 56 et 27 pour CAMeLBERT. La baisse plus forte observée pour cette graine de CAMeLBERT décrit ce perturbateur précis et cette validation.

### Résultats par écriture du texte original

| Écriture détectée | Avis | Macro-F1 TF-IDF | Macro-F1 CAMeLBERT, graine 42 |
| --- | ---: | ---: | ---: |
| AR · arabe | 785 | 92,13 % | 95,38 % |
| LATIN · latin | 932 | 93,84 % | 90,50 % |
| MIXED · mixte | 46 | 86,73 % | 95,65 % |
| UNKNOWN · indéterminée | 11 | 61,18 % | 100,00 % |

Le rang des deux modèles s’inverse entre les deux catégories les plus représentées. Ces catégories décrivent des systèmes d’écriture. LATIN peut inclure de l’Arabizi et des textes en français. Les sous-ensembles MIXED et UNKNOWN sont petits. Les détails par longueur et par écriture pour chaque variante sont conservés dans la sortie JSON.

## Écart entre CAMeLBERT et TF-IDF sur la validation

Nous effectuons 1 000 rééchantillonnages par tâche, avec la graine 42 et le même tirage de groupes pour les deux systèmes. Un groupe MADAR conserve ses phrases parallèles ensemble. À chaque tirage, nous calculons la moyenne des macro-F1 des trois modèles CAMeLBERT confirmés, puis nous soustrayons le macro-F1 de TF-IDF. Les percentiles 2,5 et 97,5 de ces écarts donnent l’intervalle.

| Corpus | Groupes | Écart observé | Intervalle à 95 % |
| --- | ---: | ---: | ---: |
| TSAC | 1 721 | +0,19 pt | [−1,05 ; +1,31] pts |
| MADAR | 200 | +0,42 pt | [−1,12 ; +2,06] pts |

Les deux intervalles couvrent zéro. Les observations ne suffisent donc pas à établir un gain sur cette comparaison. Ce rejeu mesure l’incertitude liée aux exemples de validation pour le contraste complémentaire CAMeLBERT/TF-IDF. Le contraste principal du protocole final porte sur Qwen avec et sans LoRA. Ses résultats seront calculés à partir des prédictions individuelles du test final.

## Cas préparés pour la revue d’erreurs

Nous avons produit dix lots à partir des prédictions de validation vérifiées de TF-IDF caractères, BiGRU et CAMeLBERT (graine 42), ainsi que de Qwen de base avec et sans exemples dans la consigne. Chaque lot contient au plus 30 cas, avec un seul représentant par groupe. La file privée rassemble **298 entrées**. Un même cas peut figurer dans plusieurs lots. Les 800 sorties non conformes de Qwen de base sans exemple sur MADAR sont comptées comme erreurs de format et séparées des erreurs sur un label valide. Les prédictions individuelles CUDA historiques de LoRA ne sont pas disponibles dans les archives reçues. Les revues linguistiques déjà effectuées pour TF-IDF et CAMeLBERT sont documentées dans la [note précédente](76_avancement_gold_validation.md).

## Preuves et suite

- [Robustesse et détails par catégorie](../reports/gold_audit_2026-11-22/robustness.json), [bootstrap et inventaire agrégés](../reports/gold_audit_2026-11-22/validation_audit.json).
- [Graphique de robustesse](../reports/gold_audit_2026-11-22/figures/robustesse_tsac.png), [graphique du bootstrap](../reports/gold_audit_2026-11-22/figures/bootstrap_validation.png).
- Les prédictions de robustesse et les agrégats ont pris **65,53 s** sur CPU. Le rejeu du bootstrap et de l’inventaire a donné des fichiers identiques. Des contrôles synthétiques vérifient la conservation des groupes, le traitement de la classe de prédiction invalide et les comptes de changement de décision.

```sh
sh scripts/encoder_uv.sh run --locked --offline python scripts/run_gold_robustness.py --device cpu --output outputs/gold_audit_reproduction
sh scripts/project_uv.sh run --locked --offline python scripts/audit_gold_saved_predictions.py --output outputs/gold_audit_reproduction/validation_audit.json --private-output outputs/gold_audit_reproduction/review_queue.json
sh scripts/encoder_uv.sh run --locked --offline python scripts/plot_gold_audit.py outputs/gold_audit_reproduction --output outputs/gold_audit_reproduction/figures
```

Le travail complète l’audit de validation. Les jeux de défi indépendants, la revue qualitative restante, l’évaluation officielle, le bootstrap du contraste principal et le rapport complet constituent les prochaines étapes de Gold.
