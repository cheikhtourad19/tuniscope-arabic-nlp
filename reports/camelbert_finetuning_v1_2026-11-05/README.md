# CAMeLBERT CLS : référence gelée et fine-tuning

Résultats **de validation v2** générés depuis un run FULL vérifié. Aucun test officiel évalué.
L05/L07/L12 partiels : runs initiaux trois graines vérifiés ; configurations finales, analyses et test restent distincts.

| Corpus | Régime | Graine | Accuracy | Macro-F1 | Époque retenue / exécutées |
| --- | --- | ---: | ---: | ---: | ---: |
| TSAC | frozen | 13 | 0.612740 | 0.605210 | 3 / 3 |
| TSAC | finetuned | 13 | 0.938557 | 0.938435 | 3 / 3 |
| TSAC | frozen | 42 | 0.631905 | 0.609937 | 3 / 3 |
| TSAC | finetuned | 42 | 0.931229 | 0.930665 | 3 / 3 |
| TSAC | frozen | 2026 | 0.615558 | 0.602122 | 3 / 3 |
| TSAC | finetuned | 2026 | 0.938557 | 0.938307 | 3 / 3 |
| MADAR | frozen | 13 | 0.612500 | 0.457151 | 3 / 3 |
| MADAR | finetuned | 13 | 0.956250 | 0.953916 | 2 / 3 |
| MADAR | frozen | 42 | 0.636250 | 0.521718 | 3 / 3 |
| MADAR | finetuned | 42 | 0.952500 | 0.950389 | 2 / 3 |
| MADAR | frozen | 2026 | 0.611250 | 0.469142 | 3 / 3 |
| MADAR | finetuned | 2026 | 0.950000 | 0.948141 | 3 / 3 |

| Corpus | Méthode | Macro-F1 |
| --- | --- | ---: |
| TSAC | TF-IDF caractères brut | 0.928898 |
| TSAC | BiGRU, moyenne3graines | 0.853201 |
| TSAC | Gelé mean-pooling, protocole antérieur | 0.844608 ± 0.003337 |
| TSAC | CLS frozen | 0.605756 ± 0.003936 |
| TSAC | CLS finetuned | 0.935802 ± 0.004449 |
| MADAR | TF-IDF caractères brut | 0.949244 |
| MADAR | BiGRU, moyenne3graines | 0.899806 |
| MADAR | Gelé mean-pooling, protocole antérieur | 0.921926 ± 0.003579 |
| MADAR | CLS frozen | 0.482670 ± 0.034343 |
| MADAR | CLS finetuned | 0.950815 ± 0.002911 |

Durée réelle du run : **62.62 min** ; appareil **mps**. Hors installation, smoke et contrôles/export ultérieurs.
Écart-type échantillonnal ; aucune meilleure graine sélectionnée. Les trois graines ne sont pas trois tests indépendants.
Le témoin CLS partage initialisation, architecture, ordre de données et hyperparamètres avec l’adapté ; la référence mean-pooling suit une autre architecture et un budget différent.
Le faible learning rate et le budget court du témoin CLS peuvent limiter sa tête ; ne pas présenter ce résultat comme le meilleur encodeur gelé possible.
MPS utilise un avertissement pour les kernels sans déterminisme strict ; pas de répétabilité bit à bit promise.
Aucun bootstrap/test final, essai CUDA, remplacement applicatif ou expérience LoRA n’est établi par ce run.

Run : `outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full`. Poids, exemples et prédictions individuelles restent privés.

**Limite du journal initial :** scores par classe et prédictions conservés pour chaque checkpoint retenu ; les autres époques conservent pertes et macro-F1, sans détail par classe archivé. Le producteur corrigé ajoute désormais métriques et prédictions à chaque époque. Aucun détail manquant du run initial n’est inventé, aucun poids/résultat initial n’est remplacé. Les sources initiales sont archivées par empreinte dans le dossier privé source_snapshots.
