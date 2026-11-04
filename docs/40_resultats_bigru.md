# Premières expériences BiGRU : résultats vérifiés sur validation

**Date :** 4 novembre 2026. **L04, L07, L10 et L12 : contributions partielles.** Le protocole et la configuration ont été fixés avant les mesures : [protocole BiGRU](38_protocole_bigru.md), `configs/bigru_v1.json`. Nous n'avons modifié aucun hyperparamètre après lecture des scores.

## Exécutions et vérifications

Nous avons d'abord exécuté cinq tests d'entrée sur données fictives, puis huit tests neuronaux. Les 13 tests réussissent. Ils couvrent notamment le vocabulaire train-only, le padding et la direction inverse, les gradients et mises à jour, la répétabilité d'une courte exécution CPU, le rechargement et le rejet de métriques altérées.

Le smoke réel `2026-11-04_bigru_v1_cpu_smoke` a entraîné deux modèles sur 100 lignes de train et 100 de validation par corpus, deux époques et graine 42. Les pertes sont finies, les gradients non nuls et les poids modifiés. Les checkpoints ont été rechargés et les prédictions rejouées avant le full. Les scores smoke ne constituent pas les résultats scientifiques finaux.

Le full `outputs/neural_baselines/2026-11-04_bigru_v1_cpu_full/` utilise tout le train v2 et toute la validation, avec trois graines par tâche. Les six checkpoints retenus ont été rechargés ; les vocabulaires et audits ont été reconstruits depuis le train ; chaque prédiction de validation, ses scores et ses métadonnées ont été rejoués. Les matrices et macro-F1 ont également été recalculés directement par comptage.

Le vérificateur a contrôlé les mêmes identifiants, groupes et labels de validation que le run TF-IDF caractères brut `outputs/baselines/2026-10-17_grouped_v2_full/`. Ses métriques ont été recalculées depuis les prédictions sauvegardées. Les données, les sources et les manifestes sont inchangés ; aucune ligne du test officiel n'a été sélectionnée pour cette expérience.

## Résultats individuels

| Corpus | Graine | Époques exécutées | Époque retenue | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 13 | 12 | 12 | 0,847140 | 0,847238 |
| TSAC | 42 | 12 | 9 | 0,855657 | 0,855693 |
| TSAC | 2026 | 9 | 6 | 0,856806 | 0,856821 |
| MADAR | 13 | 12 | 9 | 0,904381 | 0,911250 |
| MADAR | 42 | 12 | 9 | 0,895672 | 0,901250 |
| MADAR | 2026 | 9 | 6 | 0,899366 | 0,906250 |

La graine 2026 s'arrête après trois époques sans gain depuis l'époque six. Pour les autres runs, le budget maximal est atteint. Un modèle retenu à l'époque neuf n'est pas remplacé par celui de la douzième époque si cette dernière a un macro-F1 inférieur.

## Moyenne et comparaison

| Corpus | BiGRU : macro-F1 moyen ± écart-type d'échantillon | TF-IDF caractères brut, référence existante |
| --- | ---: | ---: |
| TSAC | **0,853201 ± 0,005281** | 0,928898 |
| MADAR | **0,899806 ± 0,004371** | 0,949244 |

Nous présentons les trois graines, sans sélectionner la meilleure comme résultat du système. Les écarts-types concernent l'aléa d'apprentissage sur un seul partitionnement ; ils ne représentent pas un intervalle de confiance sur la population.

Dans ce protocole, la BiGRU initiale reste inférieure à la référence TF-IDF caractères. Nous conservons le résultat et la référence classique forte. Cela ne prouve pas que toute architecture récurrente serait inférieure : la comparaison change simultanément représentation, optimisation et classifieur, avec un seul réglage initial. Aucun bootstrap ni résultat de test ne permet ici une conclusion générale de significativité.

## Courbes et limites observées

Le dossier privé `outputs/neural_baselines/2026-11-04_bigru_v1_cpu_full/figures/` contient six PNG, un par graine et corpus. Chaque image montre les pertes train/validation, le macro-F1 de validation avec l'époque retenue et la matrice de confusion du checkpoint sélectionné. Les six images ont été inspectées et leurs empreintes vérifiées contre `provenance.json` et le `results.json` source.

Sur TSAC graine 42, la perte validation atteint environ 0,346 à l'époque cinq puis monte à environ 0,505 à l'époque douze, tandis que la perte train continue à baisser. Cette divergence est compatible avec du surapprentissage. Le macro-F1 retenu est celui de l'époque neuf : le critère de sélection préétabli reste le macro-F1, pas la perte minimale. La perte tient compte des scores probabilistes et peut évoluer différemment du label décidé.

Le vocabulaire de mots laisse environ 29,49 % des tokens de validation TSAC et 21,76 % de ceux de MADAR hors des unités retenues avant troncature. La variation orthographique et les unités rares peuvent limiter cette première représentation. Il s'agit d'une hypothèse compatible avec l'audit, pas d'une cause démontrée pour chaque erreur. Une erreur peut aussi venir du contexte, de la cible, des labels ou du domaine.

Le maximum de 128 tokens tronque un texte de validation TSAC et aucun texte MADAR. Les textes concernés restent dans les scores. Les matrices, métriques par classe et par écriture, taux UNK et troncatures restent disponibles par graine ; les petites catégories ne doivent pas être généralisées sans leurs effectifs.

## Coût et artefacts

- TSAC : **479 106 paramètres** entraînables, vocabulaire PAD/UNK compris de 6 702 entrées ; checkpoints d'environ 2,06 Mo.
- MADAR : **307 331 paramètres** entraînables, vocabulaire de 4 016 entrées ; checkpoints d'environ 1,33 Mo.
- macOS ARM64, Python 3.12.15, PyTorch 2.15.0, CPU deux threads, FP32 ; aucun GPU utilisé.
- Boucles d'apprentissage et validation : environ **151 secondes cumulées** pour les six runs, hors installation, smoke, vérification, export et rédaction. Les modèles TSAC prennent environ 30 à 41 secondes, MADAR environ 11 à 14 secondes, selon le nombre d'époques.
- Mémoire maximale cumulative du processus : **450 510 848 octets** ; ce n'est ni une mesure GPU ni une mesure isolée par modèle.

Chaque modèle possède un checkpoint `.pt`, un historique par époque et un fichier de prédictions. Le run contient un `results.json` avec versions, empreintes du code/protocole/configuration et données, graine, temps et mémoire. Les PNG et leur provenance s'ajoutent dans `figures/`. Tous ces artefacts individuels restent ignorés par Git.

Le [rapport agrégé généré](../reports/bigru_v1_2026-11-04/README.md) et son `summary.json` fournissent les résultats partageables. L'environnement et les commandes sont documentés dans [le profil BiGRU](39_environnement_bigru.md). Les verrous classique et applicatif n'ont pas été modifiés. La BiGRU n'est pas ajoutée automatiquement au backend applicatif.

## Ce qui est réalisé et ce qui reste ouvert

La référence BiGRU initiale est implémentée, entraînée sur trois graines, sauvegardée et vérifiée. Les composants de référence de L04 disposent maintenant de preuves pour règles/TF-IDF, modèle de langage statistique et BiGRU ; cela ne clôture pas automatiquement J2 ou les livrables globaux.

Restent notamment la relecture humaine des analyses linguistiques et d'erreurs, les cartes et preuves de données encore ouvertes, l'analyse d'erreurs propre aux composants neuronaux, la vérification Kaggle/MPS/CUDA, l'encodeur gelé et adapté, le LLM/LoRA, les ablations finales, le test figé et la relance indépendante. Aucun téléchargement ou apprentissage de modèle préentraîné n'est effectué par cette étape.
