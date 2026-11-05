# CAMeLBERT gelé : résultats et faisabilité locale

**4 novembre 2026 — contributions partielles à L05, L07 et L12.** Nous avons réalisé la référence « encodeur gelé + tête apprise » prévue au cahier. Ce résultat ne clôture pas L05 : le fine-tuning complet reste à réaliser. Le [protocole préalable](44_protocole_encodeur_gele.md), la configuration et les hyperparamètres n'ont pas été modifiés après observation des scores.

## Expérience réalisée

Le checkpoint général CAMeLBERT Mix, révision `9be352797bdf28a9ae21e2ae582aaaca7abdb22d`, a été téléchargé explicitement dans le cache ignoré, puis contrôlé par taille et SHA-256 officiel. Le backbone réellement chargé compte **108 490 752 paramètres**, sans pooler ; aucun paramètre de celui-ci n'est entraînable. Le chargement n'a aucune clé manquante du backbone ; les têtes de préentraînement et le pooler sont les seules clés ignorées.

Les textes bruts sont encodés une fois, longueur maximale 128, moyenne des états masquée par attention_mask, CLS/SEP inclus. Seule une tête linéaire apprend : **1 538 paramètres** pour TSAC et **2 307** pour MADAR. Elle utilise tous les exemples du train, séparément pour les graines 13, 42 et 2026. Les vecteurs sont réutilisés ; une époque désigne ici une époque de la tête, pas un nouvel entraînement de BERT.

Avant le run complet, un essai de 100 lignes par partition et tâche, deux époques et graine 42 a été exécuté puis rechargé. Le run complet est `outputs/frozen_encoder/2026-11-04_camelbert_frozen_mps_full/`. Le [rapport agrégé généré](../reports/camelbert_frozen_v1_2026-11-04/README.md) et son JSON fournissent les résultats numériques, métriques par classe et écriture et provenance. Les poids, vecteurs et prédictions individuelles restent privés.

## Scores de validation

| Corpus | Graine | Accuracy | Macro-F1 | Époque retenue / exécutées |
| --- | ---: | ---: | ---: | ---: |
| TSAC | 13 | 0,841601 | 0,841001 | 16 / 21 |
| TSAC | 42 | 0,848365 | 0,847584 | 25 / 30 |
| TSAC | 2026 | 0,846110 | 0,845241 | 23 / 28 |
| MADAR | 13 | 0,923750 | 0,919112 | 3 / 8 |
| MADAR | 42 | 0,925000 | 0,920714 | 7 / 12 |
| MADAR | 2026 | 0,930000 | 0,925954 | 14 / 19 |

L'époque conservée maximise le macro-F1 de validation ; les cinq époques sans amélioration suivantes déclenchent l'arrêt. Nous rapportons les trois graines, sans choisir celle donnant le meilleur score.

| Méthode sur la même validation v2 | TSAC macro-F1 | MADAR macro-F1 |
| --- | ---: | ---: |
| TF-IDF caractères brut, référence fixée | **0,928898** | **0,949244** |
| BiGRU, moyenne ± écart-type sur trois graines | 0,853201 ± 0,005281 | 0,899806 ± 0,004371 |
| CAMeLBERT gelé, moyenne ± écart-type sur trois graines | 0,844608 ± 0,003337 | 0,921926 ± 0,003579 |

L'encodeur gelé est inférieur à TF-IDF caractères sur les deux tâches dans ce protocole. Il est inférieur à la BiGRU sur TSAC et supérieur sur MADAR. Ces écarts sont descriptifs sur la validation : ils ne constituent pas un test de significativité ni un résultat final sur le test.

Pour TSAC, le macro-F1 par graine est environ 0,884–0,892 sur l'écriture AR et 0,789–0,796 sur LATIN. Ce constat est cohérent avec l'intérêt de l'audit Arabizi/code-switching, sans établir que la fragmentation cause l'écart : les textes et leurs difficultés diffèrent. MIXED (46 textes) et UNKNOWN (11) restent de petits sous-ensembles, sans conclusion générale à partir d'un score élevé. Les catégories d'écriture ne constituent pas des labels de langue.

## Ressources réellement observées sur le Mac

Python 3.12.15, PyTorch 2.15.0 et Transformers 4.57.6 ; encodeur **MPS**, têtes **CPU**, float32, batch physique 16 et padding dynamique. Le profil isolé est `environments/encoder/`, accessible via `scripts/encoder_uv.sh` ; les verrous antérieurs sont conservés.

| Étape d'encodage | Textes | Durée | Débit observé |
| --- | ---: | ---: | ---: |
| TSAC train | 10 053 | 47,45 s | 211,84 textes/s |
| TSAC validation | 1 774 | 9,41 s | 188,43 textes/s |
| MADAR train | 6 388 | 19,41 s | 329,05 textes/s |
| MADAR validation | 800 | 2,29 s | 349,49 textes/s |

Le run complet prend **85,98 secondes**, chargement, encodage et six apprentissages de têtes compris ; téléchargement initial et contrôles/export ultérieurs sont exclus. Les six apprentissages de têtes totalisent environ 5,22 s. L'essai préalable prend 5,9 s.

Le pic RSS du processus est **1 058 848 768 octets**. La mesure MPS en fin de run indique 434 085 888 octets alloués aux tenseurs et 1 091 174 400 octets alloués par le driver. Ces valeurs MPS sont des instantanés, **pas un pic GPU mesuré** ; la mémoire partagée Apple ne se déduit pas en additionnant RSS et allocations MPS. Le batch 16 a fonctionné, y compris avec des entrées tronquées à 128 ; nous n'avons pas recherché le batch maximal.

**Décision : la référence gelée peut être exécutée localement.** Cette mesure ne garantit pas la mémoire nécessaire au fine-tuning, qui conserve aussi gradients et états de l'optimiseur. Aucun passage Kaggle/CUDA n'est revendiqué ; le verrou Linux contient des dépendances CUDA 13 dont la compatibilité avec le driver effectif devra être vérifiée. Le budget de la future phase ne peut pas être extrapolé directement de cet essai sans gradients.

## Preuves et limites de vérification

- Les six têtes sauvegardées ont été rechargées avec `weights_only=True` ; leurs prédictions et métriques sont recalculées sur toute la validation (1 774 / 800).
- Les identifiants, groupes et labels de validation sont alignés avec la référence TF-IDF ; les manifestes et textes sources sont contrôlés par empreintes. Le loader conserve ses gardes excluant le test.
- Dix textes de validation par tâche sont réencodés depuis le checkpoint original et comparés aux vecteurs à tolérance absolue 1e-4. L'empreinte du backbone avant/après est identique et aucun gradient n'est présent sur ses poids.
- Sept tests sur données inventées et un petit BERT aléatoire couvrent pooling, padding, troncature, gel, apprentissage/rechargement et garde avant FULL. Cinq tests vérifient le bridge du notebook et ses modes explicites.
- Six PNG du run montrent courbes et matrices. Un premier export a rencontré la garde MPS dans un contexte sans GPU ; le vérificateur a été corrigé pour permettre le rejeu des seules têtes sur CPU. Le run d'entraînement est conservé, sans réentraînement ni altération des scores. Les empreintes de code historiques du run et celle du vérificateur actuel restent distinctes dans les preuves.
- Le notebook05 a été exécuté dans un vrai noyau Jupyter en **replay** : quatre cellules de code et six images affichées. L'entraînement FULL a été exécuté par le script ; nous ne présentons pas ce replay comme un second entraînement via Jupyter. Preuve privée : `outputs/notebook_checks/2026-11-04/frozen_encoder_native_replay.json`.

Les JSON de résultats et vérification donnent les détails ; la documentation ne remplace pas ces preuves. L'inférence dédiée à ce nouvel encodeur, la revue de ses erreurs, les model cards finales, la reproduction indépendante et le test final restent ouverts. Les modèles servis par l'application n'ont pas été remplacés.

## Reproduire ou présenter

Le [notebook05](../notebooks/05_encodeur_gele.ipynb) peut conserver le noyau **TuniScope (notebooks)** : les calculs sont délégués à l'environnement séparé. Son mode `full` par défaut lance un petit contrôle puis le vrai run complet et les PNG. Pour afficher uniquement le résultat existant, choisir `TRAINING_MODE = 'replay'`. Les modes `smoke` et `off` sont distincts et ne remplacent jamais silencieusement FULL.

Depuis la racine du dépôt, relecture et export séparés :

```sh
sh scripts/encoder_uv.sh run --locked --offline python scripts/verify_frozen_encoder.py outputs/frozen_encoder/2026-11-04_camelbert_frozen_mps_full
```

```sh
sh scripts/encoder_uv.sh run --locked --offline python scripts/plot_frozen_encoder.py outputs/frozen_encoder/UN_NOUVEAU_RUN_FULL
```

L'export ne remplace pas un dossier `figures/` existant. Les images déjà produites sont accessibles directement dans le sous-dossier du run. Le [protocole](44_protocole_encodeur_gele.md) décrit les précautions nécessaires à la future comparaison gelé/adapté.
