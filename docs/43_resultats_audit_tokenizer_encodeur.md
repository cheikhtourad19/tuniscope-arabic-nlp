# Audit CAMeLBERT : résultats et configuration d'entrée initiale

**Date :** 4 novembre 2026. **Contributions partielles à L03, L05 et L12.** Le [protocole](42_protocole_audit_tokenizer_encodeur.md) et `configs/camelbert_tokenizer_audit_v1.json` ont été fixés avant les mesures. Aucun poids préentraîné n'a été téléchargé et aucun modèle de classification n'a été entraîné.

## Sources et environnement vérifiés

Nous utilisons le tokenizer de `CAMeL-Lab/bert-base-arabic-camelbert-mix`, révision **`9be352797bdf28a9ae21e2ae582aaaca7abdb22d`**, fiche déclarant Apache-2.0. Cinq fichiers publics texte/configuration (316 515 octets) sont conservés dans le cache privé avec tailles et SHA-256. Les poids `.bin`, `.h5` et `.msgpack` visibles dans le dépôt distant ne sont pas récupérés.

Le profil `environments/tokenizer/` contient Transformers **4.57.6**, tokenizers **0.22.2**, huggingface-hub **0.36.2**, Python **3.12.15**. Le chargement utilise uniquement les fichiers locaux, sans code distant. Le vocabulaire compte 30 000 unités. Les environnements et verrouillages classique, BiGRU et application restent inchangés.

L'absence de PyTorch/TensorFlow/Flax dans ce profil est intentionnelle : il sert au tokenizer et aux configurations. Un message de Transformers signalant l'absence de moteurs de modèles ne constitue pas un échec de tokenisation. Aucun moteur n'est ajouté pour masquer ce message.

## Périmètre et vérification

L'audit porte sur les **19 015 textes** du train et de la validation v2 : TSAC 10 053/1 774, MADAR 6 388/800. Les groupes train et validation sont disjoints selon les manifestes. Les catégories d'écriture sont calculées par présence Unicode et ne constituent pas une annotation de langue.

Neuf tests de profil tokenizer réussissent sur un vocabulaire et des textes inventés : frontières, inconnus, dénominateurs, longueurs sans troncature, conservation des petits groupes, sélection sur train seulement, comptage des paramètres, absence d'export de textes de corpus et liste de téléchargement sans poids. Quatre tests supplémentaires du notebook vérifient notamment le mode off sans chargement et le profil enfant hors ligne.

Le run `outputs/tokenizer_audits/2026-11-04_camelbert_raw_v2/` a été rejoué intégralement ; les statistiques et la décision sont identiques. Le [rapport agrégé généré](../reports/camelbert_tokenizer_v1_2026-11-04/README.md) et son `summary.json` ne contiennent aucun texte ou découpage individuel de corpus. Les seuls textes visibles dans ce JSON sont des probes inventés et marqués comme tels.

Le notebook `04_tokenizer_encodeur.ipynb` a également été exécuté dans le noyau réel `TuniScope (notebooks)`. Ses trois cellules de code ont produit un nouvel audit et son contrôle hors ligne, avec les mêmes statistiques. Preuve privée : `outputs/notebook_checks/2026-11-04/encoder_tokenizer_native.json`.

## Découpage et inconnus sur validation

Les ratios sont calculés sur les tokens hors CLS/SEP. Les longueurs incluent ces deux marqueurs. Le dénominateur « champ » désigne une unité séparée par espaces, pas un mot annoté morphologiquement.

| Corpus | Écriture | Textes | Sous-mots/champ | Tokens UNK | P99 longueur | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | AR | 785 | 1,57 | 0,69 % | 94 | 176 |
| TSAC | LATIN | 932 | 3,89 | 0,06 % | 81 | 140 |
| TSAC | MIXED | 46 | 2,05 | 2,12 % | 54 | 54 |
| TSAC | UNKNOWN | 11 | 4,18 | 2,17 % | 14 | 14 |
| MADAR | AR | 800 | 1,53 | 0,77 % | 29 | 48 |

Le découpage est plus fragmenté sur l'écriture latine de TSAC que sur son écriture arabe, selon ce proxy. Les tokens UNK sont rares dans la catégorie LATIN, mais cela ne démontre pas une bonne représentation sémantique du français ou de l'Arabizi.

Globalement, TSAC validation contient 114 tokens UNK sur 30 254 tokens de contenu (0,377 %), répartis dans 49 textes sur 1 774. MADAR validation contient 60 UNK sur 7 755 tokens (0,774 %), dans 44 textes sur 800. Le taux par token et le taux de textes concernés sont des mesures différentes.

Le seuil de 100 textes est dépassé pour les grands groupes du train : TSAC AR 4 426, LATIN 5 355, MIXED 259, MADAR AR 6 387. TSAC UNKNOWN n'a que 13 textes train et 11 validation ; MADAR MIXED compte une seule ligne train. Nous les conservons et déclarons leur rareté, sans prétendre les avoir audités sur 100 exemples indépendants.

## Choix de longueur : 128 sous-mots

La règle préétablie choisit **128 tokens, CLS/SEP compris** : c'est la plus petite candidate qui conserve entièrement au moins 99 % de chacun des quatre groupes train suffisamment représentés. À 64, les taux de textes tronqués TSAC AR/LATIN/MIXED sont environ 1,74/2,82/2,70 % et dépassent le seuil. À 128, ils sont environ 0,294/0,336/0,772 % et le satisfont.

| Corpus | Partition | Textes qui dépassent 128 | Taux | Sous-mots qui seraient retirés |
| --- | --- | ---: | ---: | ---: |
| TSAC | Train | 33 / 10 053 | 0,328 % | 1 772 |
| TSAC | Validation | 4 / 1 774 | 0,225 % | 105 |
| MADAR | Train | 0 / 6 388 | 0 % | 0 |
| MADAR | Validation | 0 / 800 | 0 % | 0 |

Les longueurs sont mesurées sans troncature durant l'audit. Ce tableau décrit la future règle d'entrée : les textes concernés resteront dans les scores et seul leur contenu au-delà de la fenêtre sera tronqué lors de l'encodage. Nous conservons les textes bruts inchangés. L'adéquation de 128 au budget mémoire/débit devra encore être mesurée avec un modèle effectivement chargé.

Le « 128 » de la BiGRU désigne des tokens mots/symboles d'un autre tokenizer : ce n'est pas la même fenêtre linguistique et les nombres de troncatures ne sont pas directement interchangeables.

## Probes inventés et normaliseur interne

Le normaliseur est `BertNormalizer`, `clean_text=True`, `handle_chinese_chars=True`, `lowercase=False`, `strip_accents=None`. Ces réglages font partie du tokenizer préentraîné ; ils sont distincts de notre normalisation applicative minimale.

Sur les exemples inventés :

- `موش` est découpé en `مو`, `##ش` ; `moch` en `m`, `##oc`, `##h`.
- `ما عجبنيش` devient `ما`, `عجب`, `##نيش`.
- Les deux emojis de `🙂😍` sont représentés par `🙂`, `##😍` : il ne faut pas supposer que tous les emojis sont UNK, ni généraliser cet exemple à tous les symboles.
- Le probe vocalisé `بَاهي` devient `[UNK]` ; le cas motive une inspection des marques arabes, sans supprimer automatiquement les voyelles des corpus.
- Le caractère zéro-largeur du probe dédié est retiré par le nettoyage interne. Le texte original reste conservé dans notre système.

Ces observations décrivent un découpage, pas une reconnaissance correcte de la négation ou du sentiment. Elles ne sont pas une annotation des données ni une mesure de classification.

## Taille déclarée de l'architecture

À partir de la configuration BERT publique :

| Variante théorique | Paramètres |
| --- | ---: |
| Encodeur sans pooler | 108 490 752 |
| Encodeur avec pooler | 109 081 344 |
| Classification à deux labels avec pooler | 109 082 882 |
| Classification à trois labels avec pooler | 109 083 651 |

Ces dimensions restent sous un milliard. Il s'agit d'un calcul sur l'architecture standard, pas d'un `numel()` sur des poids chargés. Le modèle général déclaré est un modèle de langage masqué ; les futures têtes POS/NEG ou MSA/TN/EG devront être adaptées et vérifiées séparément.

## Reproduire

Depuis la racine, avec le profil installé et les cinq fichiers publics déjà en cache :

```sh
sh scripts/tokenizer_uv.sh sync --locked
sh scripts/tokenizer_uv.sh run --locked --offline python -m unittest discover -s tests/tokenizer -p test_encoder_tokenizer.py
sh scripts/project_uv.sh run --locked --offline python -m unittest discover -s tests -p test_encoder_tokenizer_notebook.py
sh scripts/tokenizer_uv.sh run --locked --offline python scripts/run_encoder_tokenizer_audit.py --output outputs/tokenizer_audits/NOUVEL_AUDIT
sh scripts/tokenizer_uv.sh run --locked --offline python scripts/verify_encoder_tokenizer_audit.py outputs/tokenizer_audits/NOUVEL_AUDIT --report reports/NOUVEAU_RAPPORT_TOKENIZER
```

Pour reconstruire le cache absent, lancer explicitement `python scripts/fetch_camelbert_tokenizer.py --download` depuis un Python du projet disposant du réseau. Sans `--download`, ce script vérifie uniquement le cache. Il ne propose aucune option de téléchargement de poids.

L'export de dépendances est régénéré avec `sh scripts/tokenizer_uv.sh export --locked --no-dev --format requirements-txt --output-file environments/tokenizer/requirements.txt`. Les sources de corpus restent locales et autorisées ; le notebook possède un mode off et un mode replay explicites.

## Prochaine étape et statut

La provenance du modèle, son tokenizer et le choix initial de longueur disposent maintenant de preuves. Restent le chargement contrôlé du checkpoint général, la mesure de faisabilité et de mémoire, puis la tête sur encodeur gelé et le fine-tuning. Aucun score de validation de l'encodeur, budget GPU mesuré, entraînement L05 ou test final n'est réalisé par cet audit.
