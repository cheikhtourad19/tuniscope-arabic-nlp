# Préparation de l'encodeur : provenance et audit du tokenizer

**Date :** 4 novembre 2026. **L03, L05 et L12 : contributions partielles.** Nous préparons CAMeLBERT Mix conformément à la section 9 du cahier des charges. Cette étape n'entraîne pas d'encodeur et ne télécharge aucun poids préentraîné.

## Source et révision

Modèle : `CAMeL-Lab/bert-base-arabic-camelbert-mix`, révision complète **`9be352797bdf28a9ae21e2ae582aaaca7abdb22d`**. La [fiche officielle à cette révision](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix/blob/9be352797bdf28a9ae21e2ae582aaaca7abdb22d/README.md) déclare la licence **Apache-2.0**. Le [dépôt de code des auteurs](https://github.com/CAMeL-Lab/CAMeLBERT) affiche séparément MIT : ces deux déclarations ne doivent pas être confondues.

Nous sélectionnons le modèle général et non un checkpoint déjà adapté sur MADAR. La fiche décrit un préentraînement sur de l'arabe standard, dialectal et classique ; elle ne garantit pas la qualité sur nos écritures latines ou nos commentaires tunisiens. Nous ne présentons pas cette inspection comme une reproduction des résultats publiés.

Le script `fetch_camelbert_tokenizer.py --download` n'autorise que cinq petits fichiers texte à la révision figée : README, configuration de modèle, configuration de tokenizer, tokens spéciaux et vocabulaire. Le total téléchargé est **316 515 octets**. Le dossier `.tools/tokenizers/` est ignoré par Git ; son `source.json` conserve les URL publiques, tailles et SHA-256. Les formats de poids `.bin`, `.safetensors`, `.h5` et `.msgpack` ne sont pas téléchargés. Une relance de l'audit vérifie les empreintes et utilise les fichiers locaux uniquement.

## Architecture déclarée et nombre de paramètres

La [configuration officielle](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix/blob/9be352797bdf28a9ae21e2ae582aaaca7abdb22d/config.json) déclare BERT, 12 couches, taille cachée 768, 12 têtes d'attention, couche intermédiaire 3 072, vocabulaire 30 000, 512 positions et deux types de segment. L'architecture d'origine est `BertForMaskedLM`, qui apprend à prévoir une unité masquée ; ce n'est pas encore notre classifieur.

Nous calculerons les paramètres théoriques depuis ces dimensions pour le backbone avec/sans pooler et les têtes de classification à deux/trois labels. Ce comptage décrit l'architecture ; il ne remplace pas le futur contrôle `numel()` d'un checkpoint réellement chargé ni la vérification des paramètres entraînables. Les embeddings du vocabulaire ne doivent pas être comptés deux fois si une tête de langage réutilise leurs poids.

## Audit fixé avant mesure

Configuration : `configs/camelbert_tokenizer_audit_v1.json`. Représentation unique `raw-v0`, manifestes v2, uniquement train et validation. Nous évaluons **toutes** les lignes retenues : cela dépasse les 100 textes demandés pour les groupes disponibles en nombre suffisant. Les catégories trop rares restent explicitement signalées ; elles ne sont ni dupliquées pour atteindre 100 ni supprimées.

Nous utilisons `BertTokenizerFast`, `local_files_only=True`, `trust_remote_code=False`, avec les configurations officielles et les deux tokens de frontière du modèle. La limite de l'architecture est fixée explicitement depuis `max_position_embeddings=512`, car un tokenizer local ne doit pas être supposé connaître cette limite. Aucun vocabulaire n'est appris sur nos données et aucune normalisation applicative supplémentaire n'est ajoutée.

Par corpus, partition et catégorie d'écriture Unicode AR/LATIN/MIXED/UNKNOWN, nous mesurons :

- Effectifs et groupes ; écriture ne signifie pas langue ou dialecte.
- Nombre de sous-mots hors tokens spéciaux, nombre d'unités séparées par espaces et leur rapport agrégé. Le dénominateur est un proxy déclaré, pas une annotation morphologique de mots.
- Longueurs avec CLS/SEP : médiane, P95, P99 et maximum, sans troncature pendant la mesure.
- Nombre/taux de tokens `[UNK]`, nombre/taux de textes concernés, textes sans contenu encodé.
- Pour les longueurs candidates **64, 128, 256 et 512**, effectif/taux de textes qui dépasseraient la limite et nombre d'unités retirées si nous conservions le début avec les tokens spéciaux.

La longueur initiale sera la plus petite candidate qui conserve intégralement au moins **99 % des textes de chaque groupe d'écriture du train ayant au moins 100 exemples**. Les petits groupes sont décrits et conservés mais ne décident pas seuls de la longueur. Si aucune candidate ne satisfait cette règle, nous enregistrons l'échec et retenons provisoirement la limite 512 avec sa couverture réelle. La validation fournit une mesure de contrôle ; elle n'enrichit pas le vocabulaire et ne choisit pas la longueur selon ses labels ou scores.

## Inspection d'exemples inventés

Un jeu de probes séparé illustre la négation arabe et latine, des chiffres Arabizi, les emojis, une phrase française/mixte, les voyelles et répétitions. Seuls ces exemples inventés peuvent être affichés publiquement avec leurs tokens. Les textes des corpus et leurs découpages par ligne restent hors des rapports publics.

L'inspection du normaliseur interne décrit ce qui arrive réellement aux caractères : conserver le texte brut dans notre chargeur n'implique pas que le tokenizer laisse tout inchangé. Nous distinguons l'apparition d'UNK, le découpage en plusieurs morceaux et les transformations du normaliseur. Une négation découpée n'est pas automatiquement une négation comprise.

## Vérification et limites

Les tests sur tokenizer minuscule inventé vérifient les tokens spéciaux, inconnus, longueurs, règles de sélection, partitions sans test et absence d'écriture de texte de corpus. Le vérificateur rejouera l'audit à partir des mêmes fichiers locaux et des manifestes, avec contrôle des empreintes et des résultats agrégés. Les rapports publics ne contiennent pas de lignes du corpus.

Un faible taux UNK ne garantit ni compréhension, ni robustesse, ni meilleur macro-F1. Une même unité inconnue peut représenter un mot entier ; le taux UNK par token n'est donc pas un taux de mots correctement compris. Les catégories d'écriture ne permettent pas à elles seules d'annoter français, anglais ou Arabizi.

Le profil `environments/tokenizer/` est séparé des profils classique, BiGRU et application. Il ne contient pas de moteur d'entraînement. Après l'audit, les étapes suivantes resteront le chargement contrôlé des poids, le test de faisabilité, l'encodeur gelé et le fine-tuning ; aucune ne sera présentée comme réalisée ici.

Références : [documentation BERT](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert), [tokenizers Hugging Face](https://huggingface.co/docs/tokenizers/).
