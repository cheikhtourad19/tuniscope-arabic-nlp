# TuniScope — Carte de données MADAR Corpus-26

**Livrable :** L02, carte de données propre au sous-ensemble MADAR. **Version :** 0.2, 16 octobre 2026. **État :** fondée sur les sources, manifestes v1 et audit d'écriture vérifiés ; normalisation et vérifications finales encore à compléter.

## Identification, accès et usage

Nous utilisons une sélection du [MADAR Parallel Corpus](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), distribution publique v1.1 obtenue après inscription et acceptation des conditions le 2 octobre 2026. Son archive, son empreinte et ses conditions d'accès sont consignées dans le [registre des sources](../SOURCE_REGISTER.md). L'usage est limité à la recherche et à l'évaluation dans le cadre autorisé ; nous ne redistribuons pas le corpus ni son lien privé reçu par courriel. Toute publication de dérivés textuels ou de modèles doit respecter les conditions de la ressource.

La tâche du projet est l'identification de variété : `MSA` pour l'arabe standard moderne, `TN` pour Tunis et Sfax regroupées, et `EG` pour Le Caire. Une observation correspond à la traduction d'une phrase source dans l'une de ces quatre variétés. Le regroupement `TN` ne signifie pas que Tunis et Sfax représentent tous les parlers tunisiens ; `EG` ne couvre ici que Le Caire. Les phrases de MADAR relèvent du domaine du voyage.

## Structure et sélection

Chaque fichier TSV porte les colonnes `sentID.BTEC`, `split`, `lang` et `sent`. L'identifiant `sentID.BTEC` relie les traductions parallèles d'une même phrase : les quatre traductions doivent rester dans la même partition. Nous utilisons les 2 000 identifiants communs aux fichiers MSA, Tunis, Sfax et Cairo dans la partie Corpus-26, soit **8 000 lignes** avant exclusion. Les lignes supplémentaires Corpus-6 des trois autres fichiers sont hors du périmètre initial, pas des exemples « nettoyés » ou supprimés.

| État | MSA | TN | EG | Total |
| --- | ---: | ---: | ---: | ---: |
| Corpus-26 sélectionné avant exclusion | 2 000 | 4 000 | 2 000 | 8 000 |
| Exclusions du train de travail | 3 | 6 | 3 | 12 |
| Train v1 | 1 597 | 3 194 | 1 597 | 6 388 |
| Validation officielle | 200 | 400 | 200 | 800 |
| Test officiel | 200 | 400 | 200 | 800 |

Nous conservons les partitions officielles Corpus-26 : 1 600 groupes source dans le train, 200 en validation et 200 dans le test avant exclusion. Quatre chaînes de texte exactes traversaient des partitions différentes ; elles concernaient **trois groupes source du train**. Nous avons écarté les quatre traductions de chacun de ces groupes côté train, soit 12 lignes. Le manifeste v1 contient alors 1 597 groupes train, 200 validation et 200 test ; aucun groupe source ni aucune chaîne exacte retenue ne traverse ces partitions. Les exclusions et leurs références se trouvent dans les [manifestes v1](11_manifestes_donnees_v1.md), sans texte protégé.

## Étiquettes, préparation et vérification

Les étiquettes `MSA`, `TN` et `EG` découlent des fichiers de variété sélectionnés ; ce ne sont pas des annotations individuelles que nous avons recréées. Les quatre traductions parallèles partagent un `group_id` fondé sur `sentID.BTEC`. Les fichiers sources UTF-8, leurs colonnes, leurs empreintes SHA-256 et l'intégrité des groupes ont été vérifiés. Les fichiers originaux et leurs partitions officielles n'ont pas été modifiés.

Le manifeste est une table de provenance et de partition ; il ne contient pas la colonne `sent`. Un [chargeur raw-v0](14_chargement_donnees_en_memoire.md) retrouve le texte en mémoire depuis les sources autorisées, sans créer de copie textuelle sur disque. La [normalisation minimale `norm-nfc-space-v1`](16_normalisation_minimale_v1.md) a changé 18 lignes train et trois lignes validation en RAM, surtout pour les espaces, tout en conservant le brut. Elle ne crée aucun doublon exact train/validation, mais révèle 82 groupes de formes répétées à l'intérieur d'une partition, dont 13 à plusieurs labels de variété. Nous ne les supprimons ni ne les réétiquetons automatiquement. La clé d'égalité exacte ne démontre pas l'absence de paraphrases entre partitions.

Un [audit séparé](18_audit_tatweel_repetitions.md) trouve une ligne train avec tatweel et une avec au moins trois lettres identiques ; aucune de ces règles candidates n'a été imposée à la variante de référence.

## Limites et contrôles restants

Les classes sont déséquilibrées : `TN` contient deux villes et deux fois plus de lignes que `MSA` ou `EG`. Nous publierons donc les résultats par classe, en plus du macro-F1. Le domaine du voyage limite la généralisation à des textes spontanés ou à d'autres thèmes. La variété dominante dans un texte très court ou identique entre villes peut être linguistiquement indéterminée, même si le label de source est connu.

L'[audit d'écriture et de longueur](15_audit_ecriture_longueur.md) trouve dans le train **6 387 lignes `AR` et une `MIXED`** selon la présence de lettres Unicode ; les 800 lignes de validation sont `AR`. Il fournit aussi les longueurs P50/P95 par variété. Cela ne démontre pas la langue effective ou la non-ambiguïté de chaque texte. Nous devons examiner les éventuelles ambiguïtés de variété, définir la normalisation et confirmer les modalités de citation et de diffusion de tout artefact dérivé avant publication. **Cette carte apporte la provenance, les comptes et la méthode de séparation à L02, mais L02 n'est pas achevé.**
