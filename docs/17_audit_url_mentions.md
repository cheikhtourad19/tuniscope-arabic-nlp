# TuniScope — Audit des URL et mentions pour la normalisation

**Livrables concernés :** L02 (contrôle de composition et limites), contribution partielle ; L03 (règle candidate reproductible), contribution partielle. **Phase :** S3. **Date :** 16 octobre 2026.

## Question et règle candidate

Le cahier des charges propose de remplacer les URL et les mentions personnelles par des marqueurs stables. Nous avons d'abord évalué leur fréquence **dans les lignes train/validation retenues**, sans ouvrir le test officiel. La règle candidate `norm-nfc-space-web-v2-candidate` applique la [normalisation NFC/espaces v1](16_normalisation_minimale_v1.md), puis remplace une URL reconnue par `<URL>` et une mention commençant par `@` par `<USER>`. Elle conserve la ponctuation finale, les émojis, la négation, les chiffres Arabizi et les autres mots. Elle laisse les adresses électroniques ordinaires inchangées dans ce contrôle.

Le code est dans `scripts/normalize_web_tokens.py` et l'audit agrégé dans `scripts/audit_web_token_policy.py`. La reconnaissance par expressions régulières est volontairement limitée aux formes `http://`, `https://`, `www.` et aux mentions `@nom` sans espace. Elle peut manquer des graphies atypiques et ne garantit pas l'anonymisation complète. Le texte brut reste disponible dans `text_raw`, en mémoire et sous les restrictions de la source.

## Résultat vérifié

| Corpus / partition | Lignes examinées | Lignes avec URL reconnue | Lignes avec mention reconnue | Lignes changées par rapport à v1 | Nouveau doublon exact train/validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC train | 10 053 | 1 | 0 | 1 | 0 |
| TSAC validation | 1 774 | 0 | 0 | 0 | 0 |
| MADAR train | 6 388 | 0 | 0 | 0 | 0 |
| MADAR validation | 800 | 0 | 0 | 0 | 0 |

Il s'agit de **comptes de motifs reconnus par la règle**, pas d'une certitude qu'aucune autre forme d'URL, de mention ou de donnée personnelle n'existe. Les 15 tests locaux ont réussi, dont des exemples synthétiques vérifiant la ponctuation, les mentions en écriture arabe, les adresses électroniques et la conservation de la négation, des émojis et de l'Arabizi. Aucun corpus ou manifeste n'a été modifié.

## Décision de protocole

Nous gardons `norm-nfc-space-v1` comme **première variante normalisée de référence** face à `raw-v0`. La règle avec marqueurs reste une **variante candidate**, non imposée silencieusement à tous les modèles : elle ne change qu'une seule ligne TSAC train dans le périmètre audité. Nous pourrons l'essayer comme ablation ciblée ou l'employer dans un contexte de confidentialité si l'usage le justifie, après contrôle des motifs manqués. Cette décision ne découle d'aucune performance du test ; le test officiel est resté fermé. La politique du tatweel, des répétitions et des autres substitutions demeure à préciser. **L02, L03 et S3 restent incomplets.**
