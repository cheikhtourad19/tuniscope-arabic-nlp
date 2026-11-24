# Jeux de défi Gold — Fiche de collecte et de vérification

Ce document prépare **L02 — Données** et **L07 — Évaluation**.

Les jeux de défi constituent une évaluation indépendante. Les résultats d’audit déjà produits concernent TSAC et MADAR en validation. La collecte locale reste ouverte. Nous consignons la date de création, les choix effectués avant le gel et l’écart éventuel au calendrier initial. Nous ne changeons plus les modèles à partir de ces exemples.

| Jeu | Cible principale | Règle de provenance |
| --- | --- | --- |
| Sentiment tunisien | 300 avis · 100 en écriture arabe, 100 en Arabizi, 100 avec mélange de langues · au moins 100 POS et 100 NEG | Création humaine indépendante. Les avis ambigus sont mis de côté avec le label AMB. |
| Dialecte | 150 phrases · 50 MSA, 50 TN, 50 EG | Les phrases EG écrites pour le jeu sont vérifiées par une personne compétente. Une ressource autorisée peut fournir les exemples EG à la place. |

La catégorie « mélange de langues » se juge sur le contenu. Elle peut apparaître entièrement en caractères latins. Un détecteur d’alphabet ne suffit pas à l’attribuer.

## Consigne de collecte

1. Rédiger des avis et des phrases indépendamment des exemples déjà utilisés. Pour les phrases dialectales, répartir les sujets et les longueurs pour éviter qu’un label se déduise simplement du thème.
2. Affecter le label avant de consulter une prédiction. Utiliser AMB pour un avis dont la polarité reste incertaine. Mettre les variantes d’un même exemple dans le même groupe.
3. Garder un identifiant pseudonyme et une référence privée de consentement ou de licence. Écarter les informations personnelles du texte. Vérifier les références et la compétence linguistique avant le gel.
4. Exporter les deux fichiers et les contrôler. Une personne qui collecte ou relit peut utiliser le registre, sans lancer les modèles.

## Format privé JSONL

Une ligne correspond à un objet JSON. Ces lignes constituent des gabarits à remplir.

```json
{"id":"","group_id":"","text":"","label":"","category":"","origin":"human","source_ref":"","permission_ref":"","review_ref":""}
```

- **Sentiment** · `label` vaut POS, NEG ou AMB. `category` vaut AR, ARABIZI ou MIXED_LANG.
- **Dialecte** · `label` et `category` valent MSA, TN ou EG. `origin` vaut human ou authorized_resource. Pour EG de création humaine, `review_ref` identifie la revue de compétence et de label.
- **Dans les deux jeux** · `source_ref` indique l’origine pseudonymisée. `permission_ref` renvoie à la preuve privée. Les fichiers et preuves restent dans un dossier local ignoré par Git.

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/check_gold_challenges.py --sentiment outputs/gold_challenge/sentiment.jsonl --dialect outputs/gold_challenge/dialect.jsonl --compare-working
```

Le contrôle détecte les champs manquants, les doublons normalisés dans chaque fichier et les quotas incomplets. L’option de comparaison cherche les textes déjà présents dans train/validation. Il reste à vérifier la pertinence des labels, le consentement ou la licence, et les quasi-doublons. Le passage du contrôle prépare le gel du jeu, qui se documente séparément.
