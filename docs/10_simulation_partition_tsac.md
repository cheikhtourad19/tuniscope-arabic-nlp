# TuniScope — Simulation de la partition TSAC train/validation

**Livrables concernés :** L02 (protocole de partitionnement, équilibre et prévention des fuites), contribution partielle ; L03 (script reproductible), contribution partielle.

**Version :** `tsac-grouped-validation-v1`, 16 octobre 2026. **État :** simulation en mémoire ; aucun manifeste ni jeu de données préparé n'a été écrit.

## Choix préalable et méthode

Nous fixons une validation de **15 %** du train TSAC retenu dans la projection et une **graine 42**. La graine est le nombre qui rend le tirage pseudo-aléatoire reproductible. Cette proportion est un choix de protocole, non un résultat sélectionné d'après les performances du test. Le test officiel demeure séparé et n'est pas ouvert par ce contrôle.

Nous partons des 11 484 groupes de la règle [`tsac-candidate-components-v1`](09_regle_groupement_tsac.md). Après un tri stable des références source, le script mélange les 209 groupes de plusieurs lignes avec la graine 42 et en prend 15 % (31 groupes) pour la validation, tant que cela ne dépasse pas le quota de chaque étiquette. Il complète ensuite chaque quota POS et NEG avec des groupes d'une seule ligne, eux aussi mélangés de manière reproductible. Un groupe contenant POS et NEG compte dans les deux quotas selon ses lignes d'origine ; nous ne changeons aucun label. Un groupe sélectionné est toujours attribué en entier.

Le script `python3 scripts/audit_tsac_validation_split.py` revérifie les empreintes des relevés privés par l'intermédiaire de l'audit de groupes, calcule les affectations **en mémoire**, contrôle qu'aucun groupe ne traverse la séparation et affiche seulement des agrégats et une empreinte de l'affectation. L'empreinte permet de constater qu'une relance produit exactement la même affectation sans publier le relevé ligne par ligne.

## Résultat vérifié

| Ensemble simulé | POS | NEG | Total |
| --- | ---: | ---: | ---: |
| Train conservé | 5 255 | 4 798 | 10 053 |
| Validation | 927 | 847 | 1 774 |
| Total projeté après exclusions candidates | 6 182 | 5 645 | 11 827 |

La validation représente **14,9996 %** des lignes projetées. Elle comprend 31 groupes de plusieurs lignes, dont un groupe à étiquettes mixtes. **Zéro groupe** est partagé entre train et validation. L'empreinte SHA-256 de cette affectation simulée est `693e441c86b72ea102b0431e3f2821cbfd4763583d55a97bcc6b2bf659fd3213`.

## Portée et limites

Cette vérification porte sur les **lignes train projetées** après les 1 842 exclusions candidates ; elle ne démontre pas encore qu'un manifeste ou une copie de données a été correctement produit. La similarité utilisée pour grouper les lignes reste heuristique : une parenté sémantique non repérée pourrait subsister entre les partitions. L'absence d'identifiant d'auteur ou de conversation empêche de vérifier l'indépendance à ce niveau. Les groupes mixtes ne sont pas des erreurs d'annotation démontrées.

L'étape suivante est de matérialiser et contrôler les manifestes TSAC et MADAR, en conservant les références source, les motifs d'exclusion et le test officiel figé ; puis de remplir les cartes de données. **L02 et L03 restent incomplets.**
