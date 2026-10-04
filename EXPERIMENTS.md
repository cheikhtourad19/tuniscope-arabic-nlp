# TuniScope — Journal des expériences

Aucun entraînement ni essai de modèle n'a encore été exécuté. Les audits de données reproductibles sont également consignés ici.

Créer une entrée pour chaque expérience significative.

## Modèle d'entrée

```markdown
## AAAA-MM-JJ — nom court de l'expérience

- Commit Git :
- Objectif :
- Corpus et version du manifeste :
- Graine du partitionnement :
- Modèle et révision :
- Fichier de configuration :
- Matériel et environnement :
- Résumé des résultats :
- Erreurs et observations :
- Décision ou prochaine action :
```

## 2026-10-04 — Audit initial des sources (intégrité et structure)

- Commit Git de départ : `2d9ed45` ; le script de cet audit n'était pas encore commité lors de l'exécution.
- Objectif : vérifier les archives, leurs copies extraites, l'encodage et les effectifs bruts sans modifier les corpus.
- Corpus et version du manifeste : MADAR v1.1 et TSAC, archives identifiées dans `SOURCE_REGISTER.md` ; aucun manifeste créé.
- Graine du partitionnement : sans objet ; aucune nouvelle partition.
- Modèle et révision : sans objet.
- Fichier de configuration : `scripts/audit_structure.py`, Python standard 3.9.6.
- Matériel et environnement : Mac arm64 ; aucun GPU utilisé.
- Résumé des résultats : archives conformes aux SHA-256 enregistrés, ZIP intègres, 34 fichiers MADAR et 6 fichiers TSAC identiques après extraction ; huit fichiers utiles en UTF-8. Quatre lignes vides dans le train TSAC, aucune dans son test ; aucune anomalie structurelle dans les quatre TSV MADAR.
- Erreurs et observations : deux noms de fichiers macOS `Icon` vides perdent leur retour chariot final lors de l'extraction ; cas traité explicitement dans le script. Doublons et chevauchements non encore vérifiés.
- Décision ou prochaine action : ne modifier aucun original ; effectuer les contrôles de groupes, doublons et conflits avant de figer les manifestes. Détails et empreintes dans `docs/03_data_audit_structure.md`.
