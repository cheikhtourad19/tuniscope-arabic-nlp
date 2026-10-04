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

## 2026-10-04 — Audit des groupes MADAR et des doublons exacts TSAC

- Commit Git vérifié lors de la dernière relance : `7010dba` (script versionné à cette révision ; rapport encore en cours de rédaction).
- Objectif : vérifier le regroupement des traductions MADAR et mesurer répétitions, conflits POS/NEG et recouvrements train–test TSAC sans modifier les sources.
- Corpus et version du manifeste : MADAR v1.1 Corpus-26 et TSAC original ; aucun manifeste créé.
- Graine du partitionnement : sans objet ; partitions officielles observées sans changement.
- Modèle et révision : sans objet.
- Fichier de configuration : `scripts/audit_duplicates.py`, Python standard 3.9.6.
- Matériel et environnement : Mac arm64 ; aucun GPU utilisé.
- Résumé des résultats : 2 000 groupes MADAR complets et cohérents, mais quatre chaînes identiques entre partitions ; TSAC comporte 846 groupes de doublons exacts, dont 316 présents dans train et test, et 13 groupes à labels POS/NEG contradictoires. Comptage principal confirmé par `awk` indépendant.
- Erreurs et observations : les chiffres sont agrégés ; aucun texte protégé n'a été publié. Les quasi-doublons et la revue humaine des cas restent à faire.
- Décision ou prochaine action : ne supprimer ni relabelliser aucune ligne ; examiner les cas privés et choisir une politique de décontamination avant les manifestes. Détails dans `docs/04_data_audit_groups_and_duplicates.md`.

## 2026-10-04 — Dépistage des quasi-doublons TSAC train–test

- Commit Git de départ : `6c9c7e8` ; le script de dépistage n'était pas encore commité lors de l'exécution.
- Objectif : repérer, sans décision automatique, des commentaires très similaires entre le train et le test officiel TSAC.
- Corpus et version du manifeste : TSAC original ; aucun manifeste créé.
- Graine du partitionnement : sans objet ; aucune partition créée ou modifiée.
- Modèle et révision : sans objet ; méthode heuristique de trigrammes de caractères et `difflib.SequenceMatcher` de Python 3.9.6.
- Fichier de configuration : seuils inscrits dans `scripts/screen_near_duplicates.py` ; index de revue locaux sous `data/interim/`, exclus de Git.
- Matériel et environnement : Mac arm64 ; aucun GPU ni paquet externe utilisé.
- Résumé des résultats : 324 formes identiques train–test après NFKC/espaces regroupés (467 lignes test) ; 124 paires supplémentaires candidates, concernant 98 lignes test. Deux paires candidates peuvent rapprocher des labels opposés.
- Erreurs et observations : dépistage conservateur et non exhaustif ; 139 formes test très courtes exclues de la recherche de quasi-doublons, mais non des contrôles exacts. Aucune paire candidate n'a été validée humainement.
- Décision ou prochaine action : revue privée des cas et politique de décontamination avant les manifestes. Voir `docs/05_data_audit_near_duplicate_screen.md`.
