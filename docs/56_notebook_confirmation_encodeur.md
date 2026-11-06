# Exécution de la confirmation encodeur

**6 novembre 2026 — contribution à L05, L07 et L12.** Le protocole, le contrôle technique et le notebook sont vérifiés. **Mise à jour après exécution personnelle : les six entraînements FULL et leur contrôle sont terminés.** Voir les [résultats de confirmation](57_resultats_confirmation_encodeur.md) ; ce document décrit la procédure de lancement et de rejeu. Ne pas relancer FULL pour simplement consulter les scores.

## Objectif et configuration

Après le criblage de trois taux sur la graine 42, nous fixons les taux du rapport local dont les artefacts sont conservés : **5e-5 pour TSAC et 2e-5 pour MADAR**. Nous entraînons chaque tâche sur les graines **13, 42 et 2026**, en repartant du checkpoint général pour chacune. Le protocole [55](55_protocole_confirmation_encodeur.md) et le plan `configs/camelbert_confirmation_v1.json` fixent ces choix avant les scores de confirmation.

Le notebook ne recherche plus de taux et ne récupère pas le meilleur checkpoint du criblage. Il entraîne six nouveaux classifieurs, rapporte les six résultats, puis la moyenne et l’écart-type échantillonnal par tâche. L’architecture et les autres réglages du protocole 46 sont conservés.

| Tâche | Train par graine | Validation par graine | Taux fixé | Graines FULL |
| --- | ---: | ---: | ---: | --- |
| TSAC | 10 053 | 1 774 | 5e-5 | 13, 42, 2026 |
| MADAR | 6 388 | 800 | 2e-5 | 13, 42, 2026 |

Les partitions sont les mêmes sur les trois graines. La validation n’entre pas dans les mises à jour des poids ; elle sert au choix de l’époque. Le test officiel n’est pas sélectionné.

## Notebook et scripts

Ouvrir `notebooks/08_confirmation_encodeur.ipynb` avec le noyau **TuniScope (notebooks)**, puis redémarrer ce noyau si d’anciens modules Python ont déjà été importés. Le profil encodeur est appelé dans un processus distinct ; aucune installation de PyTorch dans le noyau léger n’est nécessaire.

Le mode par défaut est **full**, l’appareil **auto**. Sur le Mac compatible, il choisit MPS. Exécuter les cellules dans l’ordre, ou utiliser « Run All », une seule fois. Les paramètres `TUNISCOPE_CONFIRMATION_MODE` et `TUNISCOPE_CONFIRMATION_DEVICE`, s’ils sont définis dans l’environnement, remplacent ces valeurs : lire la première sortie avant de poursuivre.

Le FULL effectue successivement :

1. Un smoke apparié TSAC de 100 lignes par partition, puis le rechargement de ses deux modèles.
2. Trois adaptations complètes TSAC, leur rechargement sur toute la validation, puis trois PNG.
3. Les mêmes opérations pour MADAR avec son taux fixé.
4. Le contrôle de campagne, les moyennes et écarts-types, la synthèse et les cartes provisoires.

Les quatre petits modèles de contrôle s’ajoutent aux six modèles FULL ; ils ne sont pas inclus dans les moyennes finales. Le programme s’arrête si un contrôle échoue et ne substitue pas un smoke au FULL.

| Fichier | Responsabilité |
| --- | --- |
| `encoder_confirmation_plan.py` | Vérifier le choix fixé contre le rapport source et produire la configuration de chaque tâche |
| `encoder_confirmation_notebook_support.py` | Relier le noyau léger au profil encodeur et afficher les figures du même run |
| `run_encoder_confirmation.py` | Orchestrer les contrôles, les six modèles, les agrégats et les preuves |
| `run_encoder_finetuning.py` | Charger le modèle général, entraîner, sélectionner l’époque et sauvegarder |
| `encoder_finetuning.py` | Construire le modèle et effectuer entraînement et inférence |
| `verify_encoder_finetuning.py` | Recharger chaque checkpoint, rejouer la validation et contrôler les journaux |
| `plot_encoder_finetuning.py` | Produire les courbes et matrices de confusion PNG |
| `encoder_source_provenance.py` | Vérifier les empreintes du code courant ou d’un instantané historique conservé |

Tous les scripts sont dans `scripts/`. Les dépendances et verrous existants ne changent pas.

## Sorties attendues du FULL

Le notebook crée un dossier neuf `outputs/encoder_finetuning/confirmation/notebooks/<identifiant>_full/` et imprime son chemin exact. Il contient notamment :

```text
confirmation.json                 # index et agrégats des trois graines
campaign_verification.json        # contrôle des fichiers et preuves de rejeu
summary.md                        # six scores et moyennes, lisibles
model_card_tsac.json               # carte provisoire de validation
model_card_madar.json
evidence_without_weights.zip      # preuves et figures, sans poids
tsac_smoke/ et madar_smoke/        # contrôles techniques distincts
tsac_full/ et madar_full/
    results.json
    verification.json
    figures/                      # trois PNG par tâche
    <corpus>_finetuned_seed_<graine>/
        model.safetensors
        tuniscope.json, history.json, predictions.json, …
```

Les fichiers exacts supplémentaires, dont tokenizer et prédictions par époque, sont référencés dans les empreintes du run. Chaque PNG montre pertes train/validation, macro-F1 par époque et matrice de confusion du checkpoint retenu. Les résultats restent privés ; ne pas ajouter les checkpoints ou textes à Git.

**Sauvegarde :** `evidence_without_weights.zip` ne permet pas de refaire l’inférence seul. Pour conserver les modèles, sauvegarder également les dossiers de checkpoints. Une archive laissée dans la même session Kaggle ne protège pas contre la disparition de cette session. Les sources protégées continuent d’être déposées manuellement et conservées séparément.

La campagne précédente de six adaptations locales a cumulé 46 min 38 s pour les stages FULL, hors contrôles/rejeu/export. Prévoir un ordre de grandeur de **50 minutes ou davantage** sur ce Mac, sans garantie : les époques retenues, la charge et le matériel peuvent modifier cette durée. Ce n’est pas une mesure de la nouvelle confirmation.

## Rejeu et commande équivalente

Pour revoir un FULL terminé, mettre `TRAINING_MODE = 'replay'`, `REPLAY_EXPECTED_MODE = 'full'` et `REPLAY_RUN` au chemin exact imprimé par cette exécution, dans la première cellule de code, puis relancer les cellules. Ne pas laisser `full` si l’objectif est seulement de consulter les résultats. La valeur de chemin fournie dans le notebook est à remplacer par un run existant.

Ce rejeu vérifie les artefacts, les prédictions enregistrées, les métriques et les preuves de rechargement déjà produites ; il n’effectue pas de nouvelle inférence. Un rejeu des poids eux-mêmes peut être demandé séparément avec `verify_encoder_finetuning.py` pour chacun des deux dossiers FULL.

La commande équivalente, depuis la racine, utilise un dossier neuf :

```sh
sh scripts/encoder_uv.sh run --locked --offline python -u scripts/run_encoder_confirmation.py \
  --mode full --device auto \
  --output outputs/encoder_finetuning/confirmation/ma_confirmation_full
```

Le code demeure partagé CPU/MPS/CUDA. Une campagne CUDA constitue une expérience distincte ; la préparation Kaggle documentée dans le guide 53 doit être effectuée dans toute session vide. La nouvelle confirmation complète n’a pas été vérifiée sur CUDA à ce stade.

## Preuves déjà obtenues

Campagne locale réelle : `outputs/encoder_finetuning/confirmation/2026-11-06_selected_rates_mps_smoke/`. Deux stages de tâche, quatre modèles, huit époques avec métriques par classe et prédictions conservées. Les quatre checkpoints ont été rechargés sur leurs 100 exemples de validation : différence maximale de probabilité **0**.

Les stages d’entraînement ont duré **14,753339 s pour TSAC** et **10,312520 s pour MADAR**. La campagne jusqu’au contrôle final a duré **52,055826 s**, avec les appels séparés de vérification et d’export ; ces mesures ne sont pas une estimation proportionnelle d’un FULL.

Le contrôle de campagne est lié à `confirmation.json` par SHA-256 `5aa72d2e4f217c8d91752e7a07a3634374014167c105521ca90e72ffe65b4cdd`. Le notebook a été exécuté en rejeu smoke dans un vrai noyau Jupyter isolé : **quatre cellules de code et quatre images PNG**, sans nouvel apprentissage. Preuve privée : `outputs/notebook_checks/2026-11-06/confirmation_native_smoke_replay.json`. Deux figures adaptées ont aussi été inspectées visuellement.

L’archive réelle contient 58 membres, aucun `.pt`, `.bin` ou `.safetensors`. Les tests ciblés couvrent les taux et graines, l’arrêt sur échec de smoke, les modes notebook, le calcul de dispersion, les cartes et l’exclusion des poids de l’archive. Le contrôle du criblage historique continue de réussir grâce aux sources anciennes conservées ; ses résultats n’ont pas été réécrits.

Un instantané privé des sources exécutées et des paramètres est conservé dans `outputs/encoder_finetuning/source_snapshots/2026-11-06_confirmation/`. Les cellules de code du notebook 07 restent identiques à l’instantané antérieur ; ses nouvelles sorties personnelles ont été conservées, avec une copie privée distincte. Le protocole de confirmation déjà lié aux preuves par empreinte ne doit pas être modifié silencieusement.

## État des livrables

- **L05 :** notebook, six checkpoints FULL sélectionnés et rechargements vérifiés ; résultats dans le document 57. Le test final reste distinct.
- **L07 :** résultats de confirmation et agrégats obtenus ; test figé, bootstrap et évaluations complémentaires restent ouverts.
- **L08 :** deux cartes provisoires réelles vérifiées ; intégration de ces modèles et artefacts RAG restent ouvertes.
- **L12 :** environnement inchangé, protocole, commandes, rapport généré et rejeu FULL natif vérifiés ; reproduction complète indépendante non établie.

Cette préparation n’entraîne aucun LLM, ne remplace aucun modèle de l’application et n’ouvre pas le test officiel.
