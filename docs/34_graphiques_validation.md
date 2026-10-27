# Visualisation reproductible des résultats de validation

**Date :** 27 octobre 2026. **Contributions partielles :** L04, L07, L11 et L12. Les figures représentent une expérience classique existante, pas un nouvel entraînement ni une évaluation sur le test officiel.

## Objectif et séparation des opérations

**Mise à jour du 27 octobre :** le notebook 02 intègre désormais l'apprentissage complet, la sélection du bon run et l'export automatique des six figures. Le notebook 03 reste un lecteur autonome facultatif sans entraînement. Voir `docs/35_entrainement_complet_et_export_figures.md` ; les paragraphes de vérification ci-dessous décrivent l'étape initiale de création du lecteur autonome.

Nous séparons le notebook 02, qui peut réellement entraîner, du notebook `03_visualisation_validation.ipynb`, qui relit et visualise un run sélectionné explicitement. La version actuelle du notebook 02 utilise `full` par défaut : exécuter toutes ses cellules ne constitue donc pas une simple consultation. Ses réglages et sorties existants ne sont pas modifiés par cette étape.

Le notebook 03 appelle `scripts/plot_validation_results.py`. Ce module ne contient aucun ajustement de modèle et ne lit pas de corpus brut pour construire les graphiques. Il contrôle les empreintes des prédictions, l'alignement des identifiants/groupes/labels et recalcule les métriques globales, par classe et par écriture. L'arithmétique indépendante utilise la bibliothèque standard et est comparée à l'évaluateur historique dans les tests fictifs.

Avec `VERIFY_CHECKPOINTS = True`, le notebook lance en plus le vérificateur historique dans l'environnement d'entraînement : modèles locaux de confiance rechargés, prédictions de dev refaites et comparées, sans `fit`. Cette vérification demande les sources locales autorisées de train/dev. Sans cette option, seules les sauvegardes de prédictions sont relues ; ne pas annoncer un rechargement de modèles dans ce cas.

## Dépendance de visualisation séparée

Matplotlib 3.10.7 est installé dans `.tools/notebook-kernel/`, le noyau local **TuniScope (notebooks)**. Il n'est pas ajouté à `.venv`, `pyproject.toml`, `requirements.txt` ou `uv.lock`. Les vérificateurs historiques contrôlent l'empreinte de `uv.lock` : celle-ci reste inchangée.

La dépendance optionnelle est déclarée dans `requirements-visualization.txt`. Si ce noyau existe mais n'a pas encore Matplotlib, installation manuelle préalable avec accès réseau :

```sh
sh scripts/project_uv.sh pip install --python .tools/notebook-kernel/bin/python -r requirements-visualization.txt
```

Il ne s'agit pas d'un nouveau verrouillage complet des dépendances transitives du noyau. L'environnement d'entraînement verrouillé reste la référence pour l'apprentissage et le rechargement. Le notebook ne télécharge rien automatiquement. Sur une autre machine ou dans le cloud, installer explicitement la dépendance de visualisation dans le noyau utilisé ; cette exécution cloud reste non vérifiée.

## Exécution et sélection du run

Dans le notebook 03, choisir :

```python
SOURCE_RUN = ROOT / 'outputs/baselines/2026-10-17_grouped_v2_full'
VERIFY_CHECKPOINTS = True
```

Puis exécuter les cellules dans l'ordre. Un nouveau dossier avec identifiant est créé sous `outputs/figures_validation/`. Les six PNG y sont enregistrés avant leur affichage dans le notebook. Aucune fenêtre graphique externe n'est nécessaire.

Pour un nouvel entraînement du notebook 02, récupérer le chemin exact du run `full` dans sa sortie `runs`, puis le reporter dans `SOURCE_RUN`. Aucun choix silencieux du dossier « le plus récent ». Les modes smoke sont acceptés, mais portent un avertissement visible sur chaque figure.

L'équivalent en terminal, depuis le dépôt et avec le noyau local configuré :

```sh
.tools/notebook-kernel/bin/python scripts/plot_validation_results.py \
  --run outputs/baselines/2026-10-17_grouped_v2_full \
  --output outputs/figures_validation_nouvelle_execution \
  --verify
```

Le dossier de sortie doit être **nouveau**, situé sous `outputs/`. Les runs TF-IDF caractères, mots et combinés v2 sont acceptés ; les lexiques et augmentations disposent d'autres schémas et ne sont pas pris en charge par ce lecteur.

## Figures produites

| Fichier | Contenu et lecture |
| --- | --- |
| `01_scores.png` | Accuracy et macro-F1 de la majorité et du TF-IDF brut/normalisé ; une partie TSAC, une MADAR ; axes des scores de 0 à 1 et valeurs arrondies à trois décimales. |
| `02_matrice_tsac.png` | Catégorie réelle en ligne, prédiction en colonne ; brut et normalisé côte à côte, effectifs et macro-F1 exact à six décimales. |
| `03_matrice_madar.png` | Même principe pour MSA/TN/EG ; échelle de couleur commune aux deux variantes d'une même figure. |
| `04_classes.png` | Précision, rappel et F1 de chaque catégorie du TF-IDF brut, avec effectifs. |
| `05_ecritures.png` | Macro-F1 par écriture du TF-IDF brut, avec effectifs ; orange si moins de 30 exemples ou une catégorie absente. Écriture ne signifie pas langue. |
| `06_durees.png` | Temps d'ajustement et de prédiction sauvegardés pour les variantes TF-IDF. Ce ne sont pas les temps de la relecture ni une latence applicative. |

`summary.json` conserve les agrégats numériques non arrondis, la source et sa date, les empreintes des résultats, du générateur et des PNG, la version Matplotlib, le mode et le statut de rechargement. Aucun texte de corpus, vocabulaire appris ou identifiant privé de ligne n'est exporté dans ce résumé ou dans les images. Les sorties restent ignorées par Git.

## Vérifications exécutées

- Run complet caractères v2 historique rechargé ; six résultats contrôlés. Export final : `outputs/figures_validation_2026-10-27_v2/`. Les durées sont celles du run du 17 octobre, pas celles du 27 octobre.
- Collecte et recalcul contrôlés aussi sur les runs mots, combinés et le full caractères issu du notebook du 25 octobre (`d99058b90e8447fa9cc676ffe1a472d3_characters_full`). Aucun nouvel apprentissage.
- Nouveau notebook propre, structure compilée, cinq cellules de code exécutées dans le noyau Jupyter local `tuniscope-notebook` ; six messages d'affichage PNG reçus. Le notebook source reste sans sorties. Ce contrôle de noyau n'est pas une inspection de l'interface VS Code ni une répétition de la soutenance par nous.
- Douze nouveaux tests fictifs réussis : arithmétique, refus test/v1, intégrité, alignement, absence de données individuelles exportées, chemins, mode smoke, absence d'écrasement et rechargement explicitement demandé.
- Suite générale : 113 tests exécutés, 112 réussis, un échec structurel préexistant car les notebooks 01/02 conservent leurs sorties interactives. Ces sorties ont été préservées ; ne pas annoncer une suite générale entièrement verte. Une copie propre avant diffusion reste à préparer explicitement.
- Figures inspectées : légende de la vue par classe ajustée pour ne pas masquer les valeurs. L'environnement d'entraînement et les anciens runs ne sont pas modifiés.

L'installation réseau a d'abord échoué dans l'environnement restreint, puis a réussi via l'autorisation d'exécution appropriée. Le premier démarrage de noyau a également rencontré une restriction d'ouverture de port local ; la vérification a réussi dans le contexte autorisé. Il ne s'agit pas d'erreurs des modèles ni de changements des scores.

## Limites scientifiques

Les fichiers actuels ne contiennent **pas d'historique de loss ou d'accuracy par époque** pour ces classifieurs classiques. Nous ne construisons donc pas de courbe d'apprentissage par époque à partir des six systèmes. Les futurs modèles neuronaux devront enregistrer explicitement un historique avant de le tracer.

Les graphiques ne prouvent ni un gain statistiquement significatif, ni la généralisation hors domaine, ni une reproductibilité sur installation vierge ou cloud. Les barres arrondies peuvent masquer les très petits écarts brut/normalisé ; les matrices et `summary.json` en conservent le détail. Les trois graines, le bootstrap, le test officiel et la démonstration applicative restent hors du périmètre de cette étape.
