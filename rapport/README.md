# TuniScope — Rapport scientifique

Le rapport est rédigé en français. Le fichier `cover.tex` contient la couverture en couleur. `main.tex` définit la composition scientifique en noir et gris pour les autres pages. Le texte est dans `sections/`, les schémas PlantUML dans `figures/` et les références bibliographiques dans `references.tex`.

## Compiler le PDF

Depuis la racine du projet, exécuter cette commande avec `python3`, `latexmk`, `lualatex`, `plantuml`, Graphviz et `rsvg-convert` disponibles dans le terminal.

```sh
make -C rapport
```

Le PDF se trouve dans `outputs/rapport/TuniScope_Rapport.pdf`. Les tables générées, figures intermédiaires et fichiers de compilation restent dans `rapport/build/`. La compilation lit les manifestes sans texte et les rapports agrégés du dépôt. Elle fonctionne sans les corpus bruts, les poids de modèles et les prédictions privées.

## Mettre à jour le rapport

1. Recalculer ou contrôler le rapport JSON de l’expérience. Conserver son protocole, sa partition et les empreintes de ses sources.
2. Pointer `generate_tables.py` vers le rapport agrégé retenu ou ajouter une table pour une nouvelle expérience.
3. Mettre à jour la section concernée et la date de version dans `main.tex`, puis reconstruire le PDF. Relire les nouveaux résultats dans leur contexte.

Pour réimporter le run QLoRA 4B depuis les artefacts privés déjà sauvegardés, lancer depuis la racine du projet

```sh
sh scripts/project_uv.sh run --locked --offline python rapport/import_qlora.py \
  outputs/qwen/qwen3_4b_qlora_full_v1 reports/qwen3_4b_qlora_v1/summary.json
make -C rapport
```

L’import vérifie l’adaptateur, rejoue le calcul des métriques sur les prédictions existantes et enregistre uniquement des agrégats. Une nouvelle campagne doit disposer de son propre rapport et de son identifiant avant de remplacer la référence citée.

La version actuelle contient les résultats de validation de classification et le test documentaire. La révision du protocole 4B, le test officiel de classification, la ressource externe et l’extension Platinum sont les prochaines mises à jour scientifiques.
