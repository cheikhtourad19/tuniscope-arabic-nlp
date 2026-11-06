# Résultats de la confirmation encodeur

**6 novembre 2026 — L05, L07, L08 et L12.** La confirmation de validation des réglages retenus est terminée. Les six modèles ont été entraînés depuis le notebook 08 sur le Mac/MPS, avec les taux fixés par le protocole 55. Aucun modèle supplémentaire n’a été entraîné pour cette vérification.

## Résultats individuels et dispersion

Les tableaux sont générés par `scripts/report_encoder_confirmation.py` depuis les artefacts contrôlés dans `reports/camelbert_confirmation_2026-11-06/`. Aucun score n’est saisi manuellement dans le rapport généré.

| Tâche | Taux | Graine | Macro-F1 | Accuracy | Époque retenue / exécutées |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSAC | 5e-5 | 13 | 0,933368 | 0,933484 | 3 / 3 |
| TSAC | 5e-5 | 42 | 0,930285 | 0,930665 | 3 / 3 |
| TSAC | 5e-5 | 2026 | 0,928850 | 0,928974 | 3 / 3 |
| MADAR | 2e-5 | 13 | 0,954004 | 0,956250 | 3 / 3 |
| MADAR | 2e-5 | 42 | 0,954370 | 0,957500 | 3 / 3 |
| MADAR | 2e-5 | 2026 | 0,952071 | 0,953750 | 3 / 3 |

| Tâche | Macro-F1 moyen ± écart-type échantillonnal | TF-IDF caractères brut | Différence de macro-F1 |
| --- | ---: | ---: | ---: |
| TSAC | **0,930834 ± 0,002308** | 0,928898 | +0,001936 |
| MADAR | **0,953482 ± 0,001235** | 0,949244 | +0,004238 |

La comparaison porte sur les mêmes identifiants, groupes et labels de validation : 1 774 exemples TSAC et 800 exemples MADAR par modèle. Les six modèles utilisent tout le train prévu, soit 10 053 lignes TSAC et 6 388 MADAR. Le test officiel n’est pas sélectionné.

Les gains moyens par rapport à TF-IDF sont modestes : environ **0,19 point** et **0,42 point** de macro-F1 sur une échelle de 0 à 100. Aucun intervalle de confiance ni gain significatif sur le test n’est établi. L’écart-type décrit uniquement la dispersion entre nos trois entraînements.

## Interprétation et limites

Le score exploratoire local TSAC à 5e-5 était de 0,936585 sur la graine 42 ; la moyenne de confirmation est inférieure. C’est un résultat conservé conformément au protocole, sans nouvelle sélection du taux après lecture des trois graines. La validation a déjà servi au choix du réglage et de l’époque : elle ne constitue pas une mesure finale indépendante.

La campagne initiale à 2e-5 rapportait une moyenne TSAC de 0,935802 et une moyenne MADAR de 0,950815 (document 47). La nouvelle confirmation ne montre donc pas une amélioration uniforme par rapport à cette expérience antérieure. Les campagnes restent distinctes. Le déterminisme MPS est limité pour certaines opérations ; la réexécution d’une graine ne garantit pas un score identique, et aucune cause précise des différences n’est établie ici. Nous n’utilisons pas le meilleur résultat historique pour remplacer la moyenne de confirmation.

Dans les figures TSAC/13 et MADAR/42 inspectées, la perte train baisse tandis que la perte validation remonte en fin d’apprentissage ; le macro-F1 maximal reste à l’époque 3. Ce comportement n’est pas une contradiction : la perte utilise les probabilités, tandis que le macro-F1 utilise les labels prédits. Le choix de checkpoint suit le critère macro-F1 annoncé. Ces courbes justifient la prudence sur les probabilités non calibrées ; elles ne prouvent pas à elles seules un défaut de calibration quantifié.

## Preuves d’exécution et de rechargement

Campagne conservée :

```text
outputs/encoder_finetuning/confirmation/notebooks/177d9f5ad78b4ef4907e97d97abe96f8_full/
```

SHA-256 de `confirmation.json` : `a645ae3b78236c2c6c5f6c83b5ef56f8d8cdd4c32cd2a837b2a4fb57ed0886ef`.

- Six nouveaux modèles adaptés, graines 13/42/2026 pour chaque tâche, trois époques chacun : **18 époques FULL** conservées avec métriques par classe et prédictions.
- Deux smokes appariés distincts : quatre modèles et huit époques techniques supplémentaires, exclus des moyennes FULL.
- Six checkpoints FULL rechargés par le pipeline sur toute leur validation : **7 722 prédictions**, différence maximale des probabilités **0**, avec contrôles de gradients et de modification des poids.
- Contrôle ultérieur des quatre stages : empreintes, sources, partitions, prédictions, métriques, taux, graines et agrégats concordants. Ce contrôle exploite les preuves de rechargement enregistrées ; il ne réalise pas une nouvelle inférence.
- Six PNG liés aux résultats par empreinte. Les six images du notebook sauvegardé sont identiques aux fichiers PNG de la campagne. Un rejeu dans un noyau Jupyter isolé a exécuté quatre cellules et affiché six images, sans entraînement ni nouvelle inférence.
- Deux cartes provisoires cohérentes avec configurations, trois références de checkpoints et agrégats ; poids, tokenizer et mappings conservés. Paramètres totaux et entraînables : **109 082 882** pour chaque modèle TSAC et **109 083 651** pour chaque modèle MADAR, sous le plafond de 1 milliard par modèle.
- Archive `evidence_without_weights.zip` : 134 membres vérifiés, sans poids. Elle ne remplace pas la sauvegarde des checkpoints. Une sauvegarde indépendante sur un autre support reste à réaliser ou à confirmer.

Les sorties personnelles du notebook sont préservées dans le fichier actif et une copie privée. Preuve du rejeu Jupyter : `outputs/notebook_checks/2026-11-06/confirmation_native_full_replay.json`.

## Coût observé

| Périmètre | Durée |
| --- | ---: |
| Trois modèles FULL TSAC | 1 885,292019 s — 31 min 25 s |
| Trois modèles FULL MADAR | 892,069269 s — 14 min 52 s |
| Somme des stages FULL | 2 777,361288 s — **46 min 17 s** |
| Campagne avant contrôles finaux | 2 884,380476 s — **48 min 04 s** |

Les stages FULL incluent chargement, apprentissage, validation et sauvegarde. La durée de campagne inclut également les smokes, les appels séparés de rechargement et les figures ; elle s’arrête avant les contrôles finaux de campagne, la synthèse et l’archive. Ce n’est pas une mesure absolue du temps mural de toutes les cellules du notebook, ni un benchmark répété.

## Reproduction du rapport

Depuis la racine, avec les sources autorisées et les runs locaux conservés, choisir un dossier de rapport neuf :

```sh
sh scripts/project_uv.sh run --locked --offline python scripts/report_encoder_confirmation.py \
  outputs/encoder_finetuning/confirmation/notebooks/177d9f5ad78b4ef4907e97d97abe96f8_full \
  --output reports/ma_relecture_confirmation
```

Cette commande vérifie la campagne et les références TF-IDF, puis produit un tableau lisible et un JSON agrégé sans textes du corpus. Elle ne lance aucun entraînement. Pour afficher les six figures, utiliser notebook 08 en `replay`, avec ce dossier de campagne réel et le mode attendu `full` ; ne pas conserver `full` comme mode d’action si l’objectif est seulement de revoir les résultats.

## État des livrables

La preuve d’entraînement et de validation sur trois graines de la configuration sélectionnée est obtenue pour **L05**. Les courbes, prédictions et cartes contribuent à **L07/L08** ; le notebook, les empreintes et le rapport généré contribuent à **L12**. Cela ne clôture pas l’ensemble de ces livrables : test final, bootstrap, analyses complémentaires, intégration des nouveaux modèles, sauvegarde indépendante et reproduction complète par un autre opérateur restent distincts. Les sources de données, configurations d’apprentissage, checkpoints historiques et modèles de l’application n’ont pas été modifiés par ce contrôle. La phase LLM/LoRA n’est pas encore démarrée.
