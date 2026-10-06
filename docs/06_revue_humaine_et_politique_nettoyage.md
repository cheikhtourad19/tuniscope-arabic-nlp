# TuniScope — Revue humaine et politique de nettoyage proposée

**Livrables concernés :** L02 (traçabilité des décisions, contrôles des doublons) et L03 (future préparation reproductible des données), contributions partielles.

**Date :** 6 octobre 2026. **État :** revue enregistrée et règles proposées ; aucune exclusion, correction d'étiquette ou nouvelle partition n'a encore été exécutée.

## 1. Ce qui a été observé et décidé

L'[audit exact](04_data_audit_groups_and_duplicates.md) a trouvé 316 groupes TSAC présents à la fois dans le train et le test, 13 groupes à étiquettes POS/NEG contradictoires et quatre chaînes MADAR identiques dans des partitions différentes. Le [dépistage de similarité](05_data_audit_near_duplicate_screen.md) a produit 124 paires TSAC train–test non identiques au caractère près.

Nous avons renseigné 12 jugements de polarité dans un classeur privé : six POS et six NEG. Le treizième cas, no 325, ne contient qu'un signe de ponctuation isolé ; nous le classons comme **texte non linguistique invalide**, et non POS ou NEG. Après examen visuel, nous jugeons les 124 paires quasi identiques **équivalentes en sens** malgré leurs variations orthographiques. Ces décisions sont liées aux index privés par leurs empreintes SHA-256, consignées dans `data/interim/human_review_2026-10-06.md`, exclu de Git. Nous avons déclaré la décision sur les 124 paires en bloc : leurs cellules Excel n'ont pas été remplies individuellement.

Cette revue ne mesure pas l'accord interannotateurs : nous avons jugé les cas seuls, avec assistance de Codex pour l'enregistrement et les vérifications mécaniques. L'équivalence de sens ne valide pas automatiquement les étiquettes POS/NEG d'origine. Deux paires proches, nos 54 et 57, comportent des combinaisons possibles d'étiquettes opposées ; elles restent des anomalies d'étiquetage à tracer.

## 2. Règles proposées pour une future copie de travail

Les règles suivantes seront implémentées seulement après validation de la politique et test sur un petit échantillon. Les fichiers `data/raw/` et les partitions officielles ne seront jamais modifiés.

| Situation | Action proposée dans la copie de travail | Preuve à conserver |
| --- | --- | --- |
| Ligne TSAC vide ou texte non linguistique dans le train | Exclure de l'apprentissage ; consigner le motif. Le cas 325 est identifié comme invalide. | Fichier et numéro de ligne, code de motif. |
| Texte TSAC exact présent dans train et test | Exclure **toutes** ses occurrences du train de travail ; conserver le test officiel inchangé. | Identifiants des lignes exclues et groupe exact. |
| Texte TSAC identique après harmonisation Unicode et des espaces, présent dans train et test | Appliquer la même règle de décontamination du train ; garder le texte brut et la version de la clé de comparaison. | Version de la clé, identifiants et motif distinct de l'égalité brute. |
| Paire TSAC quasi identique que nous jugeons équivalente en sens | Exclure du train de travail les occurrences de la forme train liée au test. Utiliser l'union des lignes, car plusieurs paires peuvent désigner le même texte. | Identifiants des paires revues, lignes réellement exclues, décision humaine et motif. |
| Doublons exacts TSAC limités au train et portant le même label | Après décontamination train–test, garder une occurrence déterministe par texte, sans modifier la source. | Groupe, représentant choisi et lignes non retenues. |
| Conflit POS/NEG TSAC limité au train | Après décontamination, utiliser le jugement humain seulement si le texte reste dans le train et si la décision est POS ou NEG ; ne garder qu'un représentant marqué comme revue par une seule personne. Écarter les cas invalides ou ambigus. | Labels originaux, décision humaine, statut `single_reviewed` et lignes concernées. |
| Conflit touchant le test officiel | Ne pas réétiqueter le test ; retirer ses équivalents du train de travail, puis signaler l'anomalie dans l'interprétation des scores. | Identifiants et nature de l'anomalie ; score sur le test officiel séparé d'éventuelles analyses diagnostiques. |
| Texte MADAR identique entre train et validation/test | Proposition : écarter du train de travail **tout le groupe de phrase source** associé, soit les quatre variétés, afin de préserver la cohérence des traductions parallèles. Conserver validation et test officiels. | Identifiant de groupe MADAR et quatre lignes exclues ; vérifier les quatre cas avant exécution. |

La priorité des règles évite le double comptage : décontaminer d'abord le train vis-à-vis du test, puis dédupliquer ce qui reste dans le train, puis créer la validation depuis le train restant. Une ligne détectée par plusieurs règles n'est exclue qu'une fois, mais tous ses motifs sont journalisés. Le choix du représentant est stable et documenté, par exemple le plus petit couple `(nom de fichier, numéro de ligne)`. Aucun label source ne sera modifié en place.

## 3. Vérifications avant de déclarer les données prêtes

1. Rechercher les quasi-doublons à l'intérieur du train avant de créer la validation ; les variantes liées doivent rester dans un même groupe. Les 124 paires revues ne couvrent que le recouvrement train–test.
2. Examiner la limite du filtre sur les 139 formes test de moins de huit caractères ; elles ont été incluses dans l'égalité exacte mais pas dans le dépistage de proximité.
3. Confirmer la règle pour les quatre croisements MADAR, puis fixer la graine et produire les manifestes train/validation/test avec identifiants stables, groupes, provenance, motifs d'exclusion et effectifs réels.
4. Refaire les contrôles de fuite sur les manifestes produits. Vérifier que les fichiers et empreintes du test officiel n'ont pas changé.
5. Remplir les deux cartes de données avec les effectifs **après** préparation. Les effectifs de l'audit brut ne sont pas des effectifs exploitables.

Le cas 325 existe aussi dans le test officiel. Il restera dans le fichier de test source ; son influence pourra être signalée, et une analyse diagnostique l'excluant pourra être présentée **séparément**, avec son propre manifeste et dénominateur. Le résultat principal ne masquera pas cette anomalie par une suppression silencieuse.

Cette politique reste un projet de décision. L02 n'est pas achevé tant que les manifestes, cartes et comptes d'exclusion ne sont pas vérifiés ; L03 n'est pas achevé tant qu'un traitement reproductible n'a pas été exécuté depuis les sources autorisées.
