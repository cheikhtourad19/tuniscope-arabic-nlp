# Préparation de la revue documentaire

Le 11 novembre 2026, nous avons préparé un dossier de relecture à partir du catalogue RAG v1 et de l’échantillon documentaire déjà publié. Le catalogue comprend 44 fiches, soit 22 sujets en français et en arabe standard. La préparation contrôle les identifiants, les langues, les versions, l’existence des références et la correspondance des références entre les deux langues. Ces contrôles vérifient la structure documentaire. La validité des faits et la fidélité des traductions restent à juger humainement.

## Dossier et procédure

Le script `scripts/prepare_rag_review.py` produit un catalogue lisible regroupé par sujet, avec deux versions linguistiques, des cases de décision et les liens vers les copies des sources. Il fournit aussi deux grilles vierges de 12 questions pour deux lecteurs indépendants. Les grilles demandent les identifiants de toutes les fiches permettant de répondre substantiellement à chaque question, ou « aucune ». Les annotations proposées et les prédictions des moteurs sont absentes des grilles. Le manifeste `integrity.json` enregistre les empreintes des fiches, de l’échantillon, du script, des sources et des fichiers exportés.

Le dossier personnel de cette étape se trouve dans `outputs/rag/revue_v1_2026-11-11/`. Pour produire une autre copie vierge depuis la racine du projet, choisir un nouveau chemin.

```sh
python3 scripts/prepare_rag_review.py --output outputs/rag/ma_revue_documentaire
```

Une destination existante est refusée pour protéger les décisions éventuellement saisies. Les sources copiées et leurs empreintes permettent de retrouver la version effectivement lue.

Nous devons renseigner une décision pour chaque fiche et préciser les corrections proposées. Deux personnes compétentes en français et en arabe standard examinent ensuite séparément le même échantillon de questions. Chacune travaille avant de consulter les annotations proposées, les résultats des moteurs ou le jugement de l’autre lecteur. Les deux grilles originales, les désaccords et leur arbitrage sont conservés. Une seconde lecture de la même personne ou une réponse d’IA ne remplace pas un second avis humain indépendant.

## Conditions avant l’évaluation documentaire finale

Après les revues, toute correction du catalogue ou des annotations doit être versionnée et vérifiée. Si les fiches changent, les index et la calibration liée à leur empreinte doivent être reconstruits. Les réglages se déterminent sur validation, puis sont figés avant l’évaluation des 40 questions de test documentaire.

La version BM25 de l’application a évolué depuis la première expérience du document 67. Elle utilise une règle générale pour les questions courtes et un critère supplémentaire d’écart entre scores pour les questions longues. Les chiffres historiques du document 67 décrivent le moteur et le seuil de cette expérience. Une comparaison de la version applicative finale avec E5 devra identifier exactement le code, les règles d’abstention et les mêmes questions évaluées. Les sondes courtes de mise au point sont des données de développement.

L’évaluation finale devra mesurer la recherche des fiches pertinentes, la validité des citations, la fidélité des réponses et la qualité de l’abstention, conformément au protocole documentaire. Le test de classification reste une étape distincte.

## Livrables et preuves

Cette préparation contribue à **L07**, **L08** et **L12** avec un support de revue traçable et reproductible. Les champs de jugement restent vierges à la génération. La validation humaine des fiches, les deux avis indépendants, l’arbitrage, le gel de la version finale et l’évaluation documentaire restent à réaliser avant la clôture de Silver.

## Réception de la revue des fiches

Le 11 novembre 2026, nous avons confirmé avoir terminé la lecture des fiches. Le formulaire reçu comporte **44 décisions « Conforme »**, sans correction ni incertitude. Le remplacement des cases cochées par leur état vierge reproduit exactement l’empreinte du formulaire initial. Les contenus, identifiants, versions et références sont donc identiques à ceux proposés à la relecture. Le catalogue et les 16 fichiers sources conservent également leurs empreintes initiales.

Une copie du formulaire reçu et un reçu détaillant les décisions par identifiant sont conservés dans le dossier privé `outputs/rag/revue_fiches_recue_2026-11-11/`. Le [bilan agrégé](../reports/rag_card_review_2026-11-11/summary.json) permet de retrouver les effectifs et les empreintes. L’attribution à l’auteur du projet s’appuie sur notre confirmation dans la conversation. Les champs du nom ou pseudonyme et de la date de lecture sont encore vierges dans le formulaire. Ils restent à compléter pour finaliser sa traçabilité. La date de réception ne remplace pas une date de lecture renseignée.

Cette revue éditoriale est distincte de la sélection des fiches pertinentes pour chaque question. Les deux grilles de 12 questions sont toujours vierges à cette réception. Les deux jugements humains indépendants, leur arbitrage et l’évaluation documentaire restent attendus pour L07/L08/L12 et la clôture de Silver.

## Préremplissage assisté des grilles

À notre demande explicite, les deux grilles ont ensuite été préremplies par Codex le 11 novembre 2026, avec 12 propositions de pertinence et leurs justifications dans chaque grille. Elles reprennent les mêmes propositions, dont six questions avec une fiche pertinente et six questions sans réponse dans le catalogue. L’auteur des propositions et le statut de validation humaine en attente sont indiqués dans les formulaires. Les noms fournis désignent leurs destinataires prévus.

Les grilles vierges et la provenance du préremplissage sont conservées sous `outputs/rag/prefill_ia_2026-11-11/`. Ces propositions ne permettent pas de déclarer deux jugements humains indépendants ni de mesurer un accord interévaluateurs. La contrainte de réalisation individuelle peut être présentée à l’enseignant pour demander une adaptation. Son acceptation reste à obtenir avant de considérer l’exigence comme levée. Les autres travaux techniques peuvent se poursuivre en conservant cette limite dans le bilan des livrables.
