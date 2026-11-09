# Protocole de recherche documentaire

Le module documentaire répond aux questions sur la méthodologie et le périmètre de TuniScope. Le prototype extrait un passage et en indique la fiche. La classification de sentiment et de variété reste une fonction distincte.

## Fiches et questions

La version 1 comprend 22 sujets documentés, chacun décliné en français et en arabe standard. Les 44 fiches sont des résumés originaux de documents du projet. Elles contiennent un identifiant stable, une langue, une version, un titre, un passage, un auteur et des références. Les fichiers bruts TSAC et MADAR ne servent pas d’index documentaire.

Le jeu comprend 20 questions de validation et 40 questions de test documentaire. Chaque partition est équilibrée entre les deux langues. Douze questions ont une liste de références vide, quatre en validation et huit en test. Les références pertinentes correspondent aux fiches dans la langue de la question. La pertinence vise une réponse substantielle à la demande. Une fiche simplement liée au sujet peut être insuffisante. Les annotations sont des propositions assistées par IA. La validation éditoriale et la double relecture humaine d’un échantillon sont à enregistrer avant l’évaluation définitive.

## Comparaison sur validation

Les deux méthodes classent les mêmes fiches dans la langue indiquée. Elles indexent le titre et le passage. BM25 utilise des mots, avec NFC, casse uniforme et une harmonisation des signes diacritiques réservée à la recherche. Ses paramètres sont k1 = 1,2 et b = 0,75. La recherche dense utilise l’encodeur multilingue E5 small épinglé, la moyenne des vecteurs de tokens non masqués et une normalisation L2. Les préfixes `query: ` et `passage: ` suivent la fiche du modèle, y compris pour l’arabe. Le classement dense utilise le cosinus. Une égalité est départagée par identifiant. [Fiche officielle E5](https://huggingface.co/intfloat/multilingual-e5-small/tree/fd1525a9fd15316a2d503bf26ab031a61d056e98)

Sur les questions avec réponse, Recall@3 est la fraction moyenne des fiches pertinentes retrouvées dans les trois premiers rangs. MRR@3 est la moyenne de l’inverse du rang de la première fiche pertinente parmi ces trois rangs, avec zéro si aucune ne remonte. Les questions sans réponse servent à mesurer le rejet. Chaque méthode a son seuil de premier score, choisi sur validation en maximisant la moyenne du taux d’acceptation des questions avec réponse et du taux de rejet des questions sans réponse. En cas d’égalité, le meilleur rejet puis le seuil le plus élevé sont préférés. Un score BM25 nul déclenche toujours une abstention. Nous mesurons aussi la fraction des réponses émises dont la première fiche est pertinente. Les chiffres de seuil et de réponses sur validation sont exploratoires.

La réponse extractive reprend intégralement le passage de la première fiche et cite son identifiant. Pour le contraste génératif, Qwen de base reçoit la même question et la même consigne avec trois passages ou un contexte vide. Les références dans la réponse sont contrôlées par le code. La fidélité factuelle demande une relecture humaine. La version extractive reste disponible pour les démonstrations.

Le test documentaire et le test de classification sont deux jeux différents. La première itération mesure uniquement la validation documentaire.
