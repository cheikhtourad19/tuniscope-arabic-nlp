# TuniScope — Première étude linguistique assistée sur le train

**Livrables :** L04, L10 et L12, contributions partielles. Cinquante textes documentés par IA, sans validation humaine indépendante.

## Périmètre

Nous étudions 25 commentaires TSAC et 25 phrases MSA MADAR issus exclusivement du train v2. Les quotas enrichissent les scripts et marqueurs candidats ; le lot n'est pas représentatif des corpus. Le sélecteur ne lit aucun score, prédiction ou texte de validation/test. Le snapshot de sélection conserve son état initial sans annotation ; un registre distinct contient les 50 propositions IA.

## Observations proposées

| Corpus | Textes | Négation proposée présente | Absente | Incertaine | Candidats avec négation proposée absente | Sans candidat avec négation proposée présente |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSAC | 25 | 15 | 10 | 0 | 1 | 4 |
| MADAR | 25 | 8 | 17 | 0 | 1 | 1 |

Ces comptes décrivent des propositions IA à relire. Ils ne constituent ni une mesure de précision/rappel du détecteur, ni une annotation de référence, ni une prévalence de négation dans le corpus.

Nous relevons des négations grammaticales, des réponses négatives isolées, des constructions privatives, des exceptions et des locutions non négatives malgré un marqueur apparent. Une négation peut servir un éloge, porter sur une autre cible ou rester dans une proposition subordonnée. La présence de négation n'impose pas un label NEG.

Plusieurs scripts MIXED sont dus à des restes techniques latins et non à un changement de langue. Des adjectifs peuvent décrire une quantité ou une taille, sans exprimer un avis. Les questions et impératifs du corpus dialectal ne reçoivent pas de nouveaux labels de sentiment.

## Contrôles et suite

Le rendu vérifie la sélection contre les sources/manifestes et les empreintes, les 50 identifiants, les extraits exacts, les catégories autorisées et les champs d'analyse. Ces contrôles de cohérence ne certifient pas la justesse linguistique des propositions. Les textes et notes détaillées restent privés.

La relecture humaine, l'évaluation d'une règle de portée locale, la baseline à règles et les notebooks restent à faire. Le lexique éventuellement induit doit provenir du train uniquement. Aucun label, modèle, score, manifeste ou corpus d'origine n'a été modifié.
