# Revue assistée des erreurs de CAMeLBERT

6 novembre 2026. Analyse qualitative assistée par IA à notre demande, sur la sélection de validation documentée en doc50. Les propositions ajoutées ne sont pas une annotation humaine indépendante. Aucun entraînement ni changement des labels ou scores.

| Corpus | Cas documentés | Labels plausibles | Labels discutables | Incertains | Avis humains conservés | Avis IA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSAC | 30 | 24 | 3 | 3 | 0 | 30 |
| MADAR | 30 | 18 | 0 | 12 | 0 | 30 |

## Difficultés proposées

| Corpus | Catégorie proposée | Cas |
| --- | --- | ---: |
| TSAC | Expression inconnue | 6 |
| TSAC | Avis mixte / ambigu | 7 |
| TSAC | Orthographe / Arabizi | 7 |
| TSAC | Manque de contexte | 3 |
| TSAC | Négation | 5 |
| TSAC | Autre | 1 |
| TSAC | Ironie / sarcasme | 1 |
| MADAR | Orthographe / morphologie | 11 |
| MADAR | Variétés proches / texte commun | 11 |
| MADAR | Texte très court | 3 |
| MADAR | Lexique partagé | 3 |
| MADAR | Manque de contexte | 1 |
| MADAR | Autre | 1 |

Ces catégories sont des hypothèses IA sur un échantillon qualitatif stratifié, pas la fréquence des causes dans le corpus ni des explications causales du réseau. Un mot présent dans une phrase ne prouve pas que le modèle l’a utilisé.

Les trois réserves prioritaires TSAC portent sur une lecture littérale favorable ou dépréciative qui semble contredire le label, avec ambiguïté de transcription ou d’ironie. Il ne s’agit pas de trois erreurs d’annotation établies. Les phrases MADAR jugées incertaines manquent souvent de marqueur exclusif : la provenance peut être correcte même si la phrase ressemble au standard ou à une autre variété.

Le classeur original reste intact. La copie assistée, les textes, propositions par cas et le registre d’origine de chaque champ restent Git-ignorés. Seuls ces agrégats sont publiables dans le dépôt. Le compteur indique des cas documentés et ne mesure pas des validations humaines.

Contributions partielles à L07/L10 : 60 cas documentés avec assistance déclarée, vérification technique du classeur et limites explicites. Relecture humaine des propositions, analyse des composants finaux et autres exigences globales restent distinctes.
