# Reproduction bibliographique TSAC : référence et comparabilité

**5 novembre 2026 — préparation de L04, L07 et L10. Aucune reproduction numérique exécutée à ce stade.** Le cahier (§13) demande un résultat publié comparable ou une description précise des limites d'une reproduction partielle.

## Référence retenue pour le cadrage

Mdhaffar et al., 2017, *Sentiment Analysis of Tunisian Dialect: Linguistic Resources and Experiments*, WANLP, pages55–61, DOI10.18653/v1/W17-1307. Le [texte intégral officiel](https://aclanthology.org/W17-1307.pdf), §5 et tableaux5–6, présente Doc2vec avec SVM, BernoulliNB et MLP. Pour MLP entraîné sur TSAC seul, le taux d'erreur publié est0,22, soit environ78% d'accuracy, **pas un macro-F1 mesuré par notre code**. Le tableau5 indique7145POS/6515NEG au train et1700/1700 en évaluation. Le [dépôt des auteurs](https://github.com/fbougares/TSAC) fournit les fichiers du corpus et cite cet article.

La section expérimentale consultée ne précise pas les dimensions/réglages Doc2vec, l'architecture/hyperparamètres MLP et les graines nécessaires à une réplication exacte. Le code d'entraînement n'apparaît pas dans la liste des fichiers du dépôt consulté. Ces absences ne doivent pas être comblées en prétendant retrouver la configuration originale.

## Écarts de notre expérience

La [carte TSAC locale](12_carte_donnees_tsac.md) constate7154POS/6515NEG, soit13669lignes physiques dans le train acquis, dont quatre vides. Le papier en annonce13660 : l'écart de neuf lignes et la différence entre comptage physique et exemples doivent être explicités, sans conclure automatiquement à une faute typographique ou supprimer des lignes pour obtenir le même total.

Notre décontamination et partition v2 conduisent à10053train/1774validation. Notre comparaison actuelle porte sur cette validation ; les3400lignes du test officiel sont réservées à l'évaluation finale. Nos TF-IDF, BiGRU et BERT ne reproduisent pas Doc2vec+MLP. Comparer directement leur macro-F1 de validation aux78% approximatifs du papier serait une comparaison de méthodes, données et métriques différentes, pas un gain bibliographique établi.

## Protocole restant à préparer puis exécuter

1. Consigner ce qui est spécifié par les auteurs et ce qui doit être choisi faute d'information ; ne pas présenter une valeur par défaut d'une bibliothèque récente comme la valeur historique.
2. Définir une reconstruction **partielle** Doc2vec+MLP, avec versions, tokenisation, architecture et graines explicites. Choisir ses hyperparamètres sur notre validation uniquement ; le test ne sert pas à ajuster la reconstruction.
3. Respecter les droits et les fichiers acquis ; conserver les corpus originaux. Toute expérimentation sur split officiel et toute comparaison à notre split décontaminé devront être distinguées.
4. Attendre le gel du protocole final avant l'évaluation sur le test officiel. Publier alors même métrique et effectifs, score obtenu, référence publiée et tableau des différences. Tout accès au test et toute adaptation du benchmark seront déclarés.

Le résultat publié est une **référence documentaire**. Notre statut reste « référence choisie et limites identifiées », pas « baseline reproduite » ni « reproduction partielle réussie ». Aucun modèle supplémentaire, dépendance, lecture du test ou modification des runs CAMeLBERT n'est réalisé par ce cadrage.
