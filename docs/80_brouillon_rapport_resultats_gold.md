# Rapport scientifique — Brouillon de la section résultats et limites Gold

## Effet de la quantité de données

Nous faisons varier la quantité de données d’entraînement pour le classifieur de sentiment TF-IDF caractères, avec une validation TSAC v2 commune de 1 774 avis. Chaque sous-ensemble est sélectionné par groupes et s’emboîte dans le suivant pour une même graine. Le macro-F1 moyen progresse de 85,02 % à 92,89 % entre 10 % et 100 % du train. Le passage de 50 % à 100 % apporte 1,46 point. Les trois entraînements à 100 % reproduisent les prédictions historiques. L’expérience mesure l’effet des données pour ce système précis et sur ce corpus. [Méthodes et comptes](76_avancement_gold_validation.md)

## Variantes d’écriture, confiance et abstention

Les modifications d’espaces laissent les prédictions identiques pour les modèles testés. Une duplication artificielle de lettre réduit le macro-F1 de 1,12 point pour TF-IDF et de 1,64 point pour CAMeLBERT graine 42. Ces mesures décrivent une perturbation précise. L’ordre des deux modèles s’inverse selon l’écriture d’origine sur TSAC. Les 785 textes en caractères arabes favorisent CAMeLBERT pour la graine étudiée, tandis que les 932 textes en caractères latins favorisent TF-IDF. La catégorie latine recouvre plusieurs usages linguistiques. Les catégories mixtes et indéterminées contiennent respectivement 46 et 11 textes. [Audit par écriture](77_audit_gold_robustesse_et_incertitude.md)

Sur TSAC, l’abstention exploratoire ramène le taux d’erreur parmi les réponses de 6,90 à 5,16 % avec une couverture de 95,00 %. La couverture est cependant inférieure de 6,40 points en écriture latine par rapport à l’écriture arabe. En outre, la mesure de confiance doit être distinguée de la qualité des labels. Avec une calibration par température en validation croisée groupée, l’écart moyen entre confiance et réussite sur TSAC diminue de 4,87 à 1,40 points, tandis que les classes prédites restent identiques. [Mesures de calibration et d’abstention](78_calibration_et_erreurs_lora_gold.md)

## Comparaison et analyse des erreurs

Le bootstrap apparié par groupes indique des différences CAMeLBERT moins TF-IDF de +0,19 point sur TSAC et +0,42 point sur MADAR. Les deux intervalles à 95 % couvrent zéro. Cette comparaison complémentaire sur validation reste insuffisante pour établir un avantage. Le contraste principal sur test concerne Qwen avant et après LoRA. [Bootstrap de validation](77_audit_gold_robustesse_et_incertitude.md)

Le diagnostic de l’adaptateur LoRA graine 42 utilise deux échantillons équilibrés de 144 textes. Les sorties respectent le format de classification sur ces échantillons, mais 34 des 72 avis positifs sont classés négatifs et 29 des 48 phrases tunisiennes sont classées égyptiennes. Les 60 hypothèses qualitatives examinées sont déclarées comme assistance IA et attendent une confirmation indépendante. La taille et la sélection des échantillons limitent toute extrapolation. [Matrices et dossier de revue](78_calibration_et_erreurs_lora_gold.md)

En comparant Qwen de base et ce même adaptateur sur ces 288 textes, avec le même appareil et les mêmes consignes, le macro-F1 progresse de 33,80 % à 67,29 % sur TSAC, et de 0,00 % à 65,10 % sur MADAR. Pour cette dernière tâche, les 144 sorties de la base échouent au contrôle du format. Le gain mélange donc apprentissage du format de sortie et capacité à prédire un label de référence. [Comparaison appariée](78_calibration_et_erreurs_lora_gold.md)

## Limites à compléter après évaluation finale

Les performances présentées ici proviennent de la validation, déjà utilisée pour choisir des réglages. La période et le domaine de TSAC, le domaine du voyage de MADAR et la faible représentation de certaines écritures réduisent la portée des comparaisons. Les exemples de défi introduiront des observations nouvelles, sous réserve de provenance et de validation humaine. Le contrôle du protocole gelé confirme les 36 entrées de modèles et 48 conditions prévues pour la classification finale. Le test officiel permettra d’évaluer ces conditions, le contraste principal et la couverture hors de la validation. Ces résultats et les mesures de calcul et de latence compléteront le rapport. [Protocole final](73_protocole_classification_finale.md)
