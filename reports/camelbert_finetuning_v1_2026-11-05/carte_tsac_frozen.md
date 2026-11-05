# Carte initiale — TSAC / CAMeLBERT CLS frozen

État : expérience initiale de validation, non modèle final déployé.
- Base : CAMeL-Lab/bert-base-arabic-camelbert-mix, révision9be352797bdf28a9ae21e2ae582aaaca7abdb22d ; Apache-2.0 déclaré dans la fiche officielle.
- Tokenizer WordPiece officiel inchangé, texte brut, longueur128 ; tête CLS/pooler native.
- Tâche et labels : tsac, ['NEG', 'POS'].
- Sources : TSAC avis tunisiens de réseaux sociaux ; MADAR phrases de voyage MSA/Tunis/Sfax/Cairo, selon tâche. Aucune couverture universelle des dialectes.
- Régime : frozen. AdamW2e-5,3époquesmax,patience2,batch8accum2 ; graines13/42/2026.
- Validation macro-F1 : 0.605756 ± 0.003936. Résultats par classe/écriture dans summary.json.
- Rechargements et probabilités vérifiés ; scores softmax non calibrés, aucune garantie pour hors domaine/neutre/français seul.
- Absence de test final, étude exhaustive d’erreurs, calibration et relance indépendante. MPS warn_only ne garantit pas le déterminisme strict.
- Checkpoints privés référencés par résultats/empreintes ; aucun poids ou texte protégé redistribué. Application inchangée.
