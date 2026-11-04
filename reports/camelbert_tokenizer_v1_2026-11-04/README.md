# Audit du tokenizer CAMeLBERT Mix

Révision : `9be352797bdf28a9ae21e2ae582aaaca7abdb22d`. Texte raw-v0, manifestes v2.

```text
CAMeLBERT Mix — audit du tokenizer, sans poids ni entraînement
Révision : 9be352797bdf28a9ae21e2ae582aaaca7abdb22d
Texte brut ; train/validation v2, test officiel non sélectionné.
Corpus  Split       Écriture        N   Sous-mots/champ   UNK tokens    P99    Max
TSAC    train       AR           4426              1.58       0.49%     77    266
TSAC    train       LATIN        5355              3.88       0.06%     88    388
TSAC    train       MIXED         259              1.87       1.93%    113    189
TSAC    train       UNKNOWN        13              3.23       5.63%     32     32
TSAC    validation  AR            785              1.57       0.69%     94    176
TSAC    validation  LATIN         932              3.89       0.06%     81    140
TSAC    validation  MIXED          46              2.05       2.12%     54     54
TSAC    validation  UNKNOWN        11              4.18       2.17%     14     14
MADAR   train       AR           6387              1.55       0.42%     32     57
MADAR   train       MIXED           1              1.25       0.00%     22     22
MADAR   validation  AR            800              1.53       0.77%     29     48

Longueur initiale proposée : 128 tokens, CLS/SEP compris.
Critère 99 % satisfait sur les groupes train retenus : True
UNK, découpage et longueur ne mesurent pas la qualité de classification.
```

L'audit complet a été rejoué depuis les fichiers locaux et manifestes vérifiés. Les petits groupes restent signalés ; aucune ligne protégée n'est publiée. Les seuls textes visibles dans summary.json sont des probes explicitement inventés. Les paramètres sont calculés depuis la configuration, sans chargement des poids.
