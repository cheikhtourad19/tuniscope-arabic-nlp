# Modèles de langage caractères — validation v2

Run : `2026-11-04_char_lm_v1_full` ; mode **full**, représentation **raw-v0**.

| Corpus | Ordre | Train | Validation | Cibles | Vocabulaire | Inconnus | Perplexité ↓ | Bits/cible ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TSAC | 1 | 10053 | 1774 | 71924 | 381 | 0.0385% | 41.790861 | 5.385116 |
| TSAC | 3 | 10053 | 1774 | 71924 | 381 | 0.0385% | 11.128590 | 3.476199 |
| MADAR | 1 | 6388 | 800 | 26297 | 61 | 0.0078% | 23.802216 | 4.573024 |
| MADAR | 3 | 6388 | 800 | 26297 | 61 | 0.0078% | 9.222734 | 3.205195 |

Même vocabulaire et mêmes cibles entre les deux ordres au sein de chaque corpus. EOS inclus, BOS exclu ; lissage alpha=0,1. Les perplexités des deux corpus ne sont pas directement comparables. Ce ne sont pas des scores de classification. Aucun test officiel évalué.

Comptes reconstruits depuis le train, checkpoints rechargés et validation intégralement rejouée ; probabilités et agrégations recalculées indépendamment. L04/L07/L12 restent partiels.
