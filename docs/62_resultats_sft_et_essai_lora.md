# Résultats du corpus SFT et de l’essai LoRA

**7 novembre 2026.** Le corpus supervisé de 4 000 exemples a été construit puis reproduit à l’identique à partir du train v2. Les empreintes de la sélection, du tokenizer, des consignes et des masques sont enregistrées avec le jeu local. [Protocole SFT](61_protocole_sft_qwen.md)

| Tâche | Exemples | Répartition des labels | Longueur maximale | Tokens supervisés |
| --- | ---: | --- | ---: | ---: |
| TSAC | 2 000 | 1 000 NEG, 1 000 POS | 370 | 4 000 |
| MADAR | 2 000 | 667 MSA, 667 TN, 666 EG | 138 | 4 667 |

La vérification porte sur les 4 000 frontières entre prompt et réponse. Le suffixe supervisé contient `POS` puis `<|im_end|>` dans la sonde TSAC. La sonde MADAR contient `MS`, `A` puis `<|im_end|>`, car le tokenizer découpe le label `MSA` en deux tokens. Les tokens du prompt et du padding sont masqués. Le premier lot réel donne une différence absolue de `1,19 × 10⁻⁷` entre la perte du modèle et l’entropie croisée calculée sur les seules cibles décalées.

## Vérification de l’adaptateur

Un adaptateur partagé met à jour **1 146 880 paramètres** dans les projections `q_proj` et `v_proj`. Les **596 049 920 paramètres** de la base restent gelés. Les quatre mises à jour de l’essai donnent des gradients LoRA non nuls et des pertes finies. L’empreinte de tous les paramètres gelés est identique avant et après. L’empreinte des paramètres LoRA change.

L’export contient 112 tenseurs LoRA. L’adaptateur rechargé avec la révision officielle reproduit exactement les deux continuations inventées et les logits vérifiés. Le premier export a déclenché un avertissement de recherche de configuration de vocabulaire dans PEFT en mode hors ligne. Nous avons fixé l’export des embeddings à `False` et répété l’essai. Le contrôle des clés Safetensors confirme un fichier d’adaptateur seul. Le profil d’inférence et le profil LoRA disposent de verrous distincts.

## Capacité et durée sur Mac

| Lot physique | Accumulation | Mesure par époque estimée | Deux époques estimées | Contrôle des lignes les plus longues |
| ---: | ---: | ---: | ---: | --- |
| 4 | 4 | 16,4 à 17,5 min | 33 à 35 min | Mémoire insuffisante pendant le backward |
| 2 | 8 | 19,8 min | 39,7 min | Forward et backward réussis à 370 et 348 tokens |

Les estimations utilisent 250 mises à jour par époque, chacune portant sur 16 exemples. Le lot de 2 a été chronométré sur les trois mises à jour après chauffe, avec une moyenne de 4,76 secondes. Ce temps estime l’apprentissage. La validation et la sauvegarde de campagne ont leur propre coût. L’essai chronométré contient des séquences allant jusqu’à 144 tokens.

Le contrôle complémentaire utilise les plus longues séquences et une limite MPS conservatrice de **11,84 Gio**, avec seuil haut `1,0` et seuil bas `0,9`. Un premier lancement a été rejeté avant le chargement car le seuil bas hérité dépassait le nouveau seuil haut. Avec les seuils compatibles, le lot de 4 a terminé le forward puis a atteint la limite pendant le backward. Le lot de 2 a terminé les deux opérations. La mémoire du pilote relevée après son backward était de **8,51 Go**. Ces relevés sont des instantanés du pilote MPS.

Le Mac fournit une configuration locale vérifiée avec lot physique de 2. À raison de deux époques par graine, trois graines représenteraient environ deux heures d’apprentissage sur ce profil, auxquelles s’ajoutent les validations. Un essai CUDA identique permet de décider de la plateforme pour la campagne complète.

## Reproduction

```sh
sh scripts/llm_uv.sh run --locked --offline python scripts/build_qwen_sft.py \
  --verify outputs/qwen/sft_v1_2026-11-07

PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 \
  sh scripts/lora_uv.sh run --locked --offline python scripts/run_qwen_lora_smoke.py \
  --device mps --batch-size 2 --dataset outputs/qwen/sft_v1_2026-11-07 \
  --output outputs/qwen/mon_essai_lora
```

Les preuves privées se trouvent dans `outputs/qwen/sft_v1_2026-11-07/`, `outputs/qwen/2026-11-07_lora_smoke_mps_batch2/` et les deux rapports `2026-11-07_lora_longest_batch*.json`. Le corpus SFT, les contrôles de masque et l’essai avec rechargement constituent les preuves obtenues pour L06 et L12. La campagne d’adaptation et la comparaison sur validation constituent les étapes suivantes.
