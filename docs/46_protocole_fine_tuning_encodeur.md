# Fine-tuning CAMeLBERT et témoin CLS gelé : protocole initial

**5 novembre 2026 — L05, L07, L12 partiels.** Nous démarrons l'adaptation de l'encodeur après la référence gelée. Ce protocole et `configs/camelbert_finetuning_v1.json` sont fixés avant les nouveaux scores. Aucun poids LLM, LoRA ou nouveau test officiel n'est concerné.

## Comparaison contrôlée

Le cahier (§9–10) prévoit `AutoModelForSequenceClassification`. Pour BERT, cette classe utilise CLS → pooler dense/tanh préentraîné → dropout → tête linéaire. La précédente référence prescrite au §8 utilise une moyenne masquée et reste conservée : elle ne suffit pas, seule, à isoler l'effet du dégel face à une tête CLS différente.

Nous ajoutons donc un **témoin CLS gelé** et son pendant **CLS fine-tuné**, construits avec la même classe, la même base épinglée, les mêmes données, initialisations de tête, longueur, ordre des lots et hyperparamètres. Dans le témoin, tout `model.bert`, pooler compris, est gelé ; seule `classifier` apprend. Dans le fine-tuning, tous les paramètres apprennent. Les trois paires de graines sont rapportées sans sélection d'une meilleure graine.

Pour contrôler aussi le dropout, `model.bert.eval()` est maintenu dans **les deux régimes**, même pendant le fine-tuning ; cela désactive le dropout du backbone mais **n'interdit pas les gradients**. Le dropout de la tête (0,1) est actif pendant l'apprentissage des deux régimes. La validation utilise `eval()` et `inference_mode()` sur tout le modèle. Les poids et la configuration du backbone ne sont pas remplacés par ceux de la BiGRU ou d'une autre expérience.

Le témoin CLS est volontairement un contraste à **budget et optimiseur identiques**, pas une recherche du meilleur classifieur gelé possible. Son learning rate faible et ses trois époques peuvent pénaliser une tête initialisée aléatoirement ; nous conservons donc également la référence mean-pooling/tête longuement entraînée du document45. Nous distinguons ce contraste expérimental de la comparaison globale entre systèmes optimisés différemment. L'arrêt anticipé peut produire des nombres d'époques différents malgré le même plafond et la même règle.

## Réglages initiaux

| Élément | Choix avant mesures |
| --- | --- |
| Source | CAMeLBERT Mix général, révision9be352797bdf28a9ae21e2ae582aaaca7abdb22d ; poids officiels déjà vérifiés |
| Entrées | Texte raw-v0 ; manifestes v2 ; train/validation uniquement |
| Labels | TSAC NEG/POS ; MADAR MSA/TN/EG ; deux modèles indépendants |
| Longueur | 128 sous-mots CLS/SEP compris ; padding dynamique |
| Précision et attention | float32, eager ; pas de quantification ni AMP implicite |
| Lots | 8 physiques × accumulation2 = 16 effectifs, dernière fenêtre partielle pondérée par son effectif réel |
| Optimisation | AdamW, learning rate2e-5 constant, weight decay0,01 ; tous les paramètres entraînables concernés, clippingnorme1 |
| Perte | Cross-entropy non pondérée ; labels inchangés |
| Époques | Maximum3, patience2 ; meilleur macro-F1 de validation, première époque en cas d'égalité |
| Graines | 13,42,2026 ; smoke42 et deux époques |
| Matériel | Détection CUDA→MPS→CPU, aucune substitution silencieuse en cas d'erreur |

Les valeurs1e-5 et5e-5 constituent des candidats éventuels pour une recherche ultérieure limitée à trois learning rates ; elles ne sont pas annoncées comme testées. Toute nouvelle recherche sera enregistrée avant l'ouverture du test. La longueur128 est issue de l'audit précédent, pas choisie à partir de scores du test.

## Essai, mesure et décision de calcul

Un essai utilise100 lignes par partition et corpus, tri SHA256 des identifiants, les deux régimes et la graine42. Chaque exemple train de ce mode est utilisé ; la validation reste disjointe par groupes. Le smoke doit vérifier pertes finies, gradients non nuls de la tête, gradients du Transformer seulement en régime adapté et changement de ses poids. Le témoin doit conserver exactement son backbone. Un batch inventé de8 entrées de128 positions vérifie aussi la longueur maximale ; il ne produit pas de score linguistique.

Nous relevons temps d'époque, débit, picRSS et allocations d'appareil observées. Sur MPS, une valeur échantillonnée après backward/step n'est pas un pic absolu garanti. Les mesures du petit essai servent à **estimer** le budget de trois graines sur l'ensemble des données ; l'estimation reste distincte d'un run complet. L'essai ne suffit pas à prouver la compatibilité Kaggle/CUDA : le verrou Linux actuel inclut CUDA13, dont le driver réel devra être contrôlé.

Le mode FULL exige un smoke vérifié avec la même configuration et le même appareil. Il traite tout le train et toute la validation, graines13/42/2026 ; `paired` exécute les deux régimes (12 modèles), `finetuned` seulement les six modèles adaptés et `frozen` les six témoins. Les modes et effectifs sont imprimés et sauvegardés sans repli vers smoke. Les sorties sont écrites dans un nouveau dossier privé ; les anciens runs sont conservés.

**Correction technique après le premier essai MPS, avant tout score adapté :** `embedding_dense_backward_mps` ne possède pas d'implémentation déterministe stricte dans ce runtime. Le premier essai s'arrête explicitement au premier backward adapté ; son dossier partiel est conservé. Nous gardons les graines et les algorithmes déterministes disponibles, mais utilisons `warn_only=True` sur MPS. CPU/CUDA restent stricts. Aucune répétabilité bit à bit MPS n'est revendiquée et ce réglage est enregistré dans la configuration. Cette correction ne change ni données, ni learning rate, ni architecture ; le smoke est recommencé dans un nouveau dossier.

## Artefacts et contrôles

Chaque modèle retenu fournit configuration et mappings, tokenizer, historique des époques, prédictions sans textes et métriques. Un checkpoint adapté contient tous les poids au format safetensors ; un témoin gelé contient seulement sa tête et la référence exacte de la base locale. Les sauvegardes volumineuses restent ignorées par Git. Sauvegarde du meilleur checkpoint pendant l'apprentissage et signalement visible des erreurs ; la reprise exacte d'un optimiseur interrompu n'est pas revendiquée.

Le vérificateur recharge les modèles, rejoue toute la validation sélectionnée, recalcule métriques et matrices, contrôle l'ordre des identifiants et les empreintes, et compare explicitement dix sorties à tolérance1e-4 ainsi que tous les labels prédits. Les courbes PNG sont dérivées des historiques du même run. Un script d'inférence utilise les poids sauvegardés sans entraîner ni accéder au corpus.

L05 ne sera pas déclaré terminé par la création des scripts ou le seul smoke : il faudra les runs complets retenus, courbes, rechargements, analyses et cartes. Le test final, le bootstrap et l'étude de robustesse restent des étapes distinctes de L07.

## Références techniques

- [Hugging Face — BERT](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert) : configuration, pooler et classification de séquence.
- [PyTorch — Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) : distinction entre mode évaluation et désactivation des gradients.
