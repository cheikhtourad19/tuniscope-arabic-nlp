# Référence CAMeLBERT gelé : protocole avant expérience

**4 novembre 2026 — contribution partielle à L05, L07 et L12.** Ce protocole est fixé avant les premiers scores de la tête gelée. Les résultats et limites seront consignés séparément. Le fine-tuning complet et le LLM ne sont pas lancés dans cette étape.

## Question et données

Nous mesurons ce qu'une tête linéaire peut apprendre à partir des représentations d'un encodeur arabe préentraîné **sans modifier cet encodeur**. Ce système est une référence exigée au §8 du cahier des charges. Nous utilisons les textes bruts et les mêmes manifestes v2 que les autres classifieurs : TSAC 10 053/1 774 et MADAR 6 388/800 train/validation. Le chargement exclut le test officiel. Aucun suréchantillonnage, relabellisation, poids de classe ou nouvelle normalisation n'est introduit.

## Source, environnement et représentation

Le checkpoint général [CAMeLBERT Mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix) est fixé à `9be352797bdf28a9ae21e2ae582aaaca7abdb22d`. La fiche du modèle déclare Apache-2.0 ; la licence MIT du code des auteurs est distincte. Le fichier officiel `pytorch_model.bin` compte 438 873 130 octets, SHA-256 `d234e4fe0193e74036b1db72ba9b41a2e4921d8cf0c3c421d1f6a6d8816af729`, d'après les métadonnées LFS de cette révision. Un téléchargement explicite, borné et vérifié le conserve dans `.tools/encoders/`, ignoré par Git. Le cache tokenizer existant reste inchangé. Aucun code distant n'est exécuté ; le chargement est local et utilise la désérialisation limitée aux poids de PyTorch moderne.

Le profil isolé `environments/encoder/` verrouille Python 3.12, PyTorch 2.15.0, Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.5.4, scikit-learn 1.9.1 et Matplotlib 3.10.7. Les anciens environnements restent intacts. Les poids sont chargés en float32 avec `AutoModel`, sans pooler ; les têtes de préentraînement `cls.*` et le pooler préexistants sont ignorés. Toute clé manquante du backbone constitue une erreur. Le nombre de paramètres réellement chargé est mesuré, et non repris d'un nom commercial.

La longueur est **128 sous-mots, CLS/SEP compris**, décidée dans l'audit précédent à partir du train. Le padding dynamique complète seulement le plus long texte d'un batch. Pour chaque texte, nous calculons la moyenne des 768 composantes des sorties token par token, **en excluant le padding avec attention_mask et en incluant CLS/SEP**. Le mode `eval()` désactive le dropout de l'encodeur, `requires_grad=False` et `inference_mode()` interdisent l'apprentissage de ses poids. Les vecteurs sont conservés en float32 dans le dossier privé du run, dans l'ordre exact des identifiants du manifeste ; ce cache est spécifique au checkpoint, à la longueur, au pooling et aux textes bruts.

## Tête apprise et sélection

Une couche linéaire 768→2 ou 768→3, avec biais et sans dropout, est entraînée séparément par tâche. Optimiseur AdamW, learning rate **1e-3**, weight decay **0,01**, batch **64**, au plus **50 époques**, patience **5**. Ce learning rate concerne uniquement une petite tête ; le 2e-5 du cahier concerne le futur fine-tuning du backbone et n'est pas interchangeable. Perte cross-entropy non pondérée. L'initialisation et l'ordre des batches dépendent des graines **13, 42, 2026**. La tête s'entraîne sur CPU ; les vecteurs de l'encodeur sont identiques pour les trois graines d'un même run.

À chaque époque : pertes train/validation et macro-F1 de validation. Nous gardons la première époque au meilleur macro-F1 (égalité : époque la plus ancienne), sans sélectionner une meilleure graine. Nous publions les trois scores et moyenne/écart-type échantillonnal. Les résultats du petit essai ne servent pas à choisir les hyperparamètres.

## Faisabilité, sorties et contrôle

Avant un run complet : essai sur 100 lignes par partition et tâche, deux époques et graine 42 ; ce n'est pas un score comparable aux expériences complètes. Le dispositif essaie le batch 16, relève temps, débit et mémoire observés. Détection automatique CUDA→MPS→CPU, sans repli silencieux après une erreur mémoire ; chaque run consigne son matériel. Un essai MPS n'atteste pas un essai CUDA, et un encodeur gelé ne permet pas de garantir la mémoire du fine-tuning avec gradients.

Chaque nouveau run privé contient configuration/provenance, vecteurs et métadonnées d'alignement, têtes, mappings, prédictions sans textes, courbes, métriques par classe et écriture. Le contrôle rejoue les têtes sur tous les vecteurs, recalcule les métriques et réencode dix exemples de validation par tâche depuis le checkpoint original, à tolérance absolue **1e-4**, pour détecter aussi les incohérences du cache. Il vérifie les empreintes du backbone avant/après extraction, l'absence de paramètres entraînables dans celui-ci, les groupes disjoints et les effectifs. Une reproduction indépendante et un passage Kaggle restent des preuves distinctes.

## Limites de la comparaison future

Cette référence utilise le pooling moyen prescrit. Le §9 prévoit `AutoModelForSequenceClassification` pour le futur fine-tuning ; sa tête BERT habituelle utilise CLS/pooler. Si nous changeons simultanément pooling et tête, la différence ne mesurera pas uniquement le dégel de l'encodeur. Il faudra conserver l'architecture pour une ablation contrôlée ou déclarer explicitement cette différence et ajouter le témoin nécessaire. Ni les poids BiGRU ni ses vecteurs ne sont transférés ici. Les meilleurs modèles de l'application ne sont pas remplacés automatiquement.
