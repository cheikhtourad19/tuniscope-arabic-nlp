# Acquisition et premier chargement de Qwen3

**6 novembre 2026 — préparation de L06, L08 et L12.** Périmètre : acquisition explicite de la base officielle et contrôle technique sur entrées inventées, sans lecture des corpus ni apprentissage. À la rédaction initiale, le téléchargement des poids est en cours ; les preuves d’achèvement et de chargement seront ajoutées après vérification.

## Source et version fixées

Nous retenons le modèle [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), conformément au cahier des charges. La [révision épinglée](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) est `c1899de289a04d12100db370d81485cdf75e47ca`. Le dépôt déclare **Apache-2.0** ; les fichiers `LICENSE` et `README.md` sont acquis avec les poids. Aucune redistribution des poids n’est réalisée par le dépôt du projet.

Le plan versionné `configs/qwen3_base_v1.json` enregistre les tailles et empreintes officielles de neuf fichiers : licence, fiche, configuration du modèle, configuration de génération, poids Safetensors, tokenizer et ses fichiers de vocabulaire/fusions. Total prévu : **1 519 207 673 octets**, dont **1 503 300 328 octets** de poids. SHA-256 officiel du fichier `model.safetensors` : `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

Les fichiers LFS sont vérifiés par SHA-256 ; les petits fichiers Git sont vérifiés par leur identifiant de blob, qui inclut l’en-tête Git et le contenu. Un SHA-256 local est aussi enregistré pour chacun. Le script refuse une taille ou une empreinte différente. Les téléchargements incomplets utilisent un suffixe `.part` et ne constituent pas une acquisition validée.

Stockage privé : `.tools/llms/qwen3_0_6b_c1899de289a04d12100db370d81485cdf75e47ca/`. Le fichier local `qwen_source.json` liera les fichiers vérifiés au plan. Poids et cache restent ignorés par Git ; le plan et les scripts sont versionnés.

## Environnement et appareil

Le profil `environments/llm/` est distinct des profils classique, BiGRU, tokenizer, encodeur et application. Il conserve Python 3.12, PyTorch **2.15.0**, Transformers **4.57.6**, Tokenizers **0.22.2** et NumPy **2.5.4**. Les dépendances transitives sont figées par `uv.lock`, exportées dans `requirements.txt`. Les anciens verrous ne sont pas modifiés.

La [documentation officielle Qwen](https://qwen.readthedocs.io/en/latest/inference/transformers.html) et la fiche du modèle indiquent la prise en charge de l’architecture `qwen3` à partir de Transformers 4.51.0. Notre version est compatible en principe ; un chargement réel reste la preuve à obtenir pour ce profil et ce matériel.

Commande d’installation reproductible :

```sh
sh scripts/llm_uv.sh sync --locked
```

Une fois les dépendances en cache, `--offline` peut être ajouté. Les processus lancés par ce wrapper utilisent les fichiers Hugging Face locaux, sans téléchargement implicite. Le téléchargement volontaire des fichiers officiels est réalisé séparément par `fetch_qwen3.py`.

Les mêmes poids, la même révision et le même tokenizer servent sur Mac et CUDA. `auto` choisit CUDA si disponible, puis MPS, puis CPU ; demander explicitement un appareil indisponible provoque un arrêt. La compatibilité CUDA de ce nouveau profil et du chargement Qwen doit être vérifiée sur Kaggle avant de la déclarer acquise. Le téléchargement local n’établit pas cette compatibilité.

## Précision et quantification

Les poids officiels sont stockés en **BF16**, sans conversion en 4 ou 8 bits. Le premier contrôle charge le modèle en **FP32** avec attention eager ; le fichier source reste identique. BF16, FP16 et FP32 désignent des formats de nombres flottants. La quantification basse précision évoquée pour QLoRA constitue un choix différent.

LoRA ajoute de petites matrices entraînables à une base gelée et ne nécessite pas, par définition, une base quantifiée. QLoRA combine adaptation et quantification de la base ; le cahier des charges en fait une variante conditionnelle. Aucune configuration QLoRA, bibliothèque de quantification, création d’adaptateur ou mise à jour de poids n’est réalisée ici. Le choix de précision d’un futur entraînement dépendra de contrôles réels de mémoire, gradients et stabilité sur l’appareil utilisé ; nous ne supposons pas BF16 disponible sur toutes les cartes.

## Acquisition et vérification locale

Depuis la racine du projet :

```sh
# Téléchargement volontaire, avec accès réseau.
sh scripts/project_uv.sh run --locked --offline python scripts/fetch_qwen3.py --download

# Relecture locale des tailles et empreintes, sans réseau.
sh scripts/project_uv.sh run --locked --offline python scripts/fetch_qwen3.py
```

Le `--offline` de uv concerne ici la résolution/installation des paquets ; l’option explicite `--download` autorise le script à demander les fichiers officiels par HTTPS. Sans cette option, le script ne fait que vérifier les fichiers déjà présents.

## Contrôle technique prévu

```sh
sh scripts/llm_uv.sh run --locked --offline python scripts/check_qwen_loading.py \
  --device auto --output outputs/qwen/mon_controle_chargement.json
```

Choisir un fichier de sortie neuf. Le programme vérifie les fichiers, charge `AutoModelForCausalLM` et le tokenizer sans code distant, compte les paramètres réellement chargés et contrôle le plafond de 1 milliard. Il distingue les éléments stockés dans Safetensors des paramètres uniques en mémoire : des poids liés peuvent avoir plusieurs noms dans le fichier mais représenter un seul paramètre partagé dans le modèle.

Tous les paramètres sont gelés pour l’inférence. Deux consignes inventées, en français et en arabe, sont passées au **chat template officiel avec `enable_thinking=False`**. Le préfixe assistant non-thinking est vérifié. Le programme réalise un passage avant avec sorties numériques finies et une génération plafonnée à 16 tokens par consigne.

Le décodage glouton utilisé pour ce contrôle n’est pas le protocole final des tâches. La fiche Qwen propose d’autres réglages de génération ; le choix scientifique sera fixé avant les mesures zero-shot/few-shot, puis conservé pour la comparaison avant/après adaptation. Aucune conclusion de qualité linguistique ou de classification n’est tirée de deux entrées inventées.

Le contrôle vérifie aussi l’absence de chargement 4/8 bits et relève les modules `q_proj` et `v_proj` pour préparer leur inspection ultérieure, sans créer LoRA. Il conserve versions, empreintes, compte de paramètres, sorties inventées, durée et mémoire observée. Les relevés MPS de fin sont des instantanés, pas un pic absolu ; les allocations CPU/GPU de mémoire unifiée ne s’additionnent pas.

## Suite du travail et limites

Après validation de cette acquisition : audit du tokenizer Qwen sur train/validation, définition des instructions et du parseur de labels, choix explicite des exemples few-shot depuis le train, puis référence zero-shot/few-shot. Le corpus SFT, les masques de perte et les contrôles LoRA suivront selon un protocole distinct. La longueur et le décodage des futures expériences ne sont pas choisis sur le test.

**L06** reste partiel : un modèle téléchargé et chargé n’est pas une référence évaluée ni un adaptateur appris. **L08** reçoit une base versionnée ; aucun modèle adapté n’est encore produit. **L12** reçoit un profil isolé, des empreintes et des commandes ; l’exécution CUDA et la reproduction indépendante restent à démontrer. L’application, les modèles CAMeLBERT existants et les données de test restent inchangés.
