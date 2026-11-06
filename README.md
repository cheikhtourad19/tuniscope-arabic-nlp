# TuniScope

> **Reproducible Arabic Variety Identification and Tunisian Sentiment Analysis**

TuniScope is an Arabic NLP project focused on measuring how well different approaches handle Arabic variety identification and Tunisian sentiment analysis. It compares lightweight classical methods, a fine-tuned Transformer encoder, and a compact instruction-tuned LLM.

## Problem

Arabic NLP systems often perform differently across Modern Standard Arabic, regional varieties, Arabizi, and French/English code-switching. TuniScope examines this challenge within a deliberately focused scope rather than claiming to identify every Arabic dialect.

## Tasks

| Task | Labels | Data source |
| --- | --- | --- |
| Arabic variety identification | MSA, Tunisian (Tunis/Sfax), Egyptian (Cairo) | MADAR Parallel Corpus |
| Tunisian sentiment classification | Positive, Negative | TSAC |

## Research questions

1. Does LoRA adaptation of a small LLM improve these classification tasks compared with the same model without adaptation?
2. How does the adapted LLM compare with TF-IDF baselines and a fine-tuned Arabic Transformer encoder?
3. Does cautious normalization improve robustness without removing negation, Arabizi signals, emojis, or code-switching information?

## Methodology

- **Classical baselines:** majority class, linguistic rules, TF-IDF word n-grams, and TF-IDF character n-grams.
- **Language-model reference:** smoothed character unigram/trigram models, evaluated by validation perplexity on a fixed representation and vocabulary within each corpus. See the [protocol and results](docs/37_modele_langage_caracteres.md).
- **Lightweight neural baseline:** train-from-scratch word embeddings and a bidirectional GRU, evaluated on the same v2 validation examples with three seeds. See the [BiGRU results](docs/40_resultats_bigru.md) and [isolated PyTorch environment](docs/39_environnement_bigru.md).
- **Encoder model:** an Arabic Transformer fine-tuned separately for dialect and sentiment classification.
- **Frozen encoder reference:** pinned CAMeLBERT Mix mean-pooled representations with learned linear heads, measured locally on Apple MPS before fine-tuning. See the [verified results and resource measurements](docs/45_resultats_encodeur_gele.md).
- **Generative model:** Qwen3-0.6B evaluated before and after one shared multitask LoRA adapter.
- **RAG:** a small cited knowledge base for questions about methodology and system limitations, not for deciding text labels.
- **Interface:** a local application for text/CSV input, model comparison, documented sources, and prediction export.

## Evaluation

- Macro-F1 is the primary classification metric, with per-class precision, recall, and confusion matrices.
- Final stochastic experiments use three seeds and report mean plus standard deviation.
- The test set remains frozen until the protocol is finalized.
- MADAR parallel translations are grouped to prevent leakage across splits.
- The analysis includes robustness by script, error categories, abstention behaviour, and computational cost.

## Data sources

- [MADAR Parallel Corpus](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/) supplies parallel MSA and city-dialect sentences, including Tunis, Sfax, and Cairo.
- [TSAC](https://github.com/fbougares/TSAC) supplies Tunisian social-media comments annotated as positive or negative.

Dataset use follows the respective licences and attribution requirements. Raw datasets are not redistributed with the project.

## Technical constraints

- Models must remain under one billion parameters.
- External paid APIs are excluded.
- Training is performed with a reproducible configuration and data-split protocol.
- The project targets Apple Silicon for local development and CUDA GPUs for larger training runs.

## Expected outcome

TuniScope produces an experimental NLP application together with comparative results, error analysis, reproducible training/evaluation workflows, model and data documentation, and a scientific report on the behaviour of the approaches across the two tasks.

## Local application

The React/FastAPI application is in [App/](App/README.md). It serves explicitly selected saved classical models, supports text/CSV prediction and comparison, and displays saved validation results. Model files stay Git-ignored and are copied into the backend image during the Docker build. See the application README to prepare a bundle and run Docker Compose. Encoder, LLM and RAG integration remain pending.

## Development environment

Data preparation and classical baselines use Python 3.12 with dependencies recorded in `pyproject.toml` and pinned in `uv.lock`. See the [environment setup guide](docs/20_environnement_python.md) for local installation and the prepared Kaggle/Colab workflow.

Once uv is available, run the fictitious-data installation check from the repository root:

```sh
sh scripts/project_uv.sh sync --locked
sh scripts/project_uv.sh run --locked python scripts/check_environment.py
```

This check requires no protected dataset. `requirements.txt` is a generated export of the lockfile for pip-based environments.

## Reproducible notebooks

- [Data loading and checks](notebooks/01_donnees_v2.ipynb): v2 manifests, in-memory tables, conservative normalization and aggregate statistics.
- [Initial classical baselines](notebooks/02_baselines_classiques.ipynb): train majority and word/character/combined TF-IDF models, verify the selected run, display a readable score table, and automatically export/show six PNG figures from that same run. See [integrated workflow](docs/35_entrainement_complet_et_export_figures.md).
- [BiGRU training and explanation](notebooks/03_bigru.ipynb): explain the neural layers and their place in the comparison, run smoke then full training with three seeds per task, verify checkpoints, show readable tables and six epoch/confusion PNGs. An explicit replay mode reloads an existing full run without training. See the [notebook guide](docs/41_notebook_bigru.md).
- [Encoder tokenizer audit](notebooks/04_tokenizer_encodeur.ipynb): inspect the pinned CAMeLBERT tokenizer on v2 train/validation, measure fragmentation/unknowns/truncation and reproduce the initial length decision. This uses tokenizer files only; it does not download weights or train the encoder. See the [audit results](docs/43_resultats_audit_tokenizer_encodeur.md).
- [Frozen encoder reference](notebooks/05_encodeur_gele.ipynb): extract fixed CAMeLBERT vectors, train separate linear heads on three seeds, verify reloads, and save/display six same-run PNGs. Full training is the default; replay reloads a chosen full run. The backbone stays frozen. See the [protocol](docs/44_protocole_encodeur_gele.md) and [results](docs/45_resultats_encodeur_gele.md).
- [Encoder fine-tuning](notebooks/06_fine_tuning_encodeur.ipynb): native BERT sequence classification with paired frozen/adapted CLS heads, gradient checks, three seeds, safetensors checkpoints, inference and per-model PNGs. Full mode uses all v2 training/validation data; smoke and explicitly typed replay remain separate. See the [controlled protocol](docs/46_protocole_fine_tuning_encodeur.md) and [workflow/evidence](docs/47_fine_tuning_encodeur_et_notebook.md). A full paired run trains twelve models and can take over an hour locally.
- [Encoder learning-rate screening](notebooks/07_selection_encodeur.ipynb): compare three fixed learning rates on seed 42, with automatic smoke checks, checkpoint verification and per-model PNGs. Full mode trains six adapted models on all v2 train/validation examples and proposes a rate per task; three-seed confirmation remains separate. See the [selection protocol](docs/52_protocole_selection_encodeur.md) and [Mac/Kaggle workflow and current evidence](docs/53_notebook_selection_encodeur.md).
- [Selected encoder confirmation](notebooks/08_confirmation_encodeur.ipynb): train three fresh seeds per task at fixed selected rates, verify checkpoints, report every seed plus mean/sample standard deviation, and display six PNGs. The local MPS full campaign and native notebook replay are verified: validation macro-F1 0.930834 ± 0.002308 for TSAC and 0.953482 ± 0.001235 for MADAR. See the [fixed protocol](docs/55_protocole_confirmation_encodeur.md), [execution guide](docs/56_notebook_confirmation_encodeur.md) and [results](docs/57_resultats_confirmation_encodeur.md). The official test remains unevaluated.

The notebooks contain French explanations and reuse the project scripts. The data notebook defaults to fictitious examples. Check `TRAINING_MODE` before running all cells: notebooks 02, 03, 05, 06, 07 and 08 default to `full` and launch real training. Notebooks 03, 05, 06, 07 and 08 delegate neural work to separate environments, so the existing notebook kernel can remain selected. Notebooks 05/06/07/08 require the explicitly downloaded pinned general checkpoint; no implicit downloads occur during training. Replay requires authorized local sources and trusted local model files. Initial paired CUDA compatibility is documented in [the Kaggle check](docs/49_controle_kaggle_cuda.md); this does not establish every workflow's cloud execution or independent reproducibility.
