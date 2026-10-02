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
- **Encoder model:** an Arabic Transformer fine-tuned separately for dialect and sentiment classification.
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
