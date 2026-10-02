# TuniScope — Scope and System Design

**Deliverable:** L01 Cadrage (working version)  
**Owner:** one fifth-year computer-science engineering student  
**Status:** scope fixed; no model has been trained and no result is claimed.

## 1. Purpose

TuniScope evaluates practical NLP approaches for Arabic text when the language may be Modern Standard Arabic (MSA), Tunisian, Egyptian, Arabizi, or mixed with French/English. It is an experimental comparison under limited data and compute—not a system that claims to identify every Arabic dialect or make high-stakes decisions.

## 2. Research question

**Main question:** Does LoRA adaptation of a small LLM improve Arabic variety identification and Tunisian sentiment classification compared with the same LLM before adaptation?

The adapted LLM is compared with two distinct references: classical TF-IDF-based classifiers and a fine-tuned Arabic Transformer encoder. This comparison evaluates practical trade-offs; it does not isolate the causal effect of LoRA across identical architectures.

## 3. Fixed core scope

| Item | Decision |
| --- | --- |
| Task 1 | Arabic variety identification: `MSA`, `TN`, `EG` |
| Task 2 | Tunisian sentiment classification: `POS`, `NEG` |
| Dialect data | MADAR: MSA, Tunis, Sfax, Cairo; Tunis and Sfax form the `TN` project class |
| Sentiment data | TSAC; preserve its official test files and derive validation only from training data |
| Classical models | Majority baseline, linguistic rules, TF-IDF word n-grams, TF-IDF character n-grams; light neural baseline if time allows |
| Encoder | `CAMeL-Lab/bert-base-arabic-camelbert-mix`, with separate classification checkpoints accepted for the two tasks |
| LLM | `Qwen/Qwen3-0.6B`, evaluated before and after one shared multitask LoRA adapter |
| Retrieval | Small, cited RAG collection for methodology/protocol questions only |
| Interface | Local text/CSV input, model comparison, source display, and JSON/CSV export |

The data audit will determine the final balanced training counts and exact grouped splits. It must not change the three dialect labels, the two sentiment labels, or the official TSAC test boundary without a recorded protocol revision.

## 4. System map

```text
Authorized raw corpora
        │
        ▼
Data audit → manifests → cautious normalization → leakage checks
        │
        ├── Classical baselines ───┐
        ├── Fine-tuned encoder ────┼──► shared evaluation ───► local application
        └── Qwen3 + shared LoRA ───┘          │
                                               ├── metrics and error analysis
Verified method/source cards ─► RAG ──────────┘
```

The RAG branch answers questions about the project method and its limits. It never provides evidence that a new text’s predicted sentiment or dialect is correct.

## 5. Data and evaluation protocol

- Preserve raw text and record all transformations in a separate normalized-text field.
- Keep MADAR translations with the same `sentID.BTEC` in one split to prevent parallel-sentence leakage.
- Keep TSAC’s official test split untouched; make validation only from the official training files.
- Use validation—not the test set—for hyperparameters, normalization choices, prompts, thresholds, and checkpoint selection.
- Report macro-F1 as the main classification metric, alongside per-class metrics and confusion matrices.
- Run final stochastic configurations with seeds 13, 42, and 2026; report each score, mean, and standard deviation.
- Record data manifest, data hash, configuration, model revision, hardware, runtime, memory, predictions, and errors for each meaningful experiment.

## 6. Constraints and non-goals

| Constraint | Consequence |
| --- | --- |
| One student, 14 weeks | Prioritize the required Silver core before any extension. |
| Models under 1B parameters | No large-model training from scratch. |
| No paid APIs | All inference and training use local or free notebook resources. |
| MADAR licence restrictions | Raw MADAR data is private and never committed or redistributed. |
| Apple Silicon + Kaggle CUDA | One portable codebase; device selection belongs in configuration/runtime detection. |

Out of scope until the required core is complete: all-dialect Arabic coverage, audio, commercial deployment, autonomous agents, NER, translation, active learning, and learned Arabizi transliteration.

## 7. Evidence expected at J1

- This scope and system-design document.
- A source register with access, licences, and dataset provenance.
- A data card, annotation guide, canonical schema, and leakage-safe split plan.
- A realistic solo work plan and time budget.

## References

- CAMeL Lab. [MADAR Parallel Corpus](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/).
- Bougares et al. [TSAC: Tunisian Sentiment Analysis Corpus](https://github.com/fbougares/TSAC).
- CAMeL Lab. [CAMeLBERT Mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix).
- Qwen Team. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
