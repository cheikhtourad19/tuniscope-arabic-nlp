# TuniScope — Source Register

Track access, licensing, and dataset verification here. Raw/protected data remains outside Git; this file records only provenance and decisions.

| Source | Purpose | Official URL | Access checked? | Licence / restrictions | Files / labels confirmed | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| MADAR Corpus-26 | Dialect: MSA / TN / EG | https://camel.abudhabi.nyu.edu/madar-parallel-corpus/ | Yes — registration confirmation received 2026-10-02 | Internal research/evaluation use only; no sublicensing, redistribution, assignment, or dataset modification rights. Keep raw archive private and out of Git. | Yes — archive verified; MSA, Tunis, Sfax, and Cairo TSVs use `sentID.BTEC`, `split`, `lang`, and `sent` columns. | Approved for private acquisition and audit. |
| TSAC | Tunisian sentiment: POS / NEG | https://github.com/fbougares/TSAC | Yes — public repository inspected and local archive verified 2026-10-02 | Repository displays LGPL-3.0. Cite the source paper and keep the raw corpus out of Git; confirm redistribution obligations before publishing any data derivative. | Yes — `train_pos.txt`, `train_neg.txt`, `test_pos.txt`, and `test_neg.txt` are present and passed local ZIP integrity testing. | Approved for private acquisition and audit. |

## MADAR acquisition record

- Official archive: `MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip`
- Official download URL: <https://camel.abudhabi.nyu.edu/madar-parallel-corpus/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip>
- Local storage (Git-ignored): `data/raw/madar/`
- Untouched extracted copy (Git-ignored): `data/raw/madar/original/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/`
- Archive SHA-256: `fc9c34638a9c8f8d5266d6431966b6ed933ad07b1e987514fa75f0b5eaf50a6a`
- Archive contents: verified without extraction on 2026-10-02. The release contains `MADAR.corpus.{MSA,Tunis,Sfax,Cairo}.tsv`; ZIP integrity testing completed with no errors.
- Verified row counts (including header): MSA 12,001; Tunis 12,001; Sfax 2,001; Cairo 12,001.
- Split design note: use the shared `sentID.BTEC` field to keep parallel translations grouped. The 2,000-row Sfax Corpus-26 portion can define the parallel four-variety subset; do not treat the larger Corpus-6 rows as independent from their matching source sentences.

## S1 decision

- Dialect data: MADAR Corpus-26 — access, licence, archive integrity, and required city files verified.
- Sentiment data: TSAC — access, repository licence display, local archive integrity, and labelled split files verified.
- Raw datasets will remain outside Git; only manifests, hashes, scripts, and documentation will be versioned.

## TSAC acquisition record

- Official repository archive: `TSAC-master.zip`
- Official repository: <https://github.com/fbougares/TSAC>
- Local storage (Git-ignored): `data/raw/tsac/`
- Untouched extracted copy (Git-ignored): `data/raw/tsac/original/TSAC-master/`
- Archive SHA-256: `757f2873b516574accb0e5c21d7be7ef86db414f110d5fc109b59988a1e69268`
- Archive integrity: verified locally with no ZIP errors on 2026-10-02.
- Verified contents: `LICENSE`, `README.md`, `train_pos.txt`, `train_neg.txt`, `test_pos.txt`, and `test_neg.txt`.
- Verified line counts: train POS 7,154; train NEG 6,515; test POS 1,700; test NEG 1,700.
- Handling note: preserve the official test files untouched. A validation set will later be derived only from the training files.
