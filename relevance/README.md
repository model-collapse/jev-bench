# JevBench — Relevance task family

Relevance prediction cast as **typed decisions** — the native job of cross-encoders and embedding
models, added so the benchmark tests *information-access* decisions and is **fair across paradigms**
(the core reasoning/operational tasks favor LLMs; these favor retrieval models).

Because the sources are third-party research IR datasets (BEIR / MS MARCO), this repo ships the
**builder scripts, not the derived data** — you regenerate the tasks locally from HuggingFace. This
keeps provenance clean and avoids redistributing corpora under their own licenses (see
[Provenance & licenses](#provenance--licenses)).

## Tasks

| task | type | source | decision | metric |
|---|---|---|---|---|
| **claim → evidence** | Choice (N-way) | BEIR/SciFact | pick the relevant abstract among hard negatives | exact accuracy |
| **graded relevance** | Score (0–2) | BEIR/TREC-COVID | rate a (query, passage) pair's relevance | ordinal (AUC / Spearman / QWK) |
| *(in-domain reference)* | Score (0–1) | MS MARCO (Tevatron) | binary relevance of a pair | AUC |

SciFact "queries" are scientific **claims**; TREC-COVID is biomedical — both **out-of-domain** for
web-search rerankers, by design (they measure *generalization*).

## Build

```bash
pip install datasets sentence-transformers        # + torch/transformers for LLM/scoring
# N-way selection (1 positive + k-1 hard negatives); --k 2 = pairwise
python relevance/build_choice_scifact.py --n 200 --k 4 --out data/relevance_scifact_4way.jsonl
# graded (query,passage) pairs, human qrels 0/1/2
python relevance/build_graded_treccovid.py --per-grade 90 --out data/relevance_graded_treccovid.jsonl
```

## Evaluate

- **Choice** tasks run in the main harness (they fit the "pick a candidate" contract):
  ```bash
  python bench_eval.py --backend cross-enc --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
      --data data/relevance_scifact_4way.jsonl
  ```
- **Single-pair** tasks (graded/binary) need `rel_eval.py`, which **pair-scores** `(query, passage)`
  and uses **threshold-free** metrics (scoring the `0/1/2` tokens in the generic harness is unfair to
  rerankers): AUC (binary relevant>0), Spearman (graded rank-corr), QWK (label-agnostic quantile
  bins). An LLM backend rates a level directly (QWK/MAE/accuracy).
  ```bash
  python relevance/rel_eval.py --backend cross-enc --model BAAI/bge-reranker-base \
      --data data/relevance_graded_treccovid.jsonl
  ```
  Backends: `cross-enc`, `embed`, `llm-hf`, `random`.

## What this measures: specialists vs generalists (domain transfer)

Holding metric/harness/pair-format fixed and changing **only the domain** flips the paradigm ranking:

| | embed (bi) | ms-marco (cross) | bge-reranker (cross) | random |
|---|--:|--:|--:|--:|
| **in-domain** MS MARCO — AUC | 0.967 | **0.984** | **0.985** | 0.51 |
| **out-of-domain** TREC-COVID — AUC | **0.78** | 0.73 | 0.71 | 0.40 |

Cross-encoders are **specialists** (dominant in-domain, brittle off-distribution); a well-trained
bi-encoder is a **generalist** (slightly behind in-domain, more robust out-of-domain). The vertical
SciFact/TREC-COVID sets are valuable precisely because they measure generalization — which the
in-domain number hides. Always report both paradigms.

A cross-encoder **fine-tuned on the target domain** (or an option-scoring / NLI head) becomes a
first-class, calibratable decision model; this family is the evaluation target for exactly such
models.

## Provenance & licenses

These tasks derive from third-party datasets — **not** Apache-2.0 like the JevBench core. Verify and
comply with each source's terms before redistribution/commercial use:
- **SciFact** — Wadden et al., 2020 (via BEIR).
- **TREC-COVID** — Voorhees et al. / BEIR.
- **MS MARCO** — Bajaj et al. (Microsoft), "free for research"; passages via Tevatron.
- **BEIR** — Thakur et al., 2021 (aggregator).

Builders download from HuggingFace at run time; no source corpora are committed to this repo.
