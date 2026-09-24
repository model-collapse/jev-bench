# JevBench

An open, model-agnostic benchmark for **typed decision models** — Choice, Score, and Noul —
in the spirit of Jev. This repo holds the **evaluation harness**; the dataset is released separately
(see [Data](#data)).

One command scores any model with the right metric per question, and works across model families:
LLMs (prompted or as option-scorers), cross-encoders/rerankers, embedding models, and your own
custom systems.

See **[TASKS.md](TASKS.md)** for the full task taxonomy — every task along two dimensions,
**task type** (Choice / Score / Noul) × **task context** (skill probes, applied operational
scenarios, real-world), with the context × type matrix.

```bash
pip install -r requirements.txt        # install only what your backend needs
python bench_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --data benchmark.jsonl
```

## Data

The harness reads a JSONL file (`--data`, default `data/benchmark_release.jsonl`). Each row:

```json
{
  "id": "...", "type": "score",            // choice | noul | score
  "domain": "security_incidents",
  "state": "{...structured input...}",     // string the model reads
  "question": {"instructions": "...",
               "levels": ["...","..."]},    // choice: "options": {key: text}; noul: neither
  "label": "3",                             // gold: option key | "yes"/"no" | level index
  "eval_metric": "ordinal",                 // exact | ordinal
  "reliability": "gold"                     // gold | silver
}
```

Answer space per primitive: **choice** → the option keys; **noul** → `yes`/`no`; **score** → the
level indices `0..k-1`. The dataset drop (data files + datasheet) will be linked here. For what the
2,934 tasks cover, see the taxonomy in **[TASKS.md](TASKS.md)**.

## Metrics

Each row is scored by its `eval_metric`:
- **`exact`** (all `noul`, all `choice`, computed-math questions): `prediction == label`, reported as accuracy.
- **`ordinal`** (subjective `score`): ordered levels with conventional boundaries, so off-by-one is a small error → reported as **QWK** (quadratic-weighted kappa) **+ MAE**. Bare exact-match is *not* used on ordinal rows.

The harness prints overall accuracy, an `exact` block, an `ordinal` block (QWK/MAE), and breakdowns
by primitive, reliability tier, and domain. Rows carry a `reliability` tier (`gold` = human/objective/
computed; `silver` = model-panel labeled); use `--tier gold` for the bias-free subset.

## Backends

### Auto-detection (`--backend auto`, the default)
Omit `--backend` and the harness inspects the model (config/metadata only, not weights) and routes,
printing its choice:

| detected | rule |
|---|---|
| `llm-api` | an API `--base-url` was given |
| `embed` | sentence-transformers repo with a Pooling module (bi-encoder) |
| `cross-enc` | a `*ForSequenceClassification` head (reranker / NLI) |
| `llm-hf` / `score-hf` | a causal LM — **generate** if the name looks instruct/chat, else **option-score** |

Robust for LLM-vs-cross-encoder-vs-embedding-vs-API (from architecture). The one heuristic call is
generate-vs-option-score for causal LMs (name-based; chat templates ship on many *base* models, so
they're unreliable) — always overridable. Pin an explicit `--backend` for reproducible published runs.

### Supported families
| backend | model family | how it answers |
|---|---|---|
| `llm-hf` | HuggingFace causal LM | prompt → **generate** → extract the answer key |
| `llm-api` | OpenAI-compatible chat API | prompt → generate → extract (`--base-url`, `--model`, `--api-key`) |
| `score-hf` | causal LM as **option scorer** | no generation; length-normalised `log P(candidate \| context)` after the answer delimiter, argmax (order-invariant; good for base models) |
| `cross-enc` | sentence-transformers **CrossEncoder** | score each `(context, candidate)` pair, argmax |
| `embed` | sentence-transformers bi-encoder | `cosine(context, candidate)`, argmax |
| `random` | — | uniform baseline (sanity floor) |

### Usage
```bash
# HuggingFace causal LM (generate + extract)
python bench_eval.py --backend llm-hf   --model Qwen/Qwen2.5-1.5B-Instruct --limit 500

# Causal LM as an option scorer (no generation; good for base/non-chat models)
python bench_eval.py --backend score-hf --model Qwen/Qwen2.5-0.5B

# Cross-encoder / reranker
python bench_eval.py --backend cross-enc --model cross-encoder/ms-marco-MiniLM-L-6-v2 --types choice

# OpenAI-compatible API (also vLLM, TGI, Together, … via --base-url)
OPENAI_API_KEY=sk-... python bench_eval.py --backend llm-api --model gpt-4o-mini

# Bias-free gold subset, or specific primitives
python bench_eval.py --backend embed --model sentence-transformers/all-MiniLM-L6-v2 --tier gold --types choice,noul
```

Flags: `--data` (jsonl), `--limit` (0=all), `--tier {all,gold,silver}`, `--types choice,noul,score`,
`--out results.json` (per-example predictions).

## Per-paradigm protocols

Every row is reduced to a `context` (`instructions` + `INPUT:` + `state`) and a candidate set; a
prediction must be exactly one candidate key. Decisions are deterministic (greedy/argmax, fixed seed).

- **LLM (generate):** context + candidate list + "emit `ANSWER: <value>`"; greedy generate; parse.
  Sensitive to **option order** and format — run an order-shuffled pass to quantify order-bias.
- **LLM (option-score):** score each candidate's log-prob after the delimiter, argmax. Order-invariant,
  no parse failures, works for base models.
- **Embedding:** encode context and each candidate independently → argmax cosine. Order-invariant;
  measures topical similarity, so weak on reasoning/`noul`.
- **Cross-encoder:** jointly encode `(context, candidate)` → relevance score → argmax. Order-invariant;
  a **fine-tuned** cross-encoder (option-scoring / NLI head) becomes a genuine, calibratable decision
  model — this benchmark is the evaluation target for exactly such models.
- **Other:** any object with `predict(row) -> candidate_key` (see below).

## Custom backends ("others")
```python
class MyRunner:
    def __init__(self, model, **kw): ...
    def predict(self, row):
        cands = candidates(row)   # [(key, text), ...]
        ctx   = context(row)      # instructions + INPUT
        ...
        return chosen_key         # must be one of the candidate keys
BACKENDS["mine"] = MyRunner
```
`candidates(row)`, `context(row)`, `build_prompt(row)`, `extract(text, row)` are exposed for reuse.

## Provenance & license
Code is Apache-2.0 (`LICENSE`). This is an **independent, open** benchmark; it is not affiliated with
or endorsed by TypeSafe AI. Dataset labels (released separately) come from an Apache-2.0 cross-family
model panel via independent stability voting plus deterministic recomputation of objective questions —
no frontier-model outputs are used in shipped labels.
