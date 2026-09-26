# JevBench

An open, model-agnostic benchmark for **typed decision models** — Choice, Score, and Noul —
in the spirit of Jev. This repo holds the **evaluation harness**; the dataset is released separately
(see [Data](#data)).

One command scores any model with the right metric per question, and works across model families:
LLMs (prompted or as option-scorers), cross-encoders/rerankers, embedding models, and your own
custom systems.

See **[LEADERBOARD.md](LEADERBOARD.md)** for the initial model leaderboard, and **[TASKS.md](TASKS.md)** for the full task taxonomy — every task along two dimensions,
**task type** (Choice / Score / Noul) × **task context** (skill probes, applied operational
scenarios, real-world), with the context × type matrix.

There is also an add-on **[relevance task family](relevance/)** (relevance prediction as a typed
decision — the native job of cross-encoders / embedding models), shipped as builder scripts with its
own provenance since it derives from third-party IR datasets.

## Results

Initial leaderboard on a **232-example gold sample** (`reliability == "gold"`). `overall`/`choice`/
`noul` are exact-accuracy %; **`score-QWK`**/`score-MAE` are the ordinal metric on the ~46-row
subjective-`score` subset (single pooled confusion matrix). The top five overall (~88–91%) are a
**statistical tie** (n=232), and `score-QWK` there is effectively a **7-item-tail test** under
quadratic weighting — read it as ordinal-quality *tiers*, not precise scores: gpt-oss/Jev/Opus 5
(~0.84–0.94) are within noise of each other, GPT-5.5 clearly good, GPT-6 a step below; only
bge-reranker and option-scoring are significantly below 0 (they mis-order). `score-MAE` is the more
stable companion and tells the same story.
`laya-typed-decisions` (67%) is **excluded** for train/test contamination and **Fable 5** is
unbenchmarkable on this account (data-retention block) — see [LEADERBOARD.md](LEADERBOARD.md) for
methodology, exclusions, and gotchas.

### By task type

| model | paradigm | overall | choice | noul | score-QWK | score-MAE |
|---|---|--:|--:|--:|--:|--:|
| Opus 5 | frontier LLM | 90.9 | 93 | 95 | 0.84 | 0.28 |
| GPT-5.5 | frontier LLM | 89.7 | 93 | 95 | 0.77 | 0.39 |
| GPT-6-astra | frontier LLM | 89.7 | 96 | 93 | 0.63 | 0.46 |
| **Jev** | typed-decision API *(ref)* | 88.8 | 94 | 92 | 0.92 | 0.20 |
| gpt-oss-20b | open LLM (Apache) | 88.4 | 91 | 94 | 0.94 | 0.15 |
| bart-mnli | NLI / entailment | 54.3 | 56 | 60 | 0.07 | 0.83 |
| laya (base, zero-shot) | open encoder | 53.9 | 47 | 72 | 0.21 | 1.24 |
| bge-reranker | cross-encoder | 40.9 | 38 | 43 | −0.16 | 1.07 |
| all-MiniLM | bi-encoder | 40.9 | 34 | 61 | −0.07 | 1.61 |
| option-scoring 0.5B | option-scoring LLM | 39.2 | 40 | 45 | −0.30 | 1.02 |

**`noul` for the NLI and bi-encoder backends uses the paradigm-appropriate mechanism** (NLI: true
entailment, premise=state / hypothesis=statement; embedder: statement-vs-negation cosine) — scoring
the bare tokens "yes"/"no" was a harness bug that deflated them (NLI 50.0→54.3; embedder noul 44→61).
The cross-encoder `noul` and all `score` for the encoders are left as-is: a reranker's noul "fix" is a
degenerate yes-bias (≈ always-yes), and `score` is a genuine paradigm mismatch. (An encoder-friendly
"state-only" `choice` query lifts embed/cross-enc further, but the gain is lexical overlap, not
reasoning, so it is not adopted.)

#### Reading `score-QWK` (including negative values)

`score-QWK` is quadratic-weighted kappa on the ordinal `score` predictions, **chance-corrected** — so
the number tells you where a model sits relative to random guessing, not just how often it's exactly
right. Read it as bands, not precise scores:

| QWK | meaning |
|---|---|
| **1.0** | perfect ordinal agreement |
| **≈ 0.7–0.95** | genuinely calibrated ordinal judgment (the reasoning models) |
| **≈ 0** | **no ordinal signal** — the model guesses, or collapses to a near-constant level |
| **< 0** | **worse than chance** — predictions are *anti-correlated* with truth: the model tends to rank severity **backwards** (e.g. calling the *least*-severe cases *most*-severe) |

So a **negative** score is a stronger, worse verdict than 0: zero means "uninformative," negative
means "**mis**-informative — you'd do better inverting its predictions." On this board only
**bge-reranker (−0.16)** and **option-scoring (−0.30)** are *significantly* below 0 (they genuinely
mis-order); **all-MiniLM's −0.07 is statistically indistinguishable from 0** (read it as "no signal,"
not "anti-correlated"). Caveats: the ordinal subset is ~46 rows dominated by ~7 low-severity items
under quadratic weighting, so treat these as tiers and read **`score-MAE`** (average levels-off, more
stable) alongside; among the strong models the fine ordering is within noise (only {gpt-oss, Opus 5} >
GPT-6 is significant).

### By topic (accuracy %)

| topic | Opus 5 | GPT-5.5 | GPT-6 | Jev | gpt-oss-20b | NLI | laya | bge-rerank | all-MiniLM | opt-score-0.5B |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| commerce | 100 | 100 | 100 | 100 | 100 | 25 | 33 | 25 | 25 | 33 |
| logic | 100 | 100 | 100 | 100 | 100 | 50 | 75 | 50 | 100 | 58 |
| org | 100 | 100 | 100 | 100 | 100 | 8 | 8 | 42 | 33 | 50 |
| quant | 100 | 100 | 100 | 100 | 100 | 33 | 67 | 42 | 42 | 42 |
| invoices | 100 | 100 | 100 | 83 | 100 | 33 | 58 | 67 | 33 | 33 |
| observability | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 42 | 42 | 75 |
| moderation | 100 | 100 | 100 | 58 | 75 | 25 | 75 | 25 | 25 | 42 |
| reviews | 100 | 100 | 100 | 100 | 100 | 67 | 67 | 33 | 17 | 100 |
| support | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 83 | 58 |
| banking | 100 | 100 | 100 | 75 | 100 | 62 | 38 | 62 | 75 | 0 |
| customer_service | 85 | 82 | 82 | 76 | 68 | 62 | 53 | 21 | 44 | 26 |
| invoice_processing | 77 | 81 | 73 | 81 | 92 | 50 | 58 | 42 | 50 | 42 |
| security_incidents | 84 | 78 | 81 | 88 | 75 | 78 | 25 | 44 | 34 | 16 |
| agent_trace_observability | 79 | 75 | 79 | 100 | 88 | 38 | 46 | 29 | 8 | 29 |

The topic view shows the paradigm split cleanly: reasoning models saturate the objective/panoramic
skills (commerce, logic, org, quant) and lead on the harder operational topics, while similarity /
small-encoder paradigms hold up only on the topical ones (support, observability) and collapse on
multi-hop/arithmetic reasoning (org, logic, agent traces).

```bash
pip install -r requirements.txt        # install only what your backend needs
python bench_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --data benchmark.jsonl
```

## Data

The harness reads the dataset via `--data` (default **`data/`** — a folder of `<topic>/<type>.jsonl` files; you can also pass a single file or one topic folder to evaluate a slice). Each row:

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
level indices `0..k-1`. The dataset lives in **[`data/`](data/)** (see [`data/README.md`](data/README.md) for the file map + datasheet); the taxonomy is in **[TASKS.md](TASKS.md)**.

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
