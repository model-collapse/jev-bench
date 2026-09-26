# JevBench — Initial Leaderboard

Decision-core results on a **232-example stratified gold sample** (`reliability == "gold"`), scored
per primitive: **exact accuracy** for `choice`/`noul`, **QWK** (quadratic-weighted kappa) for ordinal
`score`. This is an *initial* board — treat the top cluster as a statistical tie (n=232, ±~4% CI).

All numbers below are from one uniform re-run on the 232-example gold sample (per-example outputs);
`overall`/`choice`/`noul` are exact-accuracy %. `score-QWK`/`score-MAE` are the ordinal metric on the
~46-row subjective-`score` subset, **single pooled confusion matrix** (2 five-level rows dropped).
**A per-topic breakdown table is in the repo [README](README.md#results).**

| # | model | paradigm | access | overall | choice | noul | score-QWK | score-MAE |
|---|---|---|---|--:|--:|--:|--:|--:|
| 1 | Opus 5 | generative LLM | Bedrock (proprietary) | 90.9 | 93 | 95 | 0.84 | 0.28 |
| 2 | GPT-5.5 | generative LLM | Bedrock (proprietary) | 89.7 | 93 | 95 | 0.77 | 0.39 |
| 2 | GPT-6-astra | generative LLM | Bedrock (proprietary) | 89.7 | 96 | 93 | 0.63 | 0.46 |
| 4 | **Jev** (jev-latest) | native typed-decision API | api.typesafe.ai (proprietary ref) | 88.8 | 94 | 92 | 0.92 | 0.20 |
| 5 | gpt-oss-20b | generative LLM | **Apache-2.0** | 88.4 | 91 | 94 | 0.94 | 0.15 |
| 6 | bart-large-mnli | NLI / entailment | MIT | 54.3 | 56 | 60 | 0.07 | 0.83 |
| 7 | laya (base, zero-shot) | encoder + [MASK] readout | Apache-2.0 | 53.9 | 47 | 72 | 0.21 | 1.24 |
| 8 | bge-reranker-base | cross-encoder | MIT | 40.9 | 38 | 43 | −0.16 | 1.07 |
| 8 | all-MiniLM-L6-v2 | bi-encoder embedding | Apache-2.0 | 40.9 | 34 | 61 | −0.07 | 1.61 |
| 10 | option-scoring (Qwen2.5-0.5B, fixed) | option-scoring LLM | Apache-2.0 | 39.2 | 40 | 45 | −0.30 | 1.02 |

`noul` for the NLI and bi-encoder backends now uses the paradigm-appropriate mechanism (NLI: true
entailment premise=state/hypothesis=statement; embedder: statement-vs-negation cosine) — the old
"score the tokens yes/no" was a harness bug that deflated them (NLI 50.0→54.3; embedder noul 44→61).
Cross-encoder `noul` and all encoder `score` are left as-is (a reranker's noul "fix" is a degenerate
yes-bias; `score` is a genuine paradigm mismatch). An encoder-friendly state-only `choice` query would
lift embed/cross-enc further but the gain is lexical-overlap, not reasoning — not adopted.

The top five overall (~88–91%) are a statistical tie (n=232). **On `score-QWK`, read tiers not exact
values:** it's a ~46-row metric effectively decided by a 7-item low-severity tail under quadratic
weighting, so gpt-oss/Jev/Opus 5 (~0.84–0.94) are within noise of each other (only {gpt-oss, Opus 5} >
GPT-6 is significant by paired bootstrap); `score-MAE` is the more stable companion. Small/similarity
models land at ~0 or negative because they aren't ordinal judges — they collapse to a near-constant
level or call the least-severe cases most-severe; only bge-reranker and option-scoring are
significantly below 0. **Option-scoring scales with model size:** the 0.5B row is a base-model floor;
Qwen2.5-1.5B-Instruct via the same path reaches ~55% overall (60-row subset).

**Excluded:**
- **laya-typed-decisions (67.2%)** — *train/test contamination*: its model card states it was fine-tuned
  on "the typed-decisions benchmark," which overlaps this benchmark's `typed_decisions` rows (77.6% on
  those vs 58.3% on panoramic). Not a comparable number. The fair laya entry is base laya zero-shot (53.9%).
- **Fable 5** — unbenchmarkable on this account: it requires a non-default (zero-data-retention) Bedrock
  inference profile, and the Converse API has no request-level parameter for it. Not 0% — excluded.

## Reading the board
- **Reasoning-native paradigms win the decision core** (~88–91%): the top five — three frontier LLMs,
  the purpose-built typed-decision API (Jev), and the open Apache `gpt-oss-20b` — are a statistical tie.
  Jev, the specialist, sits squarely in the cluster; a capable open model matches it.
- **The real separation is by paradigm**: reasoning LLMs (~90%) ≫ small open encoder laya (54%) ≈ NLI (50%) ≫ cross-encoder (41%) ≈
  option-scoring-0.5B (39%, post-fix) ≈ bi-encoder (35%). Similarity models are weak on
  reasoning/verification but lead on the separate **relevance** family (see `relevance/`).
- **Per-primitive matters**: Opus 5 tops ordinal `score` (QWK 0.95); GPT-5.5 tops `noul`; GPT-6 tops `choice`.

## Methodology notes & gotchas (learned running this board)
- **Headline is the `gold` tier** — human/objective/computed labels — so no model is flattered by
  panel-derived `silver` labels it may be adjacent to.
- **Bedrock frontier IDs need the `us.` inference-profile prefix**; the bare id fails and (without the
  guard below) shows a misleading 0%.
- **Reasoning models need a large `--gen-tokens`** (e.g. 4096) or they spend the budget on a hidden
  reasoning channel and return no answer → false ~0%.
- **The harness now guards against silent failures**: if >20% of predictions are None (bad id, blocked
  model, extraction bug), it prints an `UNRELIABLE` warning instead of reporting a fake score.
- **Option-scoring was fixed**: a tokenization/delimiter bug had made it look far worse than it is
  (below-chance `choice`, always-"yes" `noul`). Numbers here are post-fix; see git history.

## Reproduce
```bash
# any model, gold tier, per-primitive
python bench_eval.py --backend auto --model <id> --data data --tier gold
# frontier via Bedrock (note the us. prefix + generous reasoning budget)
python bench_eval.py --backend bedrock --model us.openai.gpt-5.5 --data data --tier gold --gen-tokens 4096
# Jev (native typed-decision API); key from env, never logged
TYPESAFE_API_KEY=... python bench_eval.py --backend jev --model jev-latest --data data --tier gold
```
Sample size and exact rows may evolve; this board is a snapshot, not a certified ranking.
