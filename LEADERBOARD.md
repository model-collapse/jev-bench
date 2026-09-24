# JevBench — Initial Leaderboard

Decision-core results on a **232-example stratified gold sample** (`reliability == "gold"`), scored
per primitive: **exact accuracy** for `choice`/`noul`, **QWK** (quadratic-weighted kappa) for ordinal
`score`. This is an *initial* board — treat the top cluster as a statistical tie (n=232, ±~4% CI).

| # | model | paradigm | access | overall | choice | noul | score-QWK |
|---|---|---|---|--:|--:|--:|--:|
| 1 | GPT-5.5 | generative LLM | Bedrock (proprietary) | 90.9% | 94.4 | 96.3 | 0.89 |
| 2 | Opus 5 | generative LLM | Bedrock (proprietary) | 90.5% | 93.3 | 93.9 | 0.95 |
| 3 | GPT-6-astra | generative LLM | Bedrock (proprietary) | 89.7% | 95.6 | 92.7 | 0.82 |
| 4 | **Jev** (jev-latest) | native typed-decision API | api.typesafe.ai (proprietary ref) | 88.8% | 94.4 | 91.5 | 0.96 |
| 5 | gpt-oss-20b | generative LLM | **Apache-2.0** | 88.4% | 91.1 | 93.9 | 0.96 |
| 6 | laya (base, zero-shot) | encoder + [MASK] readout | Apache-2.0 | 53.9% | 46.7 | 72.0 | 0.10 |
| 7 | bart-large-mnli | NLI / entailment | MIT | 50.0% | 55.6 | 47.6 | 0.53 |
| 8 | bge-reranker-base | cross-encoder | MIT | 40.9% | 37.8 | 42.7 | 0.42 |
| 9 | option-scoring (Qwen2.5-0.5B, fixed) | option-scoring LLM | Apache-2.0 | 39.2% | 40.0 | 45.1 | −0.15 |
| 10 | all-MiniLM-L6-v2 | bi-encoder embedding | Apache-2.0 | 34.9% | 34.4 | 43.9 | −0.04 |

All rows are on the same 232-example gold sample. **Option-scoring scales with model size:** the 0.5B
row above is the base-model floor; Qwen2.5-1.5B-Instruct via the same option-scoring path reaches ~55%
(choice 55 / noul 50 / score-QWK 0.48, measured on a 60-row subset) — so the paradigm tracks capability,
and the molora 4B target should land higher.

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
