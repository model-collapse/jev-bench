# JevBench core dataset

**2934 ground-truth examples**, one file per **`<topic>/<type>.jsonl`** cell. Each line is one example
with `label`, `label_source`, `reliability` (gold/silver), and `eval_metric` (exact/ordinal). Load a
single file, a topic folder, or the whole `data/` dir (the harness accepts any).

- reliability: gold 1682 · silver 1252   (all rows are ground truth)
- eval_metric: exact 2206 · ordinal 728

## Files (topic/type → n, source)
| topic | type | n | source |
|---|---|--:|---|
| agent_trace_observability | choice | 156 | typed_decisions |
| agent_trace_observability | noul | 60 | typed_decisions |
| agent_trace_observability | score | 168 | typed_decisions |
| banking | choice | 300 | banking77 |
| commerce | choice | 100 | panoramic |
| customer_service | choice | 137 | typed_decisions |
| customer_service | noul | 96 | typed_decisions |
| customer_service | score | 184 | typed_decisions |
| invoice_processing | choice | 65 | typed_decisions |
| invoice_processing | noul | 200 | typed_decisions |
| invoice_processing | score | 184 | typed_decisions |
| invoices | noul | 100 | panoramic |
| logic | noul | 100 | panoramic |
| moderation | score | 100 | panoramic |
| observability | noul | 100 | panoramic |
| org | choice | 100 | panoramic |
| quant | noul | 100 | panoramic |
| reviews | choice | 100 | panoramic |
| security_incidents | choice | 99 | typed_decisions |
| security_incidents | noul | 193 | typed_decisions |
| security_incidents | score | 192 | typed_decisions |
| support | choice | 100 | panoramic |

## Scoring
`exact` → accuracy (choice/noul/computed-math). `ordinal` → QWK + MAE (subjective score;
off-by-one is a small error). Use `--tier gold` for the bias-free subset. See top-level
`../README.md` for the harness and `../TASKS.md` for the task taxonomy.

## Provenance
panoramic = synthetic/computed · banking = banking77 (CC-BY-SA, human) · the operational topics
(customer_service, security_incidents, invoice_processing, agent_trace_observability) =
Apache-2.0 cross-family panel via stability voting + deterministic recompute (no frontier outputs).
The separate `../relevance/` family is built from third-party IR data and is NOT part of this core.
