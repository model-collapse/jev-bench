# JevBench — Task Taxonomy

Every task in JevBench is described by **two orthogonal dimensions**:

1. **Task type** — the *shape of the decision* (its answer space): **Choice**, **Score**, or **Noul**.
2. **Task context** — the *domain/scenario* the input comes from, from isolated skill-probes to full
   operational decisions.

A task is one `(context × type)` cell; most cells hold one or two question templates. The benchmark
has **2,934** examples: **Choice 1,157 · Noul 949 · Score 828**.

---

## Dimension 1 — Task type (answer space)

| type | answer space | metric | n | reads like |
|---|---|---|---|---|
| **Choice** | exactly one of N enumerated options (each with a description) | `exact` (accuracy) | 1,157 | an `enum` — classification / routing / disposition |
| **Noul** | `yes` / `no` | `exact` (accuracy) | 949 | a `bool` — verification / entailment / detection |
| **Score** | one level on a fixed ordinal scale `0..k` | `ordinal` (QWK + MAE) or `exact` when computed | 828 | a bounded ordinal — severity / urgency / risk rating |

- **Choice** tests discrimination among labeled alternatives (which intent / team / disposition).
- **Noul** tests truth judgments — often requiring arithmetic, logic, or evidence weighing (does it
  reconcile? does the conclusion follow? is this malicious?).
- **Score** tests *calibrated ordinal judgment*. Because level boundaries are conventional, Score is
  graded ordinally (off-by-one is a small error) — except where the level is deterministically
  computable, which is graded exactly.

---

## Dimension 2 — Task context

Contexts fall into three families by how the label is grounded:

### A. Panoramic probes — isolated skills, objective ground truth (all **gold**, `exact`)
Synthetic micro-tasks, each isolating a single capability so a model's failures are diagnosable.

| context | type | skill probed | example question |
|---|---|---|---|
| `moderation` | Score | safety-severity rating | *Rate the safety severity of this message.* |
| `reviews` | Choice | sentiment | *What is the sentiment of this review?* |
| `support` | Choice | topic routing | *Which team should handle this message?* |
| `observability` | Noul | error detection | *Does this log line indicate an error?* |
| `org` | Choice | multi-hop relational lookup | *Which department is Alice's manager in?* |
| `invoices` | Noul | arithmetic (sum check) | *Does the stated total equal the sum of the line items?* |
| `logic` | Noul | deductive entailment | *Does the conclusion logically follow from the premises?* |
| `quant` | Noul | aggregation + comparison | *Is the average of these numbers greater than N?* |
| `commerce` | Choice | constraint satisfaction | *Pick the product that is BOTH under \$50 AND in stock.* |

Each is 100 examples. `quant`/`invoices`/`logic`/`commerce`/`org` are **reasoning** probes (arithmetic,
logic, constraints, multi-hop); the rest are **atomic** perception/classification.

### B. Applied operational decisions — realistic scenarios (**gold + silver**)
Rich structured `state` (JSON: conversations, alerts, invoices, agent traces) with several typed
decisions each — the core of the benchmark.

| context | Choice | Noul | Score | what the input is |
|---|---|---|---|---|
| `customer_service` | topic; next-action | needs-human-agent | churn-risk; time-sensitivity | a support conversation + account/orders |
| `security_incidents` | alert disposition | malicious?; credential-compromised? | impact-severity; response-urgency | a security alert + context/principal/history |
| `invoice_processing` | invoice disposition | reconciles?; duplicate? | discrepancy-materiality; processing-urgency | an invoice + PO + delivery + vendor history |
| `agent_trace_observability` | run outcome; observability action | requires-human-review | behaviour-risk; attention-urgency | an autonomous agent run trace + constraints |

Counts: customer_service 417 · security_incidents 484 · invoice_processing 449 · agent_trace_observability 384.
These exercise every type over the *same* context, so a model is tested on choosing, verifying, and
rating within one realistic decision surface.

### C. Real-world human-labeled — `banking77` (Choice, 300, **gold**)
77-way intent classification from real customer banking messages (human labels + induced intent
definitions). A high-cardinality, fine-grained Choice task with genuine human ground truth.

---

## The full matrix (context × type, example counts)

| context | Choice | Noul | Score | total |
|---|--:|--:|--:|--:|
| banking (banking77) | 300 | — | — | 300 |
| commerce | 100 | — | — | 100 |
| reviews | 100 | — | — | 100 |
| support | 100 | — | — | 100 |
| org | 100 | — | — | 100 |
| moderation | — | — | 100 | 100 |
| observability | — | 100 | — | 100 |
| invoices | — | 100 | — | 100 |
| logic | — | 100 | — | 100 |
| quant | — | 100 | — | 100 |
| customer_service | 137 | 96 | 184 | 417 |
| security_incidents | 99 | 193 | 192 | 484 |
| invoice_processing | 65 | 200 | 184 | 449 |
| agent_trace_observability | 156 | 60 | 168 | 384 |
| **total** | **1,157** | **949** | **828** | **2,934** |

## How the two dimensions interact with scoring

- **Objective cells** (panoramic; invoice reconcile/duplicate/materiality) have deterministic ground
  truth → `exact` (or `ordinal` for the computed Score). They are the reasoning/verification backbone.
- **Subjective cells** (applied Score, most applied Noul/Choice judgments) are model-panel labeled
  (`silver`), so subjective Score uses the `ordinal` metric to tolerate the conventional-boundary
  wobble; use `--tier gold` to evaluate only against fully-trustworthy labels.

Reporting is always **per type and per context**, never a single blended number — a model can be
strong at Choice-routing yet weak at arithmetic Noul or calibrated Score, and the taxonomy is built
to surface exactly that.
