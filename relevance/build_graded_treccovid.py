#!/usr/bin/env python3
"""Graded relevance-as-Score from BEIR/TREC-COVID (vertical: COVID biomedical) with human GRADED
qrels (0 not-relevant / 1 partially / 2 relevant). Each row is a single (query, passage) pair the
model must rate on the 0-2 scale — an ordinal typed decision with human gold.

Balanced sample across the three grades. Rows carry `query` and `passage` fields so a pair-scoring
backend can read them (see scripts/rel_eval.py); `state` also embeds both for generic LLM prompting.

Usage: python scripts/build_relevance_graded.py [--per-grade 90] [--maxlen 600]
"""
from __future__ import annotations
import argparse, json, os, random
from collections import defaultdict

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
LEVELS = ["Not relevant: the passage does not address the query.",
          "Partially relevant: the passage touches on the query but is incomplete or tangential.",
          "Relevant: the passage directly and substantially addresses the query."]

def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-grade", type=int, default=90); ap.add_argument("--maxlen", type=int, default=600)
    ap.add_argument("--out", default="data/relevance_graded_treccovid.jsonl")
    a = ap.parse_args()

    queries = {r["_id"]: r["text"] for r in load_dataset("BeIR/trec-covid", "queries")["queries"]}
    by_grade = defaultdict(list)
    for r in load_dataset("BeIR/trec-covid-qrels")["test"]:
        if r["score"] in (0, 1, 2) and str(r["query-id"]) in queries:
            by_grade[r["score"]].append((str(r["query-id"]), str(r["corpus-id"])))

    rng = random.Random(0)
    pairs = []
    for g in (0, 1, 2):
        rng.shuffle(by_grade[g]); pairs += [(q, c, g) for q, c in by_grade[g][:a.per_grade]]
    needed = {c for _, c, _ in pairs}

    corpus = {}
    for r in load_dataset("BeIR/trec-covid", "corpus", streaming=True)["corpus"]:
        if r["_id"] in needed:
            corpus[r["_id"]] = (r.get("title", "") + ". " + r.get("text", "")).strip()
            if len(corpus) == len(needed):
                break

    rows = []
    for qid, cid, g in pairs:
        if cid not in corpus:
            continue
        q, p = queries[qid], corpus[cid][:a.maxlen]
        rows.append({"id": f"beir_trec_covid:relevance_graded:{qid}:{cid}", "source": "beir_trec_covid",
                     "domain": "covid_relevance_graded", "type": "score",
                     "query": q, "passage": p,
                     "state": f"QUERY:\n{q}\n\nPASSAGE:\n{p}",
                     "question": {"type": "score",
                                  "instructions": "Rate how relevant the PASSAGE is to the QUERY.",
                                  "levels": LEVELS},
                     "label": str(g), "eval_metric": "ordinal", "reliability": "gold", "label_source": "human"})
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} graded (query,passage) rows to {a.out} | grades={dict(Counter(r['label'] for r in rows))}")

if __name__ == "__main__":
    main()
