#!/usr/bin/env python3
"""Build a small, vertical RELEVANCE-as-decision task from BEIR/SciFact (scientific claim -> evidence
abstract). Cast as pairwise Choice: given a claim and two passages — one relevant (human qrel), one
irrelevant — pick the relevant one. This is the cross-encoder's native op, framed as a typed Choice
decision, and it directly uses relevant/irrelevant pairs.

Hard negatives: among non-relevant abstracts, pick the one with the highest token overlap with the
claim (topical distractor), so the decision isn't trivial.

Output: data/relevance/scifact_choice.jsonl in the benchmark schema (type=choice, gold/human labels).
Usage: python scripts/build_relevance.py [--n 200] [--neg-pool 60] [--maxlen 600]
"""
from __future__ import annotations
import argparse, json, os, random, re, sys
from collections import defaultdict

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
STOP = set("the a an of to in and or for is are was were be been being this that these those with "
           "on at by as it its from into than then can could may might will would we they he she "
           "his her their our your not no do does did has have had which who whom whose about".split())

def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}

def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--neg-pool", type=int, default=80)
    ap.add_argument("--k", type=int, default=2, help="candidates per item = 1 positive + (k-1) hard negatives")
    ap.add_argument("--maxlen", type=int, default=600); ap.add_argument("--out", default="data/relevance/scifact_choice.jsonl")
    a = ap.parse_args()

    corpus = {r["_id"]: (r["title"] + ". " + r["text"]).strip()
              for r in load_dataset("BeIR/scifact", "corpus")["corpus"]}
    queries = {r["_id"]: r["text"] for r in load_dataset("BeIR/scifact", "queries")["queries"]}
    rel = defaultdict(set)
    for r in load_dataset("BeIR/scifact-qrels")["test"]:
        if r["score"] > 0:
            rel[str(r["query-id"])].add(str(r["corpus-id"]))

    all_ids = list(corpus)
    rng = random.Random(0)
    qids = [q for q in rel if q in queries and rel[q] & set(corpus)]
    rng.shuffle(qids)

    rows = []
    for qid in qids:
        if len(rows) >= a.n:
            break
        pos_ids = [d for d in rel[qid] if d in corpus]
        if not pos_ids:
            continue
        pos = rng.choice(pos_ids)
        qtok = toks(queries[qid])
        # hard negatives: top-(k-1) token-overlap non-relevant docs from a random pool
        pool = [d for d in rng.sample(all_ids, min(a.neg_pool, len(all_ids))) if d not in rel[qid]]
        pool.sort(key=lambda d: len(qtok & toks(corpus[d][:400])), reverse=True)
        negs = pool[:a.k - 1]
        if len(negs) < a.k - 1:
            continue
        cut = lambda t: t[:a.maxlen]
        opts = {pos: cut(corpus[pos]), **{n: cut(corpus[n]) for n in negs}}
        opts = dict(rng.sample(list(opts.items()), len(opts)))   # shuffle option order
        rows.append({"id": f"beir_scifact:relevance:{qid}", "source": "beir_scifact",
                     "domain": "scientific_claim_relevance", "type": "choice", "state": queries[qid],
                     "question": {"type": "choice",
                                  "instructions": "A scientific claim and two candidate abstracts are given. "
                                  "Exactly one abstract is relevant evidence for the claim. Pick the relevant one.",
                                  "options": opts},
                     "label": pos, "eval_metric": "exact", "reliability": "gold", "label_source": "human"})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} pairwise relevance-choice rows to {a.out}")
    print("example:", json.dumps({**rows[0], "question": {**rows[0]["question"],
          "options": {k: v[:60] + '…' for k, v in rows[0]["question"]["options"].items()}}}, indent=1)[:700])

if __name__ == "__main__":
    main()
