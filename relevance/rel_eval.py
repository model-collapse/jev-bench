#!/usr/bin/env python3
"""Fair evaluator for SINGLE-PAIR relevance rows (query, passage) -> grade.

The main harness reduces a question to "pick a candidate", which is unfair to cross-encoders/embedders
on absolute relevance (scoring the tokens '0'/'1'/'2' or 'yes'/'no' is meaningless). Here a
PAIR-SCORING backend produces a raw relevance score for (query, passage); we then evaluate with
THRESHOLD-FREE metrics that don't require an arbitrary cutoff:
  - AUC        : ranking of relevant (grade>0) above non-relevant  (binary view)
  - Spearman   : rank-correlation of the score with the graded label (ordinal view)
  - QWK        : after label-agnostic quantile-binning the score into k levels
An LLM backend (which emits a discrete level) is scored directly with QWK / MAE / accuracy, plus AUC
using its level as the score.

Backends: cross-enc, embed (pair scorers) ; llm-hf (rates the level) ; random.
Usage: python scripts/rel_eval.py --backend cross-enc --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
         --data data/relevance/treccovid_graded.jsonl
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

def rankdata(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r

def pearson(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a)); db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0

def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))

def auc(binary, score):
    pos = [s for y, s in zip(binary, score) if y == 1]
    neg = [s for y, s in zip(binary, score) if y == 0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def qwk(true, pred, n):
    O = [[0] * n for _ in range(n)]
    for a, b in zip(true, pred):
        O[a][b] += 1
    w = [[((i - j) ** 2) / ((n - 1) ** 2 or 1) for j in range(n)] for i in range(n)]
    rt = [sum(O[i]) for i in range(n)]; ct = [sum(O[i][j] for i in range(n)) for j in range(n)]
    tot = sum(rt) or 1
    num = sum(w[i][j] * O[i][j] for i in range(n) for j in range(n))
    den = sum(w[i][j] * rt[i] * ct[j] / tot for i in range(n) for j in range(n))
    return 1 - num / den if den else 1.0

def quantile_bins(scores, k):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    out = [0] * len(scores)
    for rank, i in enumerate(order):
        out[i] = min(k - 1, rank * k // len(scores))
    return out

# ---- backends: return a raw relevance score for (query, passage) ----
class CrossEnc:
    def __init__(self, model): from sentence_transformers import CrossEncoder; self.m = CrossEncoder(model)
    def score(self, rows): return list(self.m.predict([[r["query"], r["passage"]] for r in rows]))
class Embed:
    def __init__(self, model): from sentence_transformers import SentenceTransformer; self.m = SentenceTransformer(model)
    def score(self, rows):
        from sentence_transformers.util import cos_sim
        qe = self.m.encode([r["query"] for r in rows]); pe = self.m.encode([r["passage"] for r in rows])
        return [float(cos_sim(qe[i], pe[i])) for i in range(len(rows))]
class Random:
    def __init__(self, model=None): import random; self.r = random.Random(0)
    def score(self, rows): return [self.r.random() for _ in rows]
class LLM:
    """Emits a discrete level; we return it as the 'score' (also usable directly for QWK/MAE)."""
    def __init__(self, model):
        import torch; from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
    def score(self, rows):
        out = []
        for r in rows:
            lv = r["question"]["levels"]
            p = (f"QUERY:\n{r['query']}\n\nPASSAGE:\n{r['passage'][:600]}\n\n"
                 + "Rate the passage's relevance to the query:\n"
                 + "\n".join(f"- {i}: {x}" for i, x in enumerate(lv)) + "\nAnswer with the number. ANSWER: ")
            ids = self.tok(p, return_tensors="pt").input_ids
            with self.torch.no_grad():
                g = self.model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=self.tok.eos_token_id)
            txt = self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r"[0-9]", txt); out.append(min(int(m.group()), len(lv) - 1) if m else 0)
        return out

BK = {"cross-enc": CrossEnc, "embed": Embed, "random": Random, "llm-hf": LLM}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=list(BK)); ap.add_argument("--model", default="")
    ap.add_argument("--data", default="data/relevance/treccovid_graded.jsonl"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data)]
    if a.limit: rows = rows[:a.limit]
    grades = [int(r["label"]) for r in rows]; k = max(grades) + 1
    binary = [1 if g > 0 else 0 for g in grades]
    print(f"rel_eval: {len(rows)} pairs | grades 0..{k-1} | backend={a.backend} {a.model or ''}", flush=True)

    scores = BK[a.backend](a.model).score(rows)
    is_llm = a.backend == "llm-hf"
    A = auc(binary, scores); S = spearman(scores, grades)
    if is_llm:
        pred = [int(s) for s in scores]
        print(f"  AUC(relevant>0)={A:.3f}  Spearman={S:.3f}  QWK={qwk(grades,pred,k):.3f} "
              f" MAE={sum(abs(p-g) for p,g in zip(pred,grades))/len(rows):.2f} "
              f" acc={100*sum(p==g for p,g in zip(pred,grades))/len(rows):.1f}%")
    else:
        binned = quantile_bins(scores, k)
        print(f"  AUC(relevant>0)={A:.3f}  Spearman={S:.3f}  QWK(quantile-binned)={qwk(grades,binned,k):.3f}")

if __name__ == "__main__":
    main()
