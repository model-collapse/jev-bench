#!/usr/bin/env python3
"""Standard evaluation harness for the Typed-Decision Benchmark.

Self-contained (only needs the release .jsonl + whatever your backend needs). Scores a model on the
benchmark with the correct metric per row (`exact` for choice/noul/computed-math; `ordinal` = QWK+MAE
for subjective score), and breaks results out by primitive, reliability tier, and domain.

Supported model families (`--backend`):
  * llm-hf     : HuggingFace causal LM + prompt -> generate -> extract the answer key.
  * llm-api    : OpenAI-compatible chat API (--base-url/--model/--api-key) -> generate -> extract.
  * score-hf   : causal LM as an OPTION SCORER (Jev/laya style) — no generation; score each
                 candidate by its length-normalised log-prob right after the answer delimiter,
                 argmax. Order-invariant, robust extraction.
  * cross-enc  : sentence-transformers CrossEncoder — score each (context, candidate) pair, argmax.
  * embed      : sentence-transformers bi-encoder — cosine(context, candidate), argmax.
  * random     : uniform-random baseline (sanity floor).

Every backend reduces a question to "pick the best candidate":
  choice -> the option keys ; noul -> yes/no ; score -> the level indices (with their text).

Usage examples:
  python scripts/bench_eval.py --backend llm-hf   --model HuggingFaceTB/SmolLM2-135M-Instruct --limit 100
  python scripts/bench_eval.py --backend score-hf --model Qwen/Qwen2.5-0.5B --limit 100
  python scripts/bench_eval.py --backend cross-enc --model cross-encoder/ms-marco-MiniLM-L-6-v2
  python scripts/bench_eval.py --backend llm-api  --model gpt-4o-mini --base-url https://api.openai.com/v1
  python scripts/bench_eval.py --backend embed    --model sentence-transformers/all-MiniLM-L6-v2 --tier gold
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter, defaultdict

# ----------------------------- data / candidates -----------------------------
def candidates(row):
    """Return [(key, text), ...] — the choices the model must pick among."""
    q = row["question"]; t = row["type"]
    if t == "noul":
        return [("yes", "yes"), ("no", "no")]
    if t == "score":
        return [(str(i), lv) for i, lv in enumerate(q["levels"])]
    opts = q.get("options") or {}
    items = opts.items() if isinstance(opts, dict) else ((o, o) for o in opts)
    return [(str(k), str(v)) for k, v in items]

def context(row):
    q = row["question"]
    return f"{q.get('instructions','')}\n\nINPUT:\n{row['state']}"

# ----------------------------- prompt / extraction (llm) ----------------------
def build_prompt(row):
    cands = candidates(row); t = row["type"]
    ctx = context(row)
    if t == "noul":
        tail = "Answer with exactly `yes` or `no`."
    elif t == "score":
        opts = "\n".join(f"- {k}: {v}" for k, v in cands)
        tail = f"Levels:\n{opts}\nAnswer with the single level NUMBER."
    else:
        opts = "\n".join(f"- {k}: {v}" for k, v in cands)
        tail = f"Options:\n{opts}\nAnswer with exactly one option key."
    return f"{ctx}\n\n{tail}\nEnd with a line: `ANSWER: <value>`."

def extract(text, row):
    """Map free text to one candidate key. Robust to format drift."""
    keys = [k for k, _ in candidates(row)]
    m = re.search(r"ANSWER:\s*(.+)", text, re.I)
    tail = m.group(1) if m else text
    low = tail.lower()
    if row["type"] == "noul":
        yi, ni = low.find("yes"), low.find("no")
        if yi == -1 and ni == -1: return None
        return "yes" if (yi != -1 and (ni == -1 or yi < ni)) else "no"
    if row["type"] == "score":
        d = re.search(r"-?\d+", tail)
        return d.group(0) if (d and d.group(0) in keys) else None
    # choice: match the option key (longest first to avoid prefixes), else whole text
    for src in (tail, text):
        for k in sorted(keys, key=len, reverse=True):
            if re.search(rf"(?<![\w-]){re.escape(k.lower())}(?![\w-])", src.lower()):
                return k
    return None

# ----------------------------- metrics ----------------------------------------
def qwk(true, pred, n):
    import numpy as np
    if not true: return 0.0
    O = np.zeros((n, n))
    for a, b in zip(true, pred): O[a, b] += 1
    w = np.array([[((i - j) ** 2) / ((n - 1) ** 2 or 1) for j in range(n)] for i in range(n)])
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else 1.0

def report(preds, out=None):
    n = len(preds)
    corr = sum(p["correct"] for p in preds)
    print(f"\n{'='*64}\nRESULTS  n={n}  overall exact-accuracy = {100*corr/n:.1f}%")
    # by eval_metric
    ex = [p for p in preds if p["eval_metric"] == "exact"]
    orl = [p for p in preds if p["eval_metric"] == "ordinal"]
    if ex:
        print(f"  [exact metric]   n={len(ex):4}  accuracy={100*sum(p['correct'] for p in ex)/len(ex):.1f}%")
    if orl:
        # ordinal: QWK + MAE over score rows (group by n_levels)
        mae = sum(p["distance"] for p in orl if p["distance"] is not None) / max(len(orl), 1)
        acc = 100 * sum(p["correct"] for p in orl) / len(orl)
        by_n = defaultdict(lambda: ([], []))
        for p in orl:
            if p["pred"] is not None and str(p["pred"]).lstrip("-").isdigit():
                by_n[p["n_levels"]][0].append(int(p["gold"])); by_n[p["n_levels"]][1].append(int(p["pred"]))
        qwks = [qwk(t, pr, nn) for nn, (t, pr) in by_n.items() if t]
        qk = sum(qwks) / len(qwks) if qwks else 0.0
        print(f"  [ordinal metric] n={len(orl):4}  QWK={qk:.3f}  MAE={mae:.2f}  exact={acc:.1f}%")
    # by type / tier / domain
    def slice_(key):
        g = defaultdict(list)
        for p in preds: g[p[key]].append(p)
        for k in sorted(g):
            rs = g[k]; print(f"    {key}={str(k):14} n={len(rs):4} acc={100*sum(x['correct'] for x in rs)/len(rs):.1f}%")
    print("  by type:");   slice_("type")
    print("  by tier:");   slice_("reliability")
    print("  by domain:"); slice_("domain")
    if out:
        json.dump(preds, open(out, "w"))
        print(f"\nwrote per-example predictions to {out}")

# ----------------------------- backends ---------------------------------------
class Random:
    def __init__(self, **_):
        import random; self.r = random.Random(0)
    def predict(self, row):
        return self.r.choice([k for k, _ in candidates(row)])

class LLM_HF:
    def __init__(self, model, max_new_tokens=48, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
        self.model.eval(); self.max_new = max_new_tokens
    def _text(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = self.tok(prompt, return_tensors="pt").input_ids
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    def predict(self, row):
        return extract(self._text(build_prompt(row)), row)

class Score_HF:
    """Option scorer: length-normalised log P(candidate | context) after the delimiter. No generation."""
    def __init__(self, model, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32); self.model.eval()
    def _logprob(self, prefix, answer):
        t = self.tok
        pre = t(prefix, return_tensors="pt").input_ids
        full = t(prefix + answer, return_tensors="pt").input_ids
        with self.torch.no_grad():
            logits = self.model(full).logits.log_softmax(-1)
        ans_ids = full[0, pre.shape[1]:]
        if len(ans_ids) == 0: return -1e9
        lp = sum(logits[0, pre.shape[1] + i - 1, tid].item() for i, tid in enumerate(ans_ids))
        return lp / len(ans_ids)
    def predict(self, row):
        prefix = build_prompt(row).replace("End with a line: `ANSWER: <value>`.", "ANSWER: ")
        best, bk = -1e18, None
        for k, txt in candidates(row):
            ans = k if row["type"] != "score" else k          # score answer = the number
            s = self._logprob(prefix, ans)
            if s > best: best, bk = s, k
        return bk

class CrossEnc:
    def __init__(self, model, **_):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model)
    def predict(self, row):
        ctx = context(row)
        cands = candidates(row)
        scores = self.m.predict([[ctx, txt] for _, txt in cands])
        return cands[int(max(range(len(cands)), key=lambda i: scores[i]))][0]

class Embed:
    def __init__(self, model, **_):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)
    def predict(self, row):
        from sentence_transformers.util import cos_sim
        cands = candidates(row)
        qe = self.m.encode(context(row))
        ce = self.m.encode([txt for _, txt in cands])
        sims = cos_sim(qe, ce)[0]
        return cands[int(sims.argmax())][0]

BACKENDS = {"random": Random, "llm-hf": LLM_HF, "score-hf": Score_HF,
            "cross-enc": CrossEnc, "embed": Embed}

# ----------------------------- auto-detection ---------------------------------
def autodetect(model, base_url=None):
    """Inspect the model and pick a backend. Returns (backend, reason). Heuristic — always
    overridable with an explicit --backend. Only downloads small config files, not weights."""
    if base_url:
        return "llm-api", "an API base-url was given"
    if not model:
        return "random", "no model id"
    # 1. sentence-transformers bi-encoder? (has a Pooling module in modules.json)
    try:
        from huggingface_hub import hf_hub_download
        mods = json.load(open(hf_hub_download(model, "modules.json")))
        if any("Pooling" in m.get("type", "") for m in mods):
            return "embed", "sentence-transformers repo with a Pooling module (bi-encoder)"
    except Exception:
        pass
    # 2. inspect the transformers config
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model)
        arch = " ".join(cfg.architectures or [])
        n_labels = getattr(cfg, "num_labels", None)
    except Exception as e:
        return "llm-hf", f"could not read config ({type(e).__name__}); defaulting to generative"
    if "ForCausalLM" in arch or "ForConditionalGeneration" in arch or getattr(cfg, "is_decoder", False):
        # generative model — the one genuinely ambiguous call: GENERATE (instruct) vs OPTION-SCORE
        # (base). Decide by name intent (chat templates ship on many base models, so they're
        # unreliable here). Always overridable with an explicit --backend.
        instruct = re.search(r"(?i)instruct|chat|-it\b|sft|dpo|rlhf", model)
        return ("llm-hf", "causal LM, name looks instruct/chat -> generate (override: --backend score-hf)") \
            if instruct else \
            ("score-hf", "causal LM, name looks like a base model -> option-scoring (override: --backend llm-hf)")
    if "ForSequenceClassification" in arch:
        if n_labels in (1, 2, None):
            return "cross-enc", f"sequence-classification head (num_labels={n_labels}) -> pairwise reranker"
        return "cross-enc", (f"WARNING: {n_labels}-way classifier — pairwise scoring may misfit; "
                             "consider a custom backend mapping class logits to candidate keys")
    return "embed", f"encoder architecture ({arch or '?'}) with no classifier head -> similarity"

# ----------------------------- API backend (openai-compatible) ----------------
class LLM_API:
    def __init__(self, model, base_url=None, api_key=None, **_):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "x"))
        self.model = model
    def predict(self, row):
        r = self.c.chat.completions.create(model=self.model,
            messages=[{"role": "user", "content": build_prompt(row)}], max_tokens=64)
        return extract(r.choices[0].message.content, row)
BACKENDS["llm-api"] = LLM_API

# ----------------------------- run --------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=list(BACKENDS) + ["auto"])
    ap.add_argument("--model", default="")
    ap.add_argument("--data", default="data/benchmark_release.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--tier", default="all", choices=["all", "gold", "silver"])
    ap.add_argument("--types", default="", help="comma list e.g. choice,noul")
    ap.add_argument("--base-url", default=None); ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    backend = a.backend
    if backend == "auto":
        backend, why = autodetect(a.model, a.base_url)
        print(f"[auto] detected backend '{backend}' — {why}")

    rows = [json.loads(l) for l in open(a.data)]
    if a.tier != "all": rows = [r for r in rows if r["reliability"] == a.tier]
    if a.types: rows = [r for r in rows if r["type"] in a.types.split(",")]
    if a.limit: rows = rows[:a.limit]
    print(f"benchmark: {len(rows)} rows | backend={backend} model={a.model or '-'}")

    runner = BACKENDS[backend](model=a.model, base_url=a.base_url, api_key=a.api_key)
    preds = []
    for i, r in enumerate(rows):
        try: pred = runner.predict(r)
        except Exception as e: pred = None
        gold = str(r["label"])
        dist = None
        if r["type"] == "score" and pred is not None and str(pred).lstrip("-").isdigit():
            dist = abs(int(pred) - int(gold))
        preds.append({"id": r["id"], "type": r["type"], "domain": r.get("domain"),
                      "reliability": r["reliability"], "eval_metric": r["eval_metric"],
                      "gold": gold, "pred": pred, "correct": str(pred) == gold, "distance": dist,
                      "n_levels": len(r["question"].get("levels", [])) or 2})
        if (i + 1) % 50 == 0: print(f"  ...{i+1}/{len(rows)}", flush=True)
    report(preds, a.out or None)

if __name__ == "__main__":
    main()
