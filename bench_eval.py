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
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict

def load_rows(path):
    """Load one .jsonl file, OR every .jsonl under a directory (recursively). The separate
    `relevance/` family is skipped in directory mode (it has its own evaluator, rel_eval.py)."""
    if os.path.isdir(path):
        files = sorted(f for f in glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
                       if f"{os.sep}relevance{os.sep}" not in f)
        rows = []
        for f in files:
            rows += [json.loads(l) for l in open(f)]
        return rows
    return [json.loads(l) for l in open(path)]

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
    if not true: return None
    O = np.zeros((n, n))
    for a, b in zip(true, pred): O[a, b] += 1
    w = np.array([[((i - j) ** 2) / ((n - 1) ** 2 or 1) for j in range(n)] for i in range(n)])
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else None  # degenerate (no gold variance)

def report(preds, out=None):
    n = len(preds)
    corr = sum(p["correct"] for p in preds)
    # guard: a systematically-failing model (bad id, blocked model, scoring bug) yields all-None
    # predictions that would otherwise masquerade as a real low/0% score. Surface it loudly.
    invalid = sum(1 for p in preds if p["pred"] is None)
    errored = sum(1 for p in preds if p.get("errored"))
    print(f"\n{'='*64}\nRESULTS  n={n}  overall exact-accuracy = {100*corr/n:.1f}%")
    if invalid:
        rate = 100 * invalid / n
        print(f"  invalid/unparseable predictions: {invalid}/{n} ({rate:.0f}%)"
              + (f"  [{errored} were call errors]" if errored else ""))
        if rate >= 20:
            print(f"  *** WARNING: {rate:.0f}% of predictions are None — the score is UNRELIABLE. "
                  "Likely a wrong model id, a blocked/unavailable model, truncated reasoning "
                  "(raise --gen-tokens), or an extraction bug — NOT a real result. ***")
    # by eval_metric
    ex = [p for p in preds if p["eval_metric"] == "exact"]
    orl = [p for p in preds if p["eval_metric"] == "ordinal"]
    if ex:
        print(f"  [exact metric]   n={len(ex):4}  accuracy={100*sum(p['correct'] for p in ex)/len(ex):.1f}%")
    if orl:
        # ordinal: QWK + MAE over score rows. POOLED — one confusion matrix over ALL ordinal rows
        # (do NOT group by n_levels and average: a tiny same-gold group yields a degenerate den=0
        # matrix that returns a free QWK=1.0 and inflates the mean).
        mae = sum(p["distance"] for p in orl if p["distance"] is not None) / max(len(orl), 1)
        acc = 100 * sum(p["correct"] for p in orl) / len(orl)
        tt, pp = [], []
        for p in orl:
            if p["pred"] is not None and str(p["pred"]).lstrip("-").isdigit():
                tt.append(int(p["gold"])); pp.append(int(p["pred"]))
        qk = qwk(tt, pp, max(tt + pp) + 1) if tt else None
        qs = f"{qk:.3f}" if qk is not None else "n/a (degenerate)"
        print(f"  [ordinal metric] n={len(orl):4}  QWK={qs}  MAE={mae:.2f}  exact={acc:.1f}%")
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
    """Option scorer (no generation): for each candidate, score the answer token(s) the prompt asks
    for — option key / level number / yes|no — as the continuation after the delimiter, then argmax.
    The option DESCRIPTIONS are already in the prompt, so scoring the bare key lets the context, not
    the surface form of the answer, decide. Two corrections are load-bearing:

      * boundary-safe scoring — tokenise the prefix and the (space-prefixed) answer SEPARATELY and
        concatenate ids; never re-tokenise `prefix + answer`. The delimiter ends in a space and BPE
        merges that space into the first answer token, so a joint re-tokenisation is NOT a token-
        prefix of the prefix alone: for short answers ("yes"/"no") the token diff is EMPTY (every
        candidate scores the same sentinel, so argmax silently returns the first one), and for longer
        answers the boundary token shifts so the logprobs are read one position out of alignment.
      * surface-form competition — option keys have very different unigram frequencies, so for
        `choice` we rank by domain-conditional PMI, logP(ans|context) - logP(ans|delimiter-only),
        which cancels that prior. For noul/score the answer's own base rate is informative (yes/no
        priors; ordinal-level frequency), so there we keep the plain length-normalised logprob."""
    DELIM = "ANSWER:"
    def __init__(self, model, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32); self.model.eval()
    def _logprob(self, prefix, answer):
        """Return (raw_sum, length_normalised) log P(answer | prefix). Boundary-safe: the answer is
        tokenised as " " + answer independently of the prefix and their ids concatenated, so the
        answer tokens are exactly defined and correctly aligned regardless of BPE space-merging."""
        t = self.tok
        pre = t(prefix.rstrip(), return_tensors="pt").input_ids
        ans = t(" " + str(answer).strip(), return_tensors="pt", add_special_tokens=False).input_ids
        n = ans.shape[1]
        if n == 0: return -1e9, -1e9
        full = self.torch.cat([pre, ans], dim=1)
        with self.torch.no_grad():
            logits = self.model(full).logits.log_softmax(-1)
        p = pre.shape[1]
        lp = sum(logits[0, p + i - 1, full[0, p + i]].item() for i in range(n))
        return lp, lp / n
    def predict(self, row):
        prefix = build_prompt(row).replace("End with a line: `ANSWER: <value>`.", self.DELIM + " ")
        choice = (row["type"] == "choice")
        best, bk = -1e18, None
        for k, _ in candidates(row):
            raw, norm = self._logprob(prefix, k)
            s = raw - self._logprob(self.DELIM, k)[0] if choice else norm   # PMI for choice
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

class Jev:
    """Native typed-decision API (TypeSafe / Jev). Returns the decision directly — no parsing.
    Key from env TYPESAFE_API_KEY (or JEV_API_KEY); never printed."""
    URL = "https://api.typesafe.ai/v1/systemone"
    def __init__(self, model=None, **_):
        self.key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.key:
            raise SystemExit("set TYPESAFE_API_KEY in the environment")
        self.model = model or "jev-latest"
    def predict(self, row):
        import urllib.request
        cands = candidates(row)
        q = {"type": row["type"], "instructions": row["question"].get("instructions", "")}
        if row["type"] == "choice":
            q["criteria"] = {k: v for k, v in cands}
        elif row["type"] == "score":
            q["criteria"] = [v for _, v in cands]        # ordered level descriptions
        body = json.dumps({"state": row["state"], "model": self.model, "questions": {"q": q}}).encode()
        req = urllib.request.Request(self.URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        ans = json.load(urllib.request.urlopen(req, timeout=60))["answers"]["q"]
        if row["type"] == "choice":
            return str(ans["choice"])
        if row["type"] == "noul":
            return "yes" if float(ans["noul"]) >= 0.5 else "no"
        probs = ans.get("probabilities")                  # score: most-likely level (argmax),
        if probs:                                         # not round(expected-value "score")
            return str(max(probs, key=lambda k: probs[k]))
        return str(int(round(float(ans["score"]))))
BACKENDS["jev"] = Jev

class Bedrock:
    """AWS Bedrock generative LLM via the Converse API (prompt -> generate -> extract)."""
    def __init__(self, model, gen_tokens=1024, **_):
        import boto3
        self.c = boto3.client("bedrock-runtime"); self.model = model; self.gt = int(gen_tokens)
    def _chat(self, prompt):
        # generous budget: reasoning models (e.g. gpt-oss, gpt-6, opus) spend tokens on a hidden
        # reasoning channel before the answer; we read only the visible `text` blocks.
        r = self.c.converse(modelId=self.model,
                            messages=[{"role": "user", "content": [{"text": prompt}]}],
                            inferenceConfig={"maxTokens": self.gt})
        return "".join(b.get("text", "") for b in r["output"]["message"]["content"])
    def predict(self, row):
        return extract(self._chat(build_prompt(row)), row)
BACKENDS["bedrock"] = Bedrock

class NLI:
    """Zero-shot entailment classifier ('other' paradigm): entailment of each candidate description."""
    def __init__(self, model, **_):
        from transformers import pipeline
        self.p = pipeline("zero-shot-classification", model=model)
    def predict(self, row):
        cands = candidates(row)
        labels = [v for _, v in cands]; keys = [k for k, _ in cands]
        res = self.p(context(row)[:2000], labels, multi_label=False)
        return keys[labels.index(res["labels"][0])]
BACKENDS["nli"] = NLI

# ----------------------------- run --------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=list(BACKENDS) + ["auto"])
    ap.add_argument("--model", default="")
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--tier", default="all", choices=["all", "gold", "silver"])
    ap.add_argument("--types", default="", help="comma list e.g. choice,noul")
    ap.add_argument("--base-url", default=None); ap.add_argument("--api-key", default=None)
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    backend = a.backend
    if backend == "auto":
        backend, why = autodetect(a.model, a.base_url)
        print(f"[auto] detected backend '{backend}' — {why}")

    rows = load_rows(a.data)
    if a.tier != "all": rows = [r for r in rows if r["reliability"] == a.tier]
    if a.types: rows = [r for r in rows if r["type"] in a.types.split(",")]
    if a.limit: rows = rows[:a.limit]
    print(f"benchmark: {len(rows)} rows | backend={backend} model={a.model or '-'}")

    runner = BACKENDS[backend](model=a.model, base_url=a.base_url, api_key=a.api_key, gen_tokens=a.gen_tokens)
    preds = []
    for i, r in enumerate(rows):
        errored = False
        try: pred = runner.predict(r)
        except Exception as e: pred = None; errored = True
        gold = str(r["label"])
        dist = None
        if r["type"] == "score" and pred is not None and str(pred).lstrip("-").isdigit():
            dist = abs(int(pred) - int(gold))
        preds.append({"id": r["id"], "type": r["type"], "domain": r.get("domain"), "errored": errored,
                      "reliability": r["reliability"], "eval_metric": r["eval_metric"],
                      "gold": gold, "pred": pred, "correct": str(pred) == gold, "distance": dist,
                      "n_levels": len(r["question"].get("levels", [])) or 2})
        if (i + 1) % 50 == 0: print(f"  ...{i+1}/{len(rows)}", flush=True)
    report(preds, a.out or None)

if __name__ == "__main__":
    main()
