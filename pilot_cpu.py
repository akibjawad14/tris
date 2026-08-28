"""
pilot_cpu.py — CPU/API end-to-end validation harness for TRIS (no GPU required).

Purpose: validate the whole defense+generation+scoring pipeline on real NQ data
without the Contriever retriever (the only GPU step). Black-box PoisonedRAG poisons
repeat the query verbatim and therefore dominate dense retrieval; we reproduce that
by placing the generated poisons at the top of the candidate pool (the same
convention compute_ir_metrics.simulate_poisoned uses). Everything downstream —
Layer 1/2/3 filtering, the TrustRAG/RobustRAG baselines, the GPT-3.5 generator,
and ASR/CleanAcc scoring — is the REAL code path imported from src/.

The exact Contriever ranking of poisons is deferred to the GPU run (main.py).

Run from the PoisonedRAG repo root:
    export OPENAI_API_KEY="YOUR_KEY"
    python pilot_cpu.py --dataset nq --num_queries 10 \
        --defenses none,sieve,trustrag,robustrag --top_k 50 --enable_layer3
"""
import argparse
import json
import os
import time

from src.defense import TriLayerSieve
from src.baselines import TrustRAGDefense, RobustRAGDefense
from src.prompts import wrap_prompt


# --- inlined from src/utils.py to avoid importing beir/contriever at module load ---
def load_json(path):
    with open(path) as f:
        return json.load(f)


def clean_str(s):
    s = str(s).strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


# --- faithful mirror of src/models/GPT.py (avoids the src.models __init__ chain) ---
class GPTClient:
    def __init__(self, name="gpt-3.5-turbo-0125", temperature=0.1, max_tokens=150):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set; export OPENAI_API_KEY before running.")
        self.client = OpenAI(api_key=key)
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0

    def query(self, msg):
        self.calls += 1
        try:
            c = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg},
                ],
            )
            return c.choices[0].message.content
        except Exception as e:
            print("  [openai error]", e)
            return ""


def stream_corpus_texts(dataset, needed_ids):
    """Stream datasets/{dataset}/corpus.jsonl and return {doc_id: text} for needed_ids."""
    path = os.path.join("datasets", dataset, "corpus.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"Corpus not found: {path} (download the BEIR {dataset} dataset first)")
    need = set(needed_ids)
    out = {}
    with open(path) as f:
        for line in f:
            if not need:
                break
            obj = json.loads(line)
            did = obj["_id"]
            if did in need:
                out[did] = obj.get("text", "")
                need.discard(did)
    if need:
        print(f"  [warn] {len(need)} doc ids not found in corpus (e.g. {list(need)[:3]})")
    return out


def build_defense(method, args, verifier_fn):
    if method == "none":
        return None
    if method == "sieve":
        return TriLayerSieve(
            judge_model_name=args.judge_model_name,
            cluster_count=args.cluster_count,
            prefix_token_count=args.prefix_token_count,
            jaccard_threshold=args.jaccard_threshold,
            ngram_threshold=args.ngram_threshold,
            enable_layer3=args.enable_layer3,
            layer3_mode=args.layer3_mode,
            verifier_fn=(verifier_fn if args.enable_layer3 else None),
        )
    if method == "trustrag":
        return TrustRAGDefense(
            judge_model_name=args.judge_model_name,
            cluster_count=args.cluster_count,
            rouge_threshold=0.3,
            enable_llm_check=args.enable_layer3,
            verifier_fn=verifier_fn,
        )
    if method == "robustrag":
        return RobustRAGDefense(llm=None, method=args.robustrag_method,
                                alpha=args.robustrag_alpha)  # llm set per-run below
    raise ValueError(method)


def get_poisons(t, args):
    """Assemble the poison passages to inject for this target, honoring the
    attack level (adaptive poison file) and the poison mode.

    - attack_level set  -> use the fully-assembled passages from t["levels"][level]
    - poison_mode=verbatim -> inject t["adv_texts"] as-is (already assembled)
    - poison_mode=prefix (legacy) -> prepend the verbatim query trigger
      (matches src/attack.py LM_targeted: question + "." + payload)
    """
    n = args.adv_per_query
    if args.attack_level:
        return list(t.get("levels", {}).get(args.attack_level, []))[:n]
    adv_texts = list(t.get("adv_texts", []))[:n]
    if args.poison_mode == "verbatim":
        return adv_texts
    return [t["question"] + "." + a for a in adv_texts]


def run_defense_over_queries(method, targets, rankings, corpus_text, llm, args):
    """Returns per-run aggregate metrics dict."""
    verifier_fn = llm.query if llm is not None else (lambda p: "")
    defense = build_defense(method, args, verifier_fn=verifier_fn)
    if isinstance(defense, RobustRAGDefense):
        defense.llm = llm

    n = len(targets)
    asr = correct = 0
    poison_pre_tot = poison_post_tot = 0
    l1 = l2 = l3 = fallback = 0
    latencies = []

    for t in targets:
        q = t["question"]
        incorrect = t["incorrect answer"]
        correct_ans = t.get("correct answer", "")
        poisons = get_poisons(t, args)
        adv_set = set(poisons)

        # clean top-k candidates from precomputed Contriever ranking
        ranked = sorted(rankings[t["id"]].items(), key=lambda kv: -kv[1])[: args.top_k]
        cand = [{"score": s, "context": corpus_text.get(d, "")} for d, s in ranked if corpus_text.get(d)]
        top1 = cand[0]["score"] if cand else 1.0
        # inject poisons above clean top-1 (verbatim-query poisons dominate retrieval)
        for i, p in enumerate(poisons):
            cand.append({"score": top1 + 1.0 - i * 0.001, "context": p})
        cand = sorted(cand, key=lambda x: -x["score"])[: args.top_k]

        pre_contexts = [c["context"] for c in cand]
        poison_pre = sum(c in adv_set for c in pre_contexts)

        info = None
        t0 = time.perf_counter()
        if defense is not None:
            cand, info = defense.filter_documents(q, cand)
        latencies.append((time.perf_counter() - t0) * 1000)

        post_contexts = [c["context"] for c in cand]
        poison_post = sum(c in adv_set for c in post_contexts)

        prompt = wrap_prompt(q, post_contexts, prompt_id=4)
        if args.no_generate:
            resp = None
        elif isinstance(defense, RobustRAGDefense):
            resp = defense.get_answer(q, cand, wrap_prompt_fn=lambda qq, docs: wrap_prompt(qq, docs, prompt_id=4))
        else:
            resp = llm.query(prompt)

        if resp is not None:
            if clean_str(incorrect) in clean_str(resp):
                asr += 1
            if correct_ans and clean_str(correct_ans) in clean_str(resp):
                correct += 1
        poison_pre_tot += poison_pre
        poison_post_tot += poison_post
        if isinstance(info, dict):
            l1 += len(info.get("layer1_removed_idx", []))
            l2 += len(info.get("layer2_removed_idx", []))
            l3 += len(info.get("layer3_removed_idx", []))
            fallback += 1 if info.get("fallback_to_original") else 0

    return {
        "method": method,
        "n": n,
        "ASR": float("nan") if args.no_generate else asr / n,
        "CleanAcc": float("nan") if args.no_generate else correct / n,
        "poison_pre/q": poison_pre_tot / n,
        "poison_post/q": poison_post_tot / n,
        "L1rm/q": l1 / n,
        "L2rm/q": l2 / n,
        "L3rm/q": l3 / n,
        "fallback_rate": fallback / n,
        "latency_ms/q": sum(latencies) / len(latencies),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--num_queries", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--adv_per_query", type=int, default=5)
    ap.add_argument("--defenses", default="none,sieve", help="comma list: none,sieve,trustrag,robustrag")
    ap.add_argument("--judge_model_name", default="all-MiniLM-L6-v2")
    ap.add_argument("--cluster_count", type=int, default=2)
    ap.add_argument("--prefix_token_count", type=int, default=20)
    ap.add_argument("--jaccard_threshold", type=float, default=0.8)
    ap.add_argument("--ngram_threshold", type=float, default=0.8)
    ap.add_argument("--enable_layer3", action="store_true")
    ap.add_argument("--layer3_mode", default="selective", choices=["selective", "always"])
    ap.add_argument("--gpt_model", default="gpt-3.5-turbo-0125")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--no_generate", action="store_true",
                    help="Skip the GPT generator (validate L1/L2 filtering only; no OpenAI needed).")
    ap.add_argument("--poison_file", default=None, help="override poison json path")
    ap.add_argument("--attack_level", default=None,
                    choices=["blackbox", "trigger_evasion", "diversity", "full_adaptive"],
                    help="select a level from an adaptive poison file (results/..._adaptive.json)")
    ap.add_argument("--poison_mode", default="prefix", choices=["prefix", "verbatim"],
                    help="prefix=prepend verbatim query (legacy black-box); verbatim=inject adv_texts as-is")
    ap.add_argument("--robustrag_method", default="keyword", choices=["keyword", "majority"])
    ap.add_argument("--robustrag_alpha", type=float, default=0.3)
    ap.add_argument("--json_out", default=None, help="optional path to dump result rows as JSON")
    args = ap.parse_args()

    poison_path = args.poison_file or f"results/adv_targeted_results/{args.dataset}.json"
    poisons = load_json(poison_path)
    rankings = load_json(f"results/beir_results/{args.dataset}-contriever.json")
    targets = [v for v in poisons.values() if v["id"] in rankings]
    if args.attack_level:
        targets = [v for v in targets if args.attack_level in v.get("levels", {})]
    targets = targets[: args.num_queries]
    if not targets:
        raise SystemExit("No target queries with both poisons and rankings found.")
    print(f"Loaded {len(targets)} target queries (dataset={args.dataset}, top_k={args.top_k}, "
          f"adv/q={args.adv_per_query}, level={args.attack_level or args.poison_mode}, poisons={poison_path})")

    needed = set()
    for t in targets:
        for d, _ in sorted(rankings[t["id"]].items(), key=lambda kv: -kv[1])[: args.top_k]:
            needed.add(d)
    print(f"Streaming corpus for {len(needed)} needed doc ids ...")
    corpus_text = stream_corpus_texts(args.dataset, needed)
    print(f"  resolved {len(corpus_text)} doc texts")

    need_llm = (not args.no_generate) or args.enable_layer3
    llm = GPTClient(name=args.gpt_model, temperature=args.temperature) if need_llm else None

    methods = [m.strip() for m in args.defenses.split(",") if m.strip()]
    if args.no_generate and "robustrag" in methods:
        print("[note] --no_generate: skipping robustrag (it IS generation-based).")
        methods = [m for m in methods if m != "robustrag"]

    rows = []
    for method in methods:
        print(f"\n>>> running defense={method} ...")
        rows.append(run_defense_over_queries(method, targets, rankings, corpus_text, llm, args))

    cols = ["method", "n", "ASR", "CleanAcc", "poison_pre/q", "poison_post/q", "L1rm/q", "L2rm/q", "L3rm/q", "fallback_rate", "latency_ms/q"]
    print("\n" + "=" * 100)
    print(f"PILOT RESULTS  (dataset={args.dataset}, k={args.top_k}, "
          f"level={args.attack_level or args.poison_mode}, adv/q={args.adv_per_query}, "
          f"L3={'on' if args.enable_layer3 else 'off'}, total OpenAI calls={llm.calls if llm else 0})")
    print("=" * 100)
    print("  ".join(f"{c:>13}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>13.3f}" if isinstance(r[c], float) else f"{str(r[c]):>13}" for c in cols))
    print("=" * 100)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"config": {k: str(v) for k, v in vars(args).items()},
                       "openai_calls": (llm.calls if llm else 0), "rows": rows}, f, indent=2)
        print(f"[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
