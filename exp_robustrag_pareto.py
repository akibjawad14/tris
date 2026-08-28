"""
exp_robustrag_pareto.py — fair RobustRAG ASR/CleanAcc Pareto sweep (answers Lm5W).

RobustRAG's submitted 5%-CleanAcc / 1%-ASR point is ONE over-harsh operating
point, not its frontier. Here we sweep the keyword-aggregation threshold alpha to
trace the ASR-CleanAcc trade-off, on the SAME queries/poisons as TRIS. The k
isolated per-document answers are computed once per query and reused for every
alpha, so the sweep is cheap (alpha only changes the offline aggregation + one
final call per operating point).

Usage:
    export OPENAI_API_KEY="YOUR_KEY"
    python exp_robustrag_pareto.py --dataset nq --num_queries 20 \
        --poison_file results/adv_targeted_results/nq_adaptive.json --attack_level blackbox \
        --max_docs 30 --alphas 0.1,0.2,0.3,0.5,0.7 --gpt_model gpt-3.5-turbo-0125 \
        --json_out results/robustrag_pareto/robustrag_pareto_blackbox.json
"""
import argparse
import json
import os
from collections import Counter

# ADDITIVE fix: Delta's src/baselines.py has an OLDER RobustRAGDefense stub
# (llm-only, no alpha/max_docs) that TypeErrors here. Import the working class
# from the new src/robustrag_fair.py instead — src/baselines.py is left untouched.
from src.robustrag_fair import RobustRAGDefense
from src.prompts import wrap_prompt
from pilot_cpu import GPTClient, stream_corpus_texts, load_json, clean_str, get_poisons


def build_candidates(t, rankings, corpus_text, args):
    q = t["question"]
    poisons = get_poisons(t, args)
    ranked = sorted(rankings[t["id"]].items(), key=lambda kv: -kv[1])[: args.top_k]
    cand = [{"score": s, "context": corpus_text.get(d, "")} for d, s in ranked if corpus_text.get(d)]
    top1 = cand[0]["score"] if cand else 1.0
    for i, p in enumerate(poisons):
        cand.append({"score": top1 + 1.0 - i * 0.001, "context": p})
    cand = sorted(cand, key=lambda x: -x["score"])[: args.top_k]
    return q, cand, set(poisons)


def score(rows_resp, incorrect, correct):
    asr = 1 if clean_str(incorrect) in clean_str(rows_resp) else 0
    cor = 1 if (correct and clean_str(correct) in clean_str(rows_resp)) else 0
    return asr, cor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--num_queries", type=int, default=20)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--adv_per_query", type=int, default=5)
    ap.add_argument("--poison_file", default=None)
    ap.add_argument("--attack_level", default=None,
                    choices=["blackbox", "trigger_evasion", "diversity", "full_adaptive"])
    ap.add_argument("--poison_mode", default="prefix", choices=["prefix", "verbatim"])
    ap.add_argument("--max_docs", type=int, default=30,
                    help="isolate over the top-N docs (cost control; RobustRAG uses all k)")
    ap.add_argument("--alphas", default="0.1,0.2,0.3,0.5,0.7")
    ap.add_argument("--gpt_model", default="gpt-3.5-turbo-0125")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    poison_path = args.poison_file or f"results/adv_targeted_results/{args.dataset}.json"
    poisons = load_json(poison_path)
    rankings = load_json(f"results/beir_results/{args.dataset}-contriever.json")
    targets = [v for v in poisons.values() if v["id"] in rankings]
    if args.attack_level:
        targets = [v for v in targets if args.attack_level in v.get("levels", {})]
    targets = targets[: args.num_queries]
    if not targets:
        raise SystemExit("No target queries found.")

    needed = set()
    for t in targets:
        for d, _ in sorted(rankings[t["id"]].items(), key=lambda kv: -kv[1])[: args.top_k]:
            needed.add(d)
    corpus_text = stream_corpus_texts(args.dataset, needed)
    print(f"{len(targets)} queries, streamed {len(corpus_text)} docs "
          f"(level={args.attack_level or args.poison_mode}, max_docs={args.max_docs})")

    llm = GPTClient(name=args.gpt_model, temperature=args.temperature)
    rr = RobustRAGDefense(llm=llm, method="keyword", max_docs=args.max_docs)
    wrapfn = lambda qq, docs: wrap_prompt(qq, docs, prompt_id=4)

    # Phase 1: isolated per-doc answers (the cost), cached per query.
    per_q = []
    for t in targets:
        q, cand, adv_set = build_candidates(t, rankings, corpus_text, args)
        iso = rr.isolated_answers(q, cand, wrapfn)
        per_q.append({"q": q, "iso": iso, "cand": cand, "adv_set": adv_set,
                      "incorrect": t["incorrect answer"], "correct": t.get("correct answer", "")})
        print(f"  isolated {len(per_q)}/{len(targets)} (calls={llm.calls})")
    n = len(per_q)

    rows = []
    # Reference point: undefended concatenated-context answer.
    va, vc = 0, 0
    for pq in per_q:
        resp = llm.query(wrap_prompt(pq["q"], [c["context"] for c in pq["cand"]], 4))
        a, c = score(resp, pq["incorrect"], pq["correct"])
        va += a; vc += c
    rows.append({"config": "vanilla(no-defense)", "ASR": va / n, "CleanAcc": vc / n})

    # Degenerate exact-string majority (the submitted RobustRAG operating point).
    ma, mc = 0, 0
    for pq in per_q:
        norm = [a.strip().lower() for a in pq["iso"]]
        ans = Counter(norm).most_common(1)[0][0] if norm else ""
        a, c = score(ans, pq["incorrect"], pq["correct"])
        ma += a; mc += c
    rows.append({"config": "rr-majority(submitted)", "ASR": ma / n, "CleanAcc": mc / n})

    # Fair keyword-aggregation frontier.
    for alpha in alphas:
        aa, ac = 0, 0
        for pq in per_q:
            ans, _ = rr.aggregate_keyword(pq["q"], pq["iso"], alpha)
            a, c = score(ans, pq["incorrect"], pq["correct"])
            aa += a; ac += c
        rows.append({"config": f"rr-keyword(a={alpha})", "ASR": aa / n, "CleanAcc": ac / n})

    print("\n" + "=" * 60)
    print(f"ROBUSTRAG PARETO  (dataset={args.dataset}, n={n}, "
          f"level={args.attack_level or args.poison_mode}, calls={llm.calls})")
    print("=" * 60)
    print(f"{'config':<26}{'ASR':>8}{'CleanAcc':>10}")
    for r in rows:
        print(f"{r['config']:<26}{r['ASR']:>8.3f}{r['CleanAcc']:>10.3f}")
    print("=" * 60)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"config": {k: str(v) for k, v in vars(args).items()},
                       "n": n, "openai_calls": llm.calls, "rows": rows}, f, indent=2)
        print(f"[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
