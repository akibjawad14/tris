#!/usr/bin/env python3
"""
gen_adaptive_poisons_delta.py  --  Adaptive (trigger-paraphrase) poison generator.

Purpose (rebuttal, GPU testbed):
  The black-box PoisonedRAG attack prepends the *verbatim* query as the retrieval
  trigger (src/attack.py: adv_text = question + "." + payload). This is exactly what
  TRIS's Layer-2 structural filter catches. The adaptive attacker instead prepends a
  *paraphrase* of the query -- evading the n-gram/Jaccard check -- but a paraphrase is
  less similar to the query in the retriever's space, so it should also lose retrieval
  rank. This script produces the paraphrased trigger so main_adaptive.py can measure
  that real retrieval penalty on the GPU.

What it does:
  Reads the paper poison file  results/adv_targeted_results/{dataset}.json  and, for
  each query, adds one field: "paraphrased_question". Everything else (id, question,
  correct answer, incorrect answer, adv_texts) is copied UNCHANGED, so the payloads are
  held constant and the only variable is the trigger. Writes {dataset}_adaptive.json.

Cost: one short chat call per query (~100/dataset) on gpt-3.5-turbo => a few cents.
Resumable: rerun to fill only the entries still missing a paraphrase.

Key: read from $OPENAI_API_KEY, else from the api_key_info block of --model_config.
No key value is ever printed.
"""
import argparse
import json
import os
import sys
import time


def load_api_key(model_config_path):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    # Fallback: read from the repo's model config (same place main.py's LLM reads it).
    try:
        with open(model_config_path) as f:
            cfg = json.load(f)
        info = cfg["api_key_info"]
        keys = info["api_keys"]
        idx = info.get("api_key_use", 0)
        idx = 0 if idx in (None, -1) else idx
        key = str(keys[idx]).strip()
    except Exception as e:
        sys.exit(f"[fatal] No OPENAI_API_KEY in env and could not read a key from "
                 f"{model_config_path}: {e}")
    if not key or key.upper().startswith("YOUR_API_KEY"):
        sys.exit("[fatal] API key is empty or still the placeholder. Set OPENAI_API_KEY "
                 "or put your key in the model config first.")
    return key


PARA_SYS = "You rewrite search queries. Output ONLY the rewritten query on one line, nothing else."
PARA_USER = (
    "Rewrite this search query so it keeps the exact same meaning and topic but uses "
    "different words and a different sentence structure. Do not answer it, do not add "
    "quotes. Keep it to a single line.\n\nQuery: {q}"
)


def paraphrase(client, model, question, temperature, max_retries=4):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PARA_SYS},
                    {"role": "user", "content": PARA_USER.format(q=question)},
                ],
                temperature=temperature,
                max_tokens=64,
            )
            out = (resp.choices[0].message.content or "").strip().strip('"').strip()
            # Guard against a degenerate echo of the original.
            if out and out.lower() != question.strip().lower():
                return out
            if out:
                return out  # accept even if similar; main.py still measures the effect
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}: {e} "
                  f"-> sleeping {wait}s", flush=True)
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nq", "hotpotqa", "msmarco"])
    ap.add_argument("--in_path", default=None,
                    help="default results/adv_targeted_results/{dataset}.json")
    ap.add_argument("--out_path", default=None,
                    help="default results/adv_targeted_results/{dataset}_adaptive.json")
    ap.add_argument("--model", default="gpt-3.5-turbo-0125")
    ap.add_argument("--model_config", default="model_configs/gpt3.5_config.json")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="0 = all queries")
    ap.add_argument("--save_every", type=int, default=10)
    args = ap.parse_args()

    in_path = args.in_path or f"results/adv_targeted_results/{args.dataset}.json"
    out_path = args.out_path or f"results/adv_targeted_results/{args.dataset}_adaptive.json"

    if not os.path.exists(in_path):
        sys.exit(f"[fatal] input poison file not found: {in_path}")

    from openai import OpenAI
    client = OpenAI(api_key=load_api_key(args.model_config))

    with open(in_path) as f:
        data = json.load(f)

    # Resume from any existing output.
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"[resume] loaded {len(out)} existing entries from {out_path}")
    else:
        out = {k: dict(v) for k, v in data.items()}

    keys = list(data.keys())
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    todo = [k for k in keys
            if not str(out.get(k, {}).get("paraphrased_question", "")).strip()]
    print(f"[plan] {len(keys)} queries in scope, {len(todo)} need a paraphrase "
          f"(model={args.model})")

    done = 0
    for i, k in enumerate(keys):
        entry = out.setdefault(k, dict(data[k]))
        if str(entry.get("paraphrased_question", "")).strip():
            continue
        q = data[k]["question"]
        pp = paraphrase(client, args.model, q, args.temperature)
        if pp is None:
            print(f"  [{i+1}/{len(keys)}] {k}: FAILED after retries; leaving blank")
            continue
        entry["paraphrased_question"] = pp
        done += 1
        print(f"  [{i+1}/{len(keys)}] {k}: {q!r} -> {pp!r}")
        if done % args.save_every == 0:
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    filled = sum(1 for k in keys if str(out[k].get("paraphrased_question", "")).strip())
    print(f"\n[done] wrote {out_path}")
    print(f"[done] {filled}/{len(keys)} entries have a paraphrased_question "
          f"({done} generated this run)")
    if filled < len(keys):
        print("[warn] some entries still blank -- rerun to fill them.")


if __name__ == "__main__":
    main()
