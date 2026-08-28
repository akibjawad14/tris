"""
main_adaptive.py  --  Adaptive-attack + defense driver for the TRIS rebuttal GPU testbed.

Derived from main.py (kept faithful to its methodology and metrics). Three additions:

  1. --poison_file      : load poisons from an explicit path (e.g. the *_adaptive.json
                          produced by gen_adaptive_poisons_delta.py) instead of the
                          hardcoded results/adv_targeted_results/{dataset}.json.
  2. --trigger_mode     : 'verbatim'   -> trigger = the exact query   (paper black-box)
                          'paraphrase' -> trigger = paraphrased_question (adaptive attack)
                          Payloads (adv_texts) are held constant, so the ONLY variable is
                          the trigger => main.py's existing per-query retrieval stats
                          (poison_count/ranks pre-defense) directly measure the retrieval
                          penalty of paraphrasing.
  3. --defense_method faithful_trustrag : retriever-space (Contriever) majority-keep
                          clustering, to compare against TRIS's independent MiniLM judge.

GPU routing is unchanged from main.py: torch.cuda.set_device(gpu_id); device='cuda'.
Writes a compact metrics summary to --summary_out for easy report-back.
"""
import argparse
import os
import json
import time
import numpy as np
import torch

from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.defense import TriLayerSieve
from src.baselines import TrustRAGDefense, RobustRAGDefense
from src.faithful_trustrag import FaithfulTrustRAG
from src.prompts import wrap_prompt


def parse_args():
    p = argparse.ArgumentParser(description='TRIS adaptive-attack GPU driver')
    # Retriever / dataset
    p.add_argument("--eval_model_code", type=str, default="contriever")
    p.add_argument('--eval_dataset', type=str, default="nq")
    p.add_argument('--split', type=str, default='test')
    p.add_argument("--query_results_dir", type=str, default='adaptive')

    # LLM
    p.add_argument('--model_config_path', default=None, type=str)
    p.add_argument('--model_name', type=str, default='gpt3.5')
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--gpu_id', type=int, default=0)

    # Defense
    p.add_argument('--enable_defense', action='store_true')
    p.add_argument('--judge_model_name', type=str, default='all-MiniLM-L6-v2')
    p.add_argument('--cluster_count', type=int, default=2)
    p.add_argument('--prefix_token_count', type=int, default=20)
    p.add_argument('--jaccard_threshold', type=float, default=0.8)
    p.add_argument('--ngram_threshold', type=float, default=0.8)
    p.add_argument('--enable_layer3', action='store_true')
    p.add_argument('--layer3_mode', type=str, default='selective', choices=['selective', 'always'])
    p.add_argument('--defense_method', type=str, default='sieve',
                   choices=['sieve', 'trustrag', 'robustrag', 'faithful_trustrag'])

    # Attack (adaptive additions)
    p.add_argument('--poison_file', type=str, default=None,
                   help='explicit poison JSON; default results/adv_targeted_results/{dataset}.json')
    p.add_argument('--trigger_mode', type=str, default='verbatim',
                   choices=['verbatim', 'paraphrase'])
    p.add_argument('--adv_per_query', type=int, default=5)
    p.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    p.add_argument('--repeat_times', type=int, default=10)
    p.add_argument('--M', type=int, default=10)
    p.add_argument('--seed', type=int, default=12)
    p.add_argument("--name", type=str, default='debug')
    p.add_argument("--summary_out", type=str, default=None,
                   help='write a compact metrics summary JSON here (for report-back)')

    args = p.parse_args()
    print(args)
    return args


def build_trigger(entry, trigger_mode):
    """Return (trigger_text, used_paraphrase_bool)."""
    if trigger_mode == 'paraphrase':
        pp = str(entry.get('paraphrased_question', '')).strip()
        if pp:
            return pp, True
        # Fall back to verbatim if a paraphrase is missing; counted + warned.
        return entry['question'], False
    return entry['question'], False


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # Datasets
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets("msmarco", "train")
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    # Poisons (explicit file for the adaptive variants)
    poison_path = args.poison_file or f'results/adv_targeted_results/{args.eval_dataset}.json'
    if not os.path.exists(poison_path):
        raise FileNotFoundError(f"poison file not found: {poison_path}")
    incorrect_answers = list(load_json(poison_path).values())
    print(f"Loaded {len(incorrect_answers)} poisoned queries from {poison_path} "
          f"(trigger_mode={args.trigger_mode})")

    # Clamp iterations to what the poison file actually contains, so a short file
    # (fewer than repeat_times*M entries) degrades gracefully instead of IndexError.
    max_iters = len(incorrect_answers) // args.M
    if max_iters < 1:
        raise ValueError(f"poison file has {len(incorrect_answers)} entries but M={args.M}")
    if args.repeat_times > max_iters:
        print(f"[warn] repeat_times={args.repeat_times} exceeds available data "
              f"({len(incorrect_answers)} entries / M={args.M}); clamping to {max_iters}")
        args.repeat_times = max_iters

    # Precomputed clean retrieval rankings
    orig_beir_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
    if args.score_function == 'cos_sim':
        orig_beir_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
    with open(orig_beir_path) as f:
        results = json.load(f)
    print('Total samples:', len(results))

    # Retriever models are needed for (a) embedding poisons and (b) the faithful
    # (retriever-space) clustering defense.
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval(); model.to(device)
    c_model.eval(); c_model.to(device)

    def contriever_embed(texts):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        inp = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            emb = get_emb(c_model, inp)
        return emb.detach().cpu().numpy()

    # LLM + defense
    llm = create_model(args.model_config_path)
    defense = None
    if args.enable_defense:
        verifier_fn = (lambda prompt: llm.query(prompt)) if args.enable_layer3 else None
        if args.defense_method == 'sieve':
            defense = TriLayerSieve(
                judge_model_name=args.judge_model_name,
                cluster_count=args.cluster_count,
                prefix_token_count=args.prefix_token_count,
                jaccard_threshold=args.jaccard_threshold,
                ngram_threshold=args.ngram_threshold,
                enable_layer3=args.enable_layer3,
                layer3_mode=args.layer3_mode,
                verifier_fn=verifier_fn,
            )
        elif args.defense_method == 'trustrag':
            defense = TrustRAGDefense(
                judge_model_name=args.judge_model_name,
                cluster_count=args.cluster_count,
                rouge_threshold=0.3,
                enable_llm_check=args.enable_layer3,
                verifier_fn=verifier_fn,
            )
        elif args.defense_method == 'faithful_trustrag':
            defense = FaithfulTrustRAG(
                embed_fn=contriever_embed,
                cluster_count=args.cluster_count,
                require_clear_majority=True,
                seed=args.seed,
            )
        elif args.defense_method == 'robustrag':
            defense = RobustRAGDefense(llm=llm)

    # Accumulators
    all_results, asr_list, correct_acc_list = [], [], []
    defense_latency_ms = []
    poison_before_counts, poison_after_counts = [], []
    l1_removed_counts, l2_removed_counts, l3_removed_counts, fallback_flags = [], [], [], []
    ret_list = []
    paraphrase_missing = 0

    for it in range(args.repeat_times):
        print(f'######################## Iter: {it+1}/{args.repeat_times} #######################')
        idx_range = range(it * args.M, it * args.M + args.M)

        # Build this iter's poisons inline (verbatim or paraphrased trigger).
        adv_text_groups = []
        for i in idx_range:
            entry = incorrect_answers[i]
            payloads = entry['adv_texts'][:args.adv_per_query]
            trigger, used_pp = build_trigger(entry, args.trigger_mode)
            if args.trigger_mode == 'paraphrase' and not used_pp:
                paraphrase_missing += 1
            adv_texts = [trigger + "." + pl for pl in payloads]
            adv_text_groups.append(adv_texts)
        adv_text_list = sum(adv_text_groups, [])

        adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
        adv_input = {k: v.cuda() for k, v in adv_input.items()}
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)

        asr_cnt, correct_cnt = 0, 0
        ret_sublist, iter_results = [], []

        for i in idx_range:
            iter_idx = i - it * args.M
            entry = incorrect_answers[i]
            question = entry['question']
            qid = entry['id']
            print(f'############# Q {iter_idx+1}/{args.M}: {question} #############')

            incco_ans = entry['incorrect answer']

            # Clean top-k pool from precomputed rankings.
            topk_idx = list(results[qid].keys())[:args.top_k]
            topk_results = [{'score': results[qid][idx], 'context': corpus[idx]['text']}
                            for idx in topk_idx]

            # Score this iter's poisons against the query with the REAL retriever.
            query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
            query_input = {k: v.cuda() for k, v in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)
            for j in range(len(adv_text_list)):
                adv_emb = adv_embs[j, :].unsqueeze(0)
                if args.score_function == 'dot':
                    adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                else:
                    adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                topk_results.append({'score': adv_sim, 'context': adv_text_list[j]})
            adv_text_set = set(adv_text_groups[iter_idx])

            topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
            pre_defense_topk = topk_results[:args.top_k]
            pre_contents = [d["context"] for d in pre_defense_topk]
            poison_pre = sum(c in adv_text_set for c in pre_contents)
            poison_pre_ranks = [r + 1 for r, c in enumerate(pre_contents) if c in adv_text_set]

            topk_results = pre_defense_topk
            defense_info = None
            if defense is not None:
                _t0 = time.perf_counter()
                topk_results, defense_info = defense.filter_documents(question, topk_results)
                defense_latency_ms.append((time.perf_counter() - _t0) * 1000)

            topk_contents = [d["context"] for d in topk_results]
            poison_post = sum(c in adv_text_set for c in topk_contents)
            poison_post_ranks = [r + 1 for r, c in enumerate(topk_contents) if c in adv_text_set]

            ret_sublist.append(poison_post)
            poison_before_counts.append(poison_pre)
            poison_after_counts.append(poison_post)
            if isinstance(defense_info, dict):
                l1_removed_counts.append(len(defense_info.get("layer1_removed_idx", [])))
                l2_removed_counts.append(len(defense_info.get("layer2_removed_idx", [])))
                l3_removed_counts.append(len(defense_info.get("layer3_removed_idx", [])))
                fallback_flags.append(1 if defense_info.get("fallback_to_original", False) else 0)

            query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
            if isinstance(defense, RobustRAGDefense):
                response = defense.get_answer(
                    question, topk_results,
                    wrap_prompt_fn=lambda q, docs: wrap_prompt(q, docs, prompt_id=4))
            else:
                response = llm.query(query_prompt)
            print(f'Output: {response}\n')

            correct_ans = entry.get('correct answer', '')
            if correct_ans and clean_str(correct_ans) in clean_str(response):
                correct_cnt += 1
            if clean_str(incco_ans) in clean_str(response):
                asr_cnt += 1

            iter_results.append({
                "id": qid, "question": question,
                "trigger_mode": args.trigger_mode,
                "retrieval_stats": {
                    "poison_count_pre_defense_topk": poison_pre,
                    "poison_ranks_pre_defense_topk": poison_pre_ranks,
                    "poison_count_post_defense": poison_post,
                    "poison_ranks_post_defense": poison_post_ranks,
                    "topk_size_post_defense": len(topk_contents),
                },
                "defense_info": defense_info,
                "output_poison": response,
                "incorrect_answer": incco_ans,
                "answer": correct_ans,
            })

        asr_list.append(asr_cnt)
        correct_acc_list.append(correct_cnt)
        ret_list.append(ret_sublist)
        all_results.append({f'iter_{it}': iter_results})
        save_results(all_results, args.query_results_dir, args.name)

    # Aggregate
    asr = np.array(asr_list) / args.M
    asr_mean = round(float(np.mean(asr)), 4)
    correct_acc_mean = round(float(np.mean(np.array(correct_acc_list) / args.M)), 4)
    poison_pre_mean = round(float(np.mean(poison_before_counts)), 4) if poison_before_counts else 0.0
    poison_post_mean = round(float(np.mean(poison_after_counts)), 4) if poison_after_counts else 0.0
    l1_mean = round(float(np.mean(l1_removed_counts)), 4) if l1_removed_counts else 0.0
    l2_mean = round(float(np.mean(l2_removed_counts)), 4) if l2_removed_counts else 0.0
    l3_mean = round(float(np.mean(l3_removed_counts)), 4) if l3_removed_counts else 0.0
    fallback_rate = round(float(np.mean(fallback_flags)), 4) if fallback_flags else 0.0
    lat_mean = round(float(np.mean(defense_latency_ms)), 2) if defense_latency_ms else 0.0

    print("\n==================== ADAPTIVE RESULT ====================")
    print(f"dataset={args.eval_dataset} trigger={args.trigger_mode} "
          f"defense={args.defense_method if args.enable_defense else 'none'} "
          f"top_k={args.top_k} n={len(asr_list)*args.M}")
    print(f"ASR_mean={asr_mean}  CleanAcc_mean={correct_acc_mean}")
    print(f"poison_in_topk_pre_defense_mean={poison_pre_mean}  post_defense_mean={poison_post_mean}")
    print(f"L1_removed/q={l1_mean}  L2_removed/q={l2_mean}  L3_removed/q={l3_mean}  "
          f"fallback_rate={fallback_rate}  defense_latency_ms/q={lat_mean}")
    if paraphrase_missing:
        print(f"[warn] {paraphrase_missing} query-instances lacked a paraphrase and fell "
              f"back to verbatim -- rerun the generator to fill them.")
    print("========================================================")

    if args.summary_out:
        summary = {
            "dataset": args.eval_dataset,
            "trigger_mode": args.trigger_mode,
            "defense": args.defense_method if args.enable_defense else "none",
            "top_k": args.top_k,
            "n_queries": len(asr_list) * args.M,
            "adv_per_query": args.adv_per_query,
            "asr_mean": asr_mean,
            "cleanacc_mean": correct_acc_mean,
            "poison_in_topk_pre_defense_mean": poison_pre_mean,
            "poison_in_topk_post_defense_mean": poison_post_mean,
            "l1_removed_per_q": l1_mean,
            "l2_removed_per_q": l2_mean,
            "l3_removed_per_q": l3_mean,
            "fallback_rate": fallback_rate,
            "defense_latency_ms_per_q": lat_mean,
            "paraphrase_missing_instances": paraphrase_missing,
            "poison_file": poison_path,
        }
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True) if os.path.dirname(args.summary_out) else None
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[summary] wrote {args.summary_out}")


if __name__ == '__main__':
    main()
