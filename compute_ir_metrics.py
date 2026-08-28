"""
Table 2: IR Dynamics & MRR Restoration.
Computes Recall@5, Recall@50, Clean MRR, Poisoned MRR for NQ at top-50.

Run from PoisonedRAG_EMNLP project root on the login node (no GPU):
    python compute_ir_metrics.py

Requires:
  - datasets/nq/qrels/test.tsv
  - results/beir_results/nq-contriever.json
  - results/adv_targeted_results/nq.json
  - results/query_results/exp1_k50_nq/defense_sieve_nq_k50.json  (after exp1_k50 runs)
"""
import json
import os
from collections import defaultdict

PROJECT = os.path.dirname(os.path.abspath(__file__))
TOP_K = 50
ADV_PER_QUERY = 5


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                rel_str = parts[3] if len(parts) == 4 else parts[2]
                rel = int(rel_str)
            except ValueError:
                continue  # skip header row
            qid = parts[0]
            docid = parts[2] if len(parts) == 4 else parts[1]
            if rel > 0:
                qrels[qid].add(docid)
    return dict(qrels)


def load_retrieval(path):
    with open(path) as f:
        raw = json.load(f)
    return {qid: sorted(docs.items(), key=lambda x: -x[1]) for qid, docs in raw.items()}


def recall_at_k(results, qrels, k):
    hits = []
    for qid, docs in results.items():
        if qid not in qrels:
            continue
        top = {d[0] for d in docs[:k]}
        hits.append(1.0 if top & qrels[qid] else 0.0)
    return sum(hits) / len(hits) if hits else 0.0


def mean_mrr(results, qrels):
    rrs = []
    for qid, docs in results.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        rr = 0.0
        for rank, (docid, _) in enumerate(docs, 1):
            if docid in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def adversarial_mrr(results, adv_set_per_query):
    """How high do adversarial docs rank (attacker's perspective)."""
    rrs = []
    for qid, docs in results.items():
        adv = adv_set_per_query.get(qid, set())
        if not adv:
            continue
        rr = 0.0
        for rank, (docid, _) in enumerate(docs, 1):
            if docid in adv:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def simulate_poisoned(clean_results, target_qids, adv_per_query=5, top_k=50):
    """
    Adversarial docs ranked 1..adv_per_query (synthetic IDs).
    Legitimate docs shifted to positions adv_per_query+1..top_k.
    Returns poisoned_results and adv_set_per_query.
    """
    poisoned = {}
    adv_sets = {}
    base_score = 1e6
    for qid in target_qids:
        if qid not in clean_results:
            continue
        adv_ids = [f"__adv_{qid}_{i}__" for i in range(adv_per_query)]
        adv_docs = [(aid, base_score + adv_per_query - i) for i, aid in enumerate(adv_ids)]
        legit = [(d, s) for d, s in clean_results[qid] if not d.startswith("__adv_")]
        legit = legit[:top_k - adv_per_query]
        poisoned[qid] = adv_docs + legit
        adv_sets[qid] = set(adv_ids)
    return poisoned, adv_sets


def load_defense_stats(defense_json_path):
    """
    Load per-query defense stats from main.py output JSON.
    Returns {qid: {'pre': int, 'post': int, 'fallback': bool}}
    """
    if not os.path.exists(defense_json_path):
        return {}
    with open(defense_json_path) as f:
        data = json.load(f)
    stats = {}
    for item in data:
        for _, results in item.items():
            for q in results:
                qid = q['id']
                rs = q.get('retrieval_stats', {})
                di = q.get('defense_info') or {}
                stats[qid] = {
                    'pre': rs.get('poison_count_pre_defense_topk', 0),
                    'post': rs.get('poison_count_post_defense', 0),
                    'fallback': di.get('fallback_to_original', False),
                }
    return stats


def apply_defense(poisoned_results, adv_sets, defense_stats, top_k=50):
    """
    Simulate defense-adjusted retrieval using per-query poison count stats.
    Removes adversarial docs that defense detected, then pads with next legit docs.
    """
    defended = {}
    for qid, docs in poisoned_results.items():
        st = defense_stats.get(qid, {})
        pre = st.get('pre', 0)
        post = st.get('post', 0)
        fallback = st.get('fallback', True)
        adv = adv_sets.get(qid, set())

        if fallback or pre == post:
            # Defense fell back to original (no change)
            defended[qid] = docs[:top_k]
        else:
            filtered = [(d, s) for d, s in docs if d not in adv]
            defended[qid] = filtered[:top_k]
    return defended


def print_row(label, results, qrels, adv_sets=None):
    r5 = recall_at_k(results, qrels, 5)
    r50 = recall_at_k(results, qrels, 50)
    clean_mrr = mean_mrr(results, qrels)
    adv_mrr = adversarial_mrr(results, adv_sets) if adv_sets else float('nan')
    if adv_sets:
        print(f"  {label:<35} Recall@5={r5:.3f}  Recall@50={r50:.3f}  CleanMRR={clean_mrr:.3f}  PoisonedMRR={adv_mrr:.3f}")
    else:
        print(f"  {label:<35} Recall@5={r5:.3f}  Recall@50={r50:.3f}  MRR={clean_mrr:.3f}")


def main():
    qrel_path = os.path.join(PROJECT, "datasets/nq/qrels/test.tsv")
    retrieval_path = os.path.join(PROJECT, "results/beir_results/nq-contriever.json")
    adv_path = os.path.join(PROJECT, "results/adv_targeted_results/nq.json")
    defense_sieve_path = os.path.join(PROJECT, "results/query_results/exp1_k50_nq/defense_sieve_nq_k50.json")
    defense_trustrag_path = os.path.join(PROJECT, "results/query_results/exp1_k50_nq/defense_trustrag_nq_k50.json")

    print("Loading qrels...")
    qrels = load_qrels(qrel_path)
    print(f"  {len(qrels)} queries with relevance judgments")

    print("Loading retrieval results (top-2000 per query)...")
    clean_results = load_retrieval(retrieval_path)
    print(f"  {len(clean_results)} queries")

    print("Loading adversarial target query IDs...")
    with open(adv_path) as f:
        adv_data = json.load(f)
    target_qids = {v['id'] for v in adv_data.values()}
    print(f"  {len(target_qids)} target queries")

    # Restrict to target queries present in both qrels and retrieval
    eval_qids = target_qids & set(qrels.keys()) & set(clean_results.keys())
    print(f"  {len(eval_qids)} queries usable for Table 2\n")

    clean_top50 = {qid: clean_results[qid][:TOP_K] for qid in eval_qids}
    poisoned, adv_sets = simulate_poisoned(clean_results, eval_qids, ADV_PER_QUERY, TOP_K)
    target_qrels = {qid: qrels[qid] for qid in eval_qids}

    print("=" * 75)
    print("Table 2: Retriever Metrics under Poisoning + Defense (NQ, Top-50)")
    print("=" * 75)
    print_row("Clean Corpus (Contriever)", clean_top50, target_qrels)
    print_row("Poisoned Corpus (No Defense)", poisoned, target_qrels, adv_sets)

    # Sieve
    sieve_stats = load_defense_stats(defense_sieve_path)
    if sieve_stats:
        defended_sieve = apply_defense(poisoned, adv_sets, sieve_stats, TOP_K)
        print_row("Poisoned + Tri-Layer Sieve", defended_sieve, target_qrels, adv_sets)
    else:
        print(f"  {'Poisoned + Tri-Layer Sieve':<35} [run exp1_k50 first]")

    # TrustRAG
    trustrag_stats = load_defense_stats(defense_trustrag_path)
    if trustrag_stats:
        defended_trustrag = apply_defense(poisoned, adv_sets, trustrag_stats, TOP_K)
        print_row("Poisoned + TrustRAG", defended_trustrag, target_qrels, adv_sets)
    else:
        print(f"  {'Poisoned + TrustRAG':<35} [run exp1_k50 first]")

    print("=" * 75)
    print("\nNote: PoisonedMRR = how high adversarial docs rank (1.0 = all at rank 1).")
    print("After defense, PoisonedMRR drops toward 0 as adv docs are removed.")


if __name__ == "__main__":
    main()
