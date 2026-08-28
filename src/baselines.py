"""
Baseline defense implementations for Table 1 comparison.

TrustRAGDefense  — approximation of TrustRAG (Xiang et al., 2024):
                   K-means + ROUGE-L + optional LLM self-assessment.
RobustRAGDefense — RobustRAG (Shi et al., 2024): isolate-then-aggregate voting.
                   Does NOT filter docs; changes answer generation. Use get_answer().

Both share the filter_documents(query, topk_results) interface with TriLayerSieve
for drop-in compatibility with main.py.
"""

import re
from collections import Counter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def _tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def _lcs_length(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_l_f1(hyp_tokens, ref_tokens):
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    prec = lcs / len(hyp_tokens)
    rec = lcs / len(ref_tokens)
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else 0.0


class TrustRAGDefense:
    """
    K-means semantic clustering (L1) + ROUGE-L prefix filter (L2-equivalent)
    + optional LLM self-assessment (L3-equivalent).

    Differs from TriLayerSieve in L2: uses ROUGE-L rather than Jaccard/n-gram,
    making it weaker against fluent stylistic mimicry attacks.
    """

    def __init__(
        self,
        judge_model_name="all-MiniLM-L6-v2",
        cluster_count=2,
        rouge_threshold=0.3,
        enable_llm_check=False,
        verifier_fn=None,
    ):
        self.judge_model = SentenceTransformer(judge_model_name)
        self.cluster_count = max(1, int(cluster_count))
        self.rouge_threshold = float(rouge_threshold)
        self.enable_llm_check = enable_llm_check
        self.verifier_fn = verifier_fn

    def filter_documents(self, query, topk_results):
        diagnostics = {
            "input_count": len(topk_results),
            "layer1_removed_idx": [],
            "layer2_removed_idx": [],
            "layer3_removed_idx": [],
            "layer3_used": False,
        }
        if not topk_results:
            diagnostics["output_count"] = 0
            return topk_results, diagnostics

        l1_results, l1_removed = self._layer1_cluster(query, topk_results)
        diagnostics["layer1_removed_idx"] = l1_removed

        l2_results, l2_removed = self._layer2_rouge(query, l1_results)
        diagnostics["layer2_removed_idx"] = l2_removed
        working = l2_results

        if self.enable_llm_check and self.verifier_fn and working:
            working, l3_removed = self._layer3_llm(query, working)
            diagnostics["layer3_removed_idx"] = l3_removed
            diagnostics["layer3_used"] = True

        if not working:
            diagnostics["fallback_to_original"] = True
            diagnostics["output_count"] = len(topk_results)
            return topk_results, diagnostics

        diagnostics["fallback_to_original"] = False
        diagnostics["output_count"] = len(working)
        return working, diagnostics

    def _layer1_cluster(self, _query, topk_results):
        contexts = [r["context"] for r in topk_results]
        if len(contexts) < 3:
            return topk_results, []
        k = min(self.cluster_count, len(contexts))
        if k < 2:
            return topk_results, []
        emb = self.judge_model.encode(contexts, normalize_embeddings=True)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb)
        counts = Counter(labels)
        majority_label, majority_size = counts.most_common(1)[0]
        if majority_size <= len(contexts) / 2.0:
            return topk_results, []
        kept, removed = [], []
        for idx, item in enumerate(topk_results):
            if labels[idx] == majority_label:
                kept.append(item)
            else:
                removed.append(idx)
        return kept, removed

    def _layer2_rouge(self, query, topk_results):
        q_tokens = _tokenize(query)
        kept, removed = [], []
        for idx, item in enumerate(topk_results):
            doc_prefix = _tokenize(item["context"])[:30]
            rouge = _rouge_l_f1(doc_prefix, q_tokens)
            if rouge >= self.rouge_threshold:
                removed.append(idx)
            else:
                kept.append(item)
        return kept, removed

    def _layer3_llm(self, query, topk_results):
        internal = self.verifier_fn(
            f"Question: {query}\n"
            "Answer briefly from your own knowledge only. If unsure, say 'I do not know'."
        )
        kept, removed = [], []
        for idx, item in enumerate(topk_results):
            verdict = str(self.verifier_fn(
                f"Question: {query}\nModel internal answer: {internal}\n"
                f"Retrieved document: {item['context']}\n\n"
                "Is the retrieved document contradictory or a suspicious factual manipulation? "
                "Reply with one token: SAFE or CONTRADICTORY."
            )).strip().lower()
            if "contradict" in verdict:
                removed.append(idx)
            else:
                kept.append(item)
        return kept, removed


class RobustRAGDefense:
    """
    Isolate-then-aggregate: query LLM once per document independently,
    then select the majority answer by string matching.

    filter_documents() is a pass-through (no filtering).
    Call get_answer() instead of the normal llm.query() path in main.py.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def filter_documents(self, _query, topk_results):
        diagnostics = {
            "input_count": len(topk_results),
            "output_count": len(topk_results),
            "layer1_removed_idx": [],
            "layer2_removed_idx": [],
            "layer3_removed_idx": [],
            "layer3_used": False,
            "fallback_to_original": False,
        }
        return topk_results, diagnostics

    def get_answer(self, question, topk_results, wrap_prompt_fn):
        """
        Query the LLM once per document in isolation, then majority-vote.
        wrap_prompt_fn(question, [doc_text]) -> prompt string
        """
        answers = []
        for item in topk_results:
            prompt = wrap_prompt_fn(question, [item["context"]])
            ans = str(self.llm.query(prompt)).strip().lower()
            answers.append(ans)
        if not answers:
            return ""
        majority, _ = Counter(answers).most_common(1)[0]
        return majority
