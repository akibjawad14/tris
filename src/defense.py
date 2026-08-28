import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def _simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def _ngram_set(tokens, n):
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


class TriLayerSieve:
    """
    Phase-1 defense module:
    - Layer 1: semantic clustering with a judge model
    - Layer 2: overlap/repetition structural filtering
    """

    def __init__(
        self,
        judge_model_name="all-MiniLM-L6-v2",
        cluster_count=2,
        prefix_token_count=20,
        jaccard_threshold=0.8,
        ngram_threshold=0.8,
        require_clear_majority=True,
        enable_layer3=False,
        layer3_mode="selective",
        verifier_fn=None,
    ):
        self.judge_model_name = judge_model_name
        self.cluster_count = max(1, int(cluster_count))
        self.prefix_token_count = max(5, int(prefix_token_count))
        self.jaccard_threshold = float(jaccard_threshold)
        self.ngram_threshold = float(ngram_threshold)
        self.require_clear_majority = require_clear_majority
        self.enable_layer3 = enable_layer3
        self.layer3_mode = layer3_mode
        self.verifier_fn = verifier_fn
        self.judge_model = SentenceTransformer(judge_model_name)

    def filter_documents(self, query, topk_results):
        """
        topk_results: list of dicts with at least {"context": str, "score": float}
        returns: (filtered_results, diagnostics)
        """
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

        l1_results, l1_removed = self._layer1_semantic_clustering(query, topk_results)
        diagnostics["layer1_removed_idx"] = l1_removed

        l2_results, l2_removed = self._layer2_structural_filter(query, l1_results)
        diagnostics["layer2_removed_idx"] = l2_removed
        working_results = l2_results

        should_run_l3 = (
            self.enable_layer3
            and self.verifier_fn is not None
            and len(working_results) > 0
            and (
                self.layer3_mode == "always"
                or (self.layer3_mode == "selective" and (len(l1_removed) > 0 or len(l2_removed) > 0))
            )
        )
        if should_run_l3:
            working_results, l3_removed = self._layer3_consistency_filter(query, working_results)
            diagnostics["layer3_removed_idx"] = l3_removed
            diagnostics["layer3_used"] = True

        diagnostics["output_count"] = len(working_results)

        # Preserve availability: if all docs are filtered, return original list.
        if len(working_results) == 0:
            diagnostics["fallback_to_original"] = True
            diagnostics["output_count"] = len(topk_results)
            return topk_results, diagnostics

        diagnostics["fallback_to_original"] = False
        return working_results, diagnostics

    def _layer1_semantic_clustering(self, query, topk_results):
        contexts = [item["context"] for item in topk_results]
        if len(contexts) < 3:
            return topk_results, []

        emb = self.judge_model.encode(contexts, normalize_embeddings=True)
        k = min(self.cluster_count, len(contexts))
        if k < 2:
            return topk_results, []

        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb)
        counts = Counter(labels)
        majority_label, majority_size = counts.most_common(1)[0]

        clear_majority = majority_size > (len(contexts) / 2.0)
        if self.require_clear_majority and not clear_majority:
            return topk_results, []

        kept, removed_idx = [], []
        for idx, item in enumerate(topk_results):
            if labels[idx] == majority_label:
                kept.append(item)
            else:
                removed_idx.append(idx)
        return kept, removed_idx

    def _layer2_structural_filter(self, query, topk_results):
        q_tokens = _simple_tokenize(query)
        q_token_set = set(q_tokens)
        q_bigrams = _ngram_set(q_tokens, 2)
        q_trigrams = _ngram_set(q_tokens, 3)
        query_prefix = " ".join(q_tokens[: min(len(q_tokens), 12)])

        kept, removed_idx = [], []
        for idx, item in enumerate(topk_results):
            doc = item["context"]
            doc_tokens = _simple_tokenize(doc)
            prefix_tokens = doc_tokens[: self.prefix_token_count]
            prefix_set = set(prefix_tokens)

            inter = len(q_token_set.intersection(prefix_set))
            union = len(q_token_set.union(prefix_set))
            jaccard = (inter / union) if union > 0 else 0.0

            prefix_bigrams = _ngram_set(prefix_tokens, 2)
            prefix_trigrams = _ngram_set(prefix_tokens, 3)
            bigram_overlap = (
                len(q_bigrams.intersection(prefix_bigrams)) / max(1, len(q_bigrams))
            )
            trigram_overlap = (
                len(q_trigrams.intersection(prefix_trigrams)) / max(1, len(q_trigrams))
            )

            prefix_text = " ".join(prefix_tokens)
            repeat_trigger = (
                len(query_prefix) > 0 and prefix_text.count(query_prefix) >= 2
            )

            suspicious = (
                jaccard >= self.jaccard_threshold
                or bigram_overlap >= self.ngram_threshold
                or trigram_overlap >= self.ngram_threshold
                or repeat_trigger
            )
            if suspicious:
                removed_idx.append(idx)
            else:
                kept.append(item)
        return kept, removed_idx

    def _layer3_consistency_filter(self, query, topk_results):
        """
        Lightweight Phase-4 verifier:
        - ask model's parametric answer without retrieved context
        - ask if each doc's claim contradicts that belief
        - drop docs marked as contradiction/highly suspicious
        """
        internal_answer_prompt = (
            f"Question: {query}\n"
            "Answer briefly from your own knowledge only. "
            "If unsure, say 'I do not know'."
        )
        internal_answer = self.verifier_fn(internal_answer_prompt)

        kept, removed_idx = [], []
        for idx, item in enumerate(topk_results):
            doc = item["context"]
            verify_prompt = (
                f"Question: {query}\n"
                f"Model internal answer: {internal_answer}\n"
                f"Retrieved document: {doc}\n\n"
                "Is the retrieved document likely contradictory to the model internal answer "
                "or a suspicious factual manipulation? Reply with one token: "
                "SAFE or CONTRADICTORY."
            )
            verdict = str(self.verifier_fn(verify_prompt)).strip().lower()
            is_contradictory = "contradict" in verdict
            if is_contradictory:
                removed_idx.append(idx)
            else:
                kept.append(item)
        return kept, removed_idx
