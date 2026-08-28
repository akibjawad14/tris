"""
faithful_trustrag.py  --  Retriever-space clustering defense (faithful TrustRAG variant).

Why this exists (rebuttal, reviewer Lm5W):
  TRIS Layer-1 clusters the retrieved documents in an *architecturally independent*
  embedding space (Sentence-BERT all-MiniLM-L6-v2) and keeps the majority cluster.
  TrustRAG (Zhou et al., 2025) clusters in the *retriever's own* space. Lm5W asked
  whether the "independent judge geometry" is actually doing anything. This class runs
  the SAME majority-keep K-Means rule as TRIS L1 but embeds with the Contriever
  retriever passed in via `embed_fn`. Running it head-to-head against `sieve` (which
  uses MiniLM) therefore isolates exactly one variable: the embedding geometry.

Interface matches src/defense.py TriLayerSieve so main_adaptive.py can treat them
interchangeably:
    filter_documents(question, topk_results) -> (kept_results, diagnostics)
where topk_results is a list of {"score": float, "context": str} and diagnostics
carries the same keys main.py logs (layer1_removed_idx, layer2_removed_idx,
layer3_removed_idx, fallback_to_original).
"""
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception as e:  # pragma: no cover
    raise ImportError("faithful_trustrag requires scikit-learn (sklearn.cluster.KMeans)") from e


class FaithfulTrustRAG:
    def __init__(self, embed_fn, cluster_count=2, require_clear_majority=True, seed=12):
        """
        embed_fn: callable(list[str]) -> np.ndarray of shape (n, d), using the RETRIEVER
                  (Contriever) encoder. This is what makes the clustering happen in the
                  retriever's own geometry.
        cluster_count / require_clear_majority: kept identical to TriLayerSieve L1 so the
                  only difference from TRIS is the embedding space.
        """
        self.embed_fn = embed_fn
        self.cluster_count = int(cluster_count)
        self.require_clear_majority = bool(require_clear_majority)
        self.seed = int(seed)

    @staticmethod
    def _l2norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def _diag(self, n_in, n_out, removed_idx, fallback):
        return {
            "input_count": n_in,
            "output_count": n_out,
            "layer1_removed_idx": list(removed_idx),
            "layer2_removed_idx": [],
            "layer3_removed_idx": [],
            "fallback_to_original": bool(fallback),
            "defense": "faithful_trustrag",
        }

    def filter_documents(self, question, topk_results):
        n = len(topk_results)
        contexts = [d["context"] for d in topk_results]

        # Too few docs to cluster meaningfully -> retain all (same as TRIS L1 behaviour).
        if n <= self.cluster_count or n <= 2:
            return topk_results, self._diag(n, n, [], fallback=True)

        embs = np.asarray(self.embed_fn(contexts), dtype=np.float32)
        if embs.ndim != 2 or embs.shape[0] != n:
            # Embedding failed / shape mismatch -> fail open (never remove blindly).
            return topk_results, self._diag(n, n, [], fallback=True)
        embs = self._l2norm(embs)

        k = min(self.cluster_count, n)
        labels = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit_predict(embs)

        counts = np.bincount(labels, minlength=k)
        max_size = counts.max()
        majority_labels = np.where(counts == max_size)[0]

        # No unique majority (tie) -> conservative retain-all, matching TRIS L1.
        if self.require_clear_majority and len(majority_labels) != 1:
            return topk_results, self._diag(n, n, [], fallback=True)

        keep_label = int(majority_labels[0])
        kept, removed_idx = [], []
        for i, lab in enumerate(labels):
            if lab == keep_label:
                kept.append(topk_results[i])
            else:
                removed_idx.append(i)

        # Safety: never return an empty candidate set.
        if not kept:
            return topk_results, self._diag(n, n, [], fallback=True)

        return kept, self._diag(n, len(kept), removed_idx, fallback=False)
