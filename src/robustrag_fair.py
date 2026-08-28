"""
robustrag_fair.py — ADDITIVE fix for E1 (do NOT edit src/baselines.py).

Why this file exists
--------------------
The round-1 RobustRAG alpha-frontier ran against the LOCAL PoisonedRAG tree,
whose `src/baselines.py` has the full `RobustRAGDefense` (method/alpha/max_docs +
isolated_answers/aggregate_keyword). The Delta copy of `src/baselines.py` is an
OLDER stub — `RobustRAGDefense(llm=None)` with only filter_documents/get_answer —
so `exp_robustrag_pareto.py` TypeErrors when it constructs
`RobustRAGDefense(method=, alpha=, max_docs=)`.

Rather than overwrite Akib's `src/baselines.py`, this module ships the working
`RobustRAGDefense` alongside it as a NEW file. `exp_robustrag_pareto.py` imports
`RobustRAGDefense` from here. Nothing of Akib's is modified.

This class is a verbatim extraction of the RobustRAG parts of the local
`src/baselines.py` — the exact code that produced `robustrag_pareto_v2.json`
(NQ, n=30: majority 0.167/0.233, keyword a=0.2 0.067/0.367). Re-running the NQ
config against this file reproduces those numbers; E1 points it at HotpotQA.
"""

import math
import re
from collections import Counter


def _tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# Compact English stopword list for RobustRAG keyword aggregation. Kept local so
# the keyword step needs no extra LLM calls (a frugal but faithful approximation
# of RobustRAG's response-keyword extraction).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "to", "from", "in", "on", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "as", "i", "you",
    "he", "she", "they", "we", "them", "him", "his", "her", "their", "our", "my",
    "what", "which", "who", "whom", "where", "when", "why", "how", "not", "no",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "have", "has", "had", "also", "there", "here",
    "than", "then", "so", "such", "into", "over", "under", "after", "before",
    "between", "answer", "question", "known", "according", "context", "document",
    "information", "based", "known", "one", "two", "some", "any", "all", "more",
    "most", "other", "only", "own", "same", "very", "just", "up", "out", "down",
    "off", "again", "further", "once", "because", "while", "during",
}


def _extract_keywords(text, max_k=12):
    """Content-word unigrams from an isolated answer, order preserved, deduped."""
    toks = _tokenize(text)
    seen, out = set(), []
    for t in toks:
        if len(t) <= 1 or t in _STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_k:
            break
    return out


class RobustRAGDefense:
    """
    RobustRAG (Xiang et al., 2024): isolate-then-aggregate decoding.

    Query the LLM once per retrieved document *in isolation*, then aggregate the
    per-document responses. Two aggregation modes:

      * method="keyword" (default, the paper's main API-friendly variant):
        secure keyword aggregation. Extract content keywords from each isolated
        answer, keep only keywords corroborated by at least `alpha` fraction of
        the documents (this is the tunable robustness knob), then produce a final
        answer constrained to the corroborated keywords. Small `alpha` -> lenient
        (high clean acc, less robust); large `alpha` -> strict (robust, lower
        clean acc). Sweeping alpha traces the ASR/CleanAcc Pareto frontier.

      * method="majority": exact-string majority vote over the isolated answers.
        Retained only to reproduce the degenerate 5%-CleanAcc operating point.

    filter_documents() is a pass-through (RobustRAG does not drop docs). Call
    get_answer(), or isolated_answers()+aggregate_keyword() to sweep alpha cheaply
    (the isolated per-doc calls are the cost and are independent of alpha).
    """

    def __init__(self, llm=None, method="keyword", alpha=0.3, max_keywords=12,
                 max_docs=None):
        self.llm = llm
        self.method = method
        self.alpha = float(alpha)
        self.max_keywords = int(max_keywords)
        # Optional cap on how many top docs to isolate over (cost control; real
        # RobustRAG isolates over all retrieved docs). None = use all.
        self.max_docs = max_docs

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

    def isolated_answers(self, question, topk_results, wrap_prompt_fn):
        """One isolated LLM answer per document. These are the expensive calls;
        cache the returned list to sweep alpha without re-querying."""
        docs = topk_results if self.max_docs is None else topk_results[: self.max_docs]
        answers = []
        for item in docs:
            prompt = wrap_prompt_fn(question, [item["context"]])
            answers.append(str(self.llm.query(prompt)).strip())
        return answers

    def aggregate_keyword(self, question, isolated_answers, alpha=None):
        """Secure keyword aggregation over pre-computed isolated answers.
        Returns (final_answer, kept_keywords). Costs one final LLM call."""
        alpha = self.alpha if alpha is None else float(alpha)
        k = len(isolated_answers)
        if k == 0:
            return "", []
        counts = Counter()
        for ans in isolated_answers:
            for kw in set(_extract_keywords(ans, self.max_keywords)):
                counts[kw] += 1
        threshold = max(1, math.ceil(alpha * k))
        kept = [kw for kw, c in counts.items() if c >= threshold]
        kept.sort(key=lambda w: -counts[w])
        kept = kept[: self.max_keywords]
        prompt = (
            "Answer the question concisely using ONLY the reliable keywords below. "
            "These keywords are corroborated across multiple independent sources; "
            "ignore any claim not supported by them. If they are insufficient to "
            "answer, say \"I don't know\".\n\n"
            f"Question: {question}\n"
            f"Reliable keywords: {', '.join(kept)}\n\nAnswer:"
        )
        return str(self.llm.query(prompt)).strip(), kept

    def get_answer(self, question, topk_results, wrap_prompt_fn):
        isolated = self.isolated_answers(question, topk_results, wrap_prompt_fn)
        if not isolated:
            return ""
        if self.method == "majority":
            norm = [a.strip().lower() for a in isolated]
            majority, _ = Counter(norm).most_common(1)[0]
            return majority
        answer, _ = self.aggregate_keyword(question, isolated, self.alpha)
        return answer
