"""TF-IDF based memory router for selective memory retrieval.

Zero external dependencies -- pure Python using stdlib only.
Relevance-ranked entry selection via cosine similarity on TF-IDF vectors.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional

_STOP_WORDS = frozenset({
    "bir", "bu", "ve", "veya", "ile", "icin", "ama", "gibi", "kadar",
    "sonra", "once", "ancak", "cunku", "eger", "henuz", "daha",
    "hem", "hic", "ise", "kendi", "ne", "nasil", "nicin",
    "nerede", "nereden", "nereli", "olan", "olarak", "oldugu",
    "olmak", "uzere", "sey", "tum", "ya", "the", "this", "that",
    "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "shall", "should", "may", "might", "must", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "and", "or",
    "but", "nor", "not", "so", "if", "then", "than", "too", "very",
    "just", "also", "about", "up", "down", "out", "off", "over",
    "under", "again", "further", "once", "here", "there", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "it", "its", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our",
})


def _tokenize(text: str) -> List[str]:
    """Tokenize, lowercase, remove stop-words and short tokens."""
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def _cosine_similarity(
    query_tokens: List[str],
    doc_tokens: List[str],
    idf: Dict[str, float],
    n_docs: int,
) -> float:
    """Cosine similarity between query and document TF-IDF vectors."""
    query_tf = Counter(query_tokens)
    doc_tf = Counter(doc_tokens)
    all_terms = set(query_tf.keys()) | set(doc_tf.keys())

    dot, q_norm, d_norm = 0.0, 0.0, 0.0
    fallback_idf = math.log((n_docs + 1) / 1)

    for term in all_terms:
        q_tfidf = query_tf.get(term, 0) * idf.get(term, fallback_idf)
        d_tfidf = doc_tf.get(term, 0) * idf.get(term, fallback_idf)
        dot += q_tfidf * d_tfidf
        q_norm += q_tfidf * q_tfidf
        d_norm += d_tfidf * d_tfidf

    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))


def tf_idf_rank_indices(query: str, entries: List[str], top_k: int = 5) -> List[int]:
    """Rank entry indices by TF-IDF cosine similarity to query.

    Returns list of indices into *entries*, sorted by relevance descending.
    When query is empty or no tokens, returns all indices (no filtering).
    """
    if not entries:
        return []
    if not query or not query.strip():
        return list(range(len(entries)))

    query_tokens = _tokenize(query)
    if not query_tokens:
        return list(range(len(entries)))

    doc_tokens_list = [_tokenize(e) for e in entries]
    n_docs = len(doc_tokens_list)

    doc_freq: Counter = Counter()
    for dt in doc_tokens_list:
        for term in set(dt):
            doc_freq[term] += 1

    idf = {t: math.log((n_docs + 1) / (f + 1)) + 1 for t, f in doc_freq.items()}

    scored = [
        (idx, _cosine_similarity(query_tokens, dt, idf, n_docs))
        for idx, dt in enumerate(doc_tokens_list)
    ]
    scored.sort(key=lambda x: -x[1])
    return [idx for idx, _ in scored[:top_k]]


class MemoryRouter:
    """Routes user queries to relevant memory entries using TF-IDF."""

    def __init__(self, memory_store, top_k: int = 5):
        self._store = memory_store
        self._top_k = top_k

    def route(self, query: str) -> Dict[str, Optional[str]]:
        """Select relevant memory blocks based on query.

        Args:
            query: User's question or message text.

        Returns:
            Dict mapping target names to formatted memory blocks.
            Empty dict when no memory is relevant.
        """
        result: Dict[str, Optional[str]] = {}
        for target in ["memory", "user"]:
            block = self._store.select_for_query(target, query, top_k=self._top_k)
            if block:
                result[target] = block
        return result
