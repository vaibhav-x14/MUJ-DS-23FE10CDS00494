import math
import re
from typing import List, Dict, Set, Tuple
from collections import Counter
from omnirag.core.schemas import Chunk
from config.settings import settings

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class BM25Retriever:
    """Production BM25Okapi sparse lexical retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        cfg = settings.retrieval.sparse_bm25
        self.k1 = k1 or cfg.k1
        self.b = b or cfg.b
        self.corpus: List[Chunk] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return [w for w in words if w not in STOPWORDS]

    def index(self, chunks: List[Chunk]) -> None:
        """Indexes a collection of document chunks."""
        self.corpus = chunks
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = {}

        num_docs = len(chunks)
        if num_docs == 0:
            return

        total_len = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            t_len = len(tokens)
            self.doc_lens.append(t_len)
            total_len += t_len

            counts = Counter(tokens)
            self.term_freqs.append(counts)

            for term in counts.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = total_len / num_docs if num_docs > 0 else 0.0

        # Calculate BM25 IDF with smoothing
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard Lucene/BM25Okapi IDF formula
            self.idf[term] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Scores all indexed chunks against query and returns top_k ranked tuples."""
        if not self.corpus:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return [(self.corpus[i], 0.0) for i in range(min(top_k, len(self.corpus)))]

        scores: List[float] = [0.0] * len(self.corpus)

        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]

            for i, tf_dict in enumerate(self.term_freqs):
                if term in tf_dict:
                    freq = tf_dict[term]
                    doc_len = self.doc_lens[i]
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                    scores[i] += term_idf * (numerator / denominator)

        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results: List[Tuple[Chunk, float]] = []
        for idx in ranked_indices[:top_k]:
            score = scores[idx]
            # Copy chunk and record score
            chunk_copy = self.corpus[idx].model_copy(update={"bm25_score": round(score, 4)})
            results.append((chunk_copy, score))

        return results
