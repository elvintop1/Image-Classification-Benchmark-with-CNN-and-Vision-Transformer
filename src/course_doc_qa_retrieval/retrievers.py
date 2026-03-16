from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Document, QueryExample, simple_tokenize


@dataclass(slots=True)
class SearchHit:
    doc_id: str
    score: float
    rank: int
    title: str
    text: str


class BaseRetriever:
    name = "base"

    def fit(self, corpus: list[Document]) -> None:
        raise NotImplementedError

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[Document] = []
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.doc_lengths: list[int] = []
        self.tokenized_corpus: list[list[str]] = []

    def fit(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        df_counts: defaultdict[str, int] = defaultdict(int)
        self.doc_freqs = []
        self.doc_lengths = []
        self.tokenized_corpus = []
        for document in corpus:
            tokens = simple_tokenize(document.text)
            token_counts = Counter(tokens)
            self.tokenized_corpus.append(tokens)
            self.doc_freqs.append(token_counts)
            self.doc_lengths.append(len(tokens))
            for token in token_counts.keys():
                df_counts[token] += 1
        total_docs = max(len(corpus), 1)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.doc_lengths else 0.0
        self.idf = {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df_counts.items()
        }

    def score_query(self, question: str) -> np.ndarray:
        query_tokens = simple_tokenize(question)
        scores = np.zeros(len(self.corpus), dtype=float)
        if not query_tokens:
            return scores
        for index, token_counts in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[index]
            denom_len = self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-8))
            score = 0.0
            for token in query_tokens:
                if token not in token_counts:
                    continue
                tf = token_counts[token]
                idf = self.idf.get(token, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + denom_len
                score += idf * (numerator / max(denominator, 1e-8))
            scores[index] = score
        return scores

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        scores = self.score_query(question)
        ranked_idx = np.argsort(-scores)[:top_k]
        hits: list[SearchHit] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            document = self.corpus[int(idx)]
            hits.append(
                SearchHit(
                    doc_id=document.doc_id,
                    score=float(scores[int(idx)]),
                    rank=rank,
                    title=document.title,
                    text=document.text,
                )
            )
        return hits


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] | list[int] = (1, 2), min_df: int = 1):
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
        self.doc_matrix = None
        self.corpus: list[Document] = []

    def fit(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        texts = [f"{doc.title}. {doc.text}" for doc in corpus]
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def score_query(self, question: str) -> np.ndarray:
        if self.doc_matrix is None:
            raise RuntimeError("TfidfRetriever must be fit before search.")
        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        return np.asarray(scores, dtype=float)

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        scores = self.score_query(question)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [
            SearchHit(
                doc_id=self.corpus[int(idx)].doc_id,
                score=float(scores[int(idx)]),
                rank=rank,
                title=self.corpus[int(idx)].title,
                text=self.corpus[int(idx)].text,
            )
            for rank, idx in enumerate(ranked_idx, start=1)
        ]


class SentenceTransformerRetriever(BaseRetriever):
    name = "dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self._model = None
        self.doc_embeddings: np.ndarray | None = None
        self.corpus: list[Document] = []

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for dense retrieval. Install requirements.txt."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def fit(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        texts = [f"{doc.title}. {doc.text}" for doc in corpus]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        self.doc_embeddings = np.asarray(embeddings, dtype=np.float32)

    def score_query(self, question: str) -> np.ndarray:
        if self.doc_embeddings is None:
            raise RuntimeError("SentenceTransformerRetriever must be fit before search.")
        query_embedding = self.model.encode(
            [question],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        scores = np.matmul(self.doc_embeddings, query_embedding)
        return np.asarray(scores, dtype=float)

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        scores = self.score_query(question)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [
            SearchHit(
                doc_id=self.corpus[int(idx)].doc_id,
                score=float(scores[int(idx)]),
                rank=rank,
                title=self.corpus[int(idx)].title,
                text=self.corpus[int(idx)].text,
            )
            for rank, idx in enumerate(ranked_idx, start=1)
        ]


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, retrievers: Iterable[BaseRetriever], weights: Iterable[float] | None = None):
        self.retrievers = list(retrievers)
        if not self.retrievers:
            raise ValueError("HybridRetriever requires at least one component retriever.")
        weights_list = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        if len(weights_list) != len(self.retrievers):
            raise ValueError("weights must align with retrievers length")
        self.weights = weights_list
        self.corpus: list[Document] = []

    def fit(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        for retriever in self.retrievers:
            retriever.fit(corpus)

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        vmin = values.min()
        vmax = values.max()
        if float(vmax - vmin) < 1e-12:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    def score_query(self, question: str) -> np.ndarray:
        combined = np.zeros(len(self.corpus), dtype=float)
        for retriever, weight in zip(self.retrievers, self.weights):
            scores = retriever.score_query(question)
            combined += weight * self._minmax(scores)
        return combined

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        scores = self.score_query(question)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [
            SearchHit(
                doc_id=self.corpus[int(idx)].doc_id,
                score=float(scores[int(idx)]),
                rank=rank,
                title=self.corpus[int(idx)].title,
                text=self.corpus[int(idx)].text,
            )
            for rank, idx in enumerate(ranked_idx, start=1)
        ]
