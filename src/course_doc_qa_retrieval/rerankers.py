from __future__ import annotations

from dataclasses import dataclass

from .retrievers import SearchHit


@dataclass(slots=True)
class RerankedHit(SearchHit):
    original_rank: int = 0
    rerank_score: float = 0.0


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. Install requirements.txt."
                ) from exc
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int | None = None) -> list[RerankedHit]:
        if not hits:
            return []
        pairs = [(query, f"{hit.title}. {hit.text}") for hit in hits]
        scores = self.model.predict(pairs)
        candidates = [
            RerankedHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                title=hit.title,
                text=hit.text,
                original_rank=hit.rank,
                rerank_score=float(score),
            )
            for hit, score in zip(hits, scores)
        ]
        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        for new_rank, hit in enumerate(candidates, start=1):
            hit.rank = new_rank
        if top_k is None:
            return candidates
        return candidates[:top_k]
