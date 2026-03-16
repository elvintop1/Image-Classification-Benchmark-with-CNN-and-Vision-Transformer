from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .data import QueryExample, load_corpus, load_queries
from .metrics import evaluate_rankings
from .rerankers import CrossEncoderReranker
from .retrievers import BM25Retriever, HybridRetriever, SearchHit, SentenceTransformerRetriever, TfidfRetriever
from .utils import ConfigError, dump_json, ensure_dir


@dataclass(slots=True)
class ExperimentArtifacts:
    metrics: dict[str, float]
    per_query: pd.DataFrame
    rankings: pd.DataFrame
    output_dir: Path



def build_retriever(config: dict[str, Any]):
    name = str(config.get("name", "bm25")).lower()
    params = config.get("params", {}) or {}
    if name == "bm25":
        return BM25Retriever(**params)
    if name == "tfidf":
        return TfidfRetriever(**params)
    if name == "dense":
        return SentenceTransformerRetriever(**params)
    if name == "hybrid":
        components = config.get("components", [])
        if not components:
            raise ConfigError("Hybrid retriever requires 'components'.")
        retrievers = [build_retriever(item) for item in components]
        weights = config.get("weights")
        return HybridRetriever(retrievers, weights=weights)
    raise ConfigError(f"Unsupported retriever: {name}")



def build_reranker(config: dict[str, Any] | None):
    if not config or not config.get("enabled", False):
        return None
    name = str(config.get("name", "cross_encoder")).lower()
    params = config.get("params", {}) or {}
    if name != "cross_encoder":
        raise ConfigError(f"Unsupported reranker: {name}")
    return CrossEncoderReranker(**params)



def rankings_to_frame(queries: list[QueryExample], rankings: dict[str, list[SearchHit]]) -> pd.DataFrame:
    rows: list[dict] = []
    for query in queries:
        relevant = set(query.relevant_doc_ids)
        hits = rankings.get(query.query_id, [])
        for hit in hits:
            rows.append(
                {
                    "query_id": query.query_id,
                    "question": query.question,
                    "doc_id": hit.doc_id,
                    "rank": hit.rank,
                    "score": hit.score,
                    "title": hit.title,
                    "is_relevant": int(hit.doc_id in relevant),
                    "text": hit.text,
                }
            )
    return pd.DataFrame(rows)



def plot_first_hit_histogram(per_query: pd.DataFrame, output_path: str | Path) -> None:
    series = per_query["first_hit_rank"].dropna()
    plt.figure(figsize=(8, 5))
    if not series.empty:
        plt.hist(series, bins=min(10, max(1, int(series.max()))))
    plt.xlabel("Rank of first relevant document")
    plt.ylabel("Number of queries")
    plt.title("First relevant hit distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()



def run_experiment(config: dict[str, Any]) -> ExperimentArtifacts:
    data_cfg = config["data"]
    output_dir = ensure_dir(config["output_dir"])
    corpus = load_corpus(data_cfg["corpus_path"])
    queries = load_queries(data_cfg["queries_path"])
    top_k = int(config.get("top_k", 10))
    rerank_top_k = int(config.get("rerank_top_k", top_k))
    metrics_ks = config.get("metrics_ks", [1, 3, 5, 10])

    retriever = build_retriever(config["retriever"])
    retriever.fit(corpus)

    reranker = build_reranker(config.get("reranker"))

    rankings: dict[str, list[SearchHit]] = {}
    for query in queries:
        hits = retriever.search(query.question, top_k=top_k)
        if reranker is not None:
            hits = reranker.rerank(query.question, hits, top_k=rerank_top_k)
            # Convert reranked hits back to SearchHit-compatible objects for downstream use.
            hits = [
                SearchHit(
                    doc_id=hit.doc_id,
                    score=hit.rerank_score,
                    rank=hit.rank,
                    title=hit.title,
                    text=hit.text,
                )
                for hit in hits
            ]
        rankings[query.query_id] = hits

    metrics, per_query_df = evaluate_rankings(queries, rankings, ks=metrics_ks)
    rankings_df = rankings_to_frame(queries, rankings)

    metrics_path = output_dir / "metrics.json"
    per_query_path = output_dir / "per_query_metrics.csv"
    rankings_path = output_dir / "rankings.csv"
    query_report_path = output_dir / "query_report.csv"
    figure_path = output_dir / "first_hit_histogram.png"

    dump_json(metrics, metrics_path)
    per_query_df.to_csv(per_query_path, index=False)
    rankings_df.to_csv(rankings_path, index=False)

    query_report = (
        rankings_df.groupby(["query_id", "question"], as_index=False)
        .agg(
            num_returned=("doc_id", "count"),
            num_relevant_hits=("is_relevant", "sum"),
            top1_doc=("doc_id", "first"),
            top1_title=("title", "first"),
        )
        .merge(per_query_df, on=["query_id", "question"], how="left")
    )
    query_report.to_csv(query_report_path, index=False)
    plot_first_hit_histogram(per_query_df, figure_path)

    return ExperimentArtifacts(metrics=metrics, per_query=per_query_df, rankings=rankings_df, output_dir=output_dir)
