from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import QueryExample
from .retrievers import SearchHit



def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances * discounts))



def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    best = sorted(relevances, reverse=True)
    ideal = dcg_at_k(best, k)
    if ideal <= 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal



def average_precision_at_k(relevances: Sequence[int], k: int) -> float:
    hits = 0
    score = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel:
            hits += 1
            score += hits / idx
    if hits == 0:
        return 0.0
    return score / hits



def reciprocal_rank_at_k(relevances: Sequence[int], k: int) -> float:
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel:
            return 1.0 / idx
    return 0.0



def recall_at_k(relevances: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant



def precision_at_k(relevances: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(relevances[:k]) / k



def hit_rate_at_k(relevances: Sequence[int], k: int) -> float:
    return 1.0 if any(relevances[:k]) else 0.0



def evaluate_rankings(
    queries: list[QueryExample],
    rankings: dict[str, list[SearchHit]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> tuple[dict[str, float], pd.DataFrame]:
    per_query_rows: list[dict] = []
    for query in queries:
        hits = rankings.get(query.query_id, [])
        relevant = set(query.relevant_doc_ids)
        relevances = [1 if hit.doc_id in relevant else 0 for hit in hits]
        row = {
            "query_id": query.query_id,
            "question": query.question,
            "num_relevant": len(relevant),
            "first_hit_rank": next((idx for idx, value in enumerate(relevances, start=1) if value), math.nan),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(relevances, len(relevant), k)
            row[f"precision@{k}"] = precision_at_k(relevances, k)
            row[f"hit@{k}"] = hit_rate_at_k(relevances, k)
        max_k = max(ks)
        row[f"mrr@{max_k}"] = reciprocal_rank_at_k(relevances, max_k)
        row[f"map@{max_k}"] = average_precision_at_k(relevances, max_k)
        row[f"ndcg@{max_k}"] = ndcg_at_k(relevances, max_k)
        per_query_rows.append(row)
    per_query_df = pd.DataFrame(per_query_rows)
    summary: dict[str, float] = {}
    for column in per_query_df.columns:
        if column in {"query_id", "question"}:
            continue
        summary[column] = float(per_query_df[column].mean(skipna=True))
    return summary, per_query_df
