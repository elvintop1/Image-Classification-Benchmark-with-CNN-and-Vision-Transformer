#!/usr/bin/env python3
from __future__ import annotations

from course_doc_qa_retrieval.experiment import run_experiment



def main() -> None:
    config = {
        "data": {
            "corpus_path": "data/sample_course_docs.jsonl",
            "queries_path": "data/sample_queries.jsonl",
        },
        "retriever": {"name": "hybrid", "components": [{"name": "bm25"}, {"name": "tfidf"}], "weights": [0.6, 0.4]},
        "top_k": 5,
        "metrics_ks": [1, 3, 5],
        "output_dir": "outputs/smoke_test",
    }
    artifacts = run_experiment(config)
    assert artifacts.metrics["recall@5"] >= 0.70, artifacts.metrics
    assert artifacts.metrics["mrr@5"] >= 0.40, artifacts.metrics
    print("Smoke test passed.")
    print(artifacts.metrics)


if __name__ == "__main__":
    main()
