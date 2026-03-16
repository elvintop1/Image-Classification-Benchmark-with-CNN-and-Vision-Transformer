#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from course_doc_qa_retrieval.experiment import run_experiment
from course_doc_qa_retrieval.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one retrieval/reranking experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    artifacts = run_experiment(config)
    print(f"Saved outputs to: {artifacts.output_dir}")
    print("Key metrics:")
    for key, value in artifacts.metrics.items():
        if key.startswith(("recall@", "mrr@", "ndcg@", "map@")):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
