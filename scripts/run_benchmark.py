#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from course_doc_qa_retrieval.experiment import run_experiment
from course_doc_qa_retrieval.utils import ensure_dir, load_yaml



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark of multiple retrieval systems.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark YAML config")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    benchmark_cfg = load_yaml(args.config)
    output_dir = ensure_dir(benchmark_cfg.get("output_dir", "outputs/benchmark"))
    config_dir = Path(args.config).resolve().parent

    rows: list[dict] = []
    for experiment_cfg in benchmark_cfg.get("experiments", []):
        name = experiment_cfg["name"]
        config_path = config_dir / experiment_cfg["config_path"]
        config = load_yaml(config_path)
        if "output_subdir" in experiment_cfg:
            config["output_dir"] = str(output_dir / experiment_cfg["output_subdir"])
        artifacts = run_experiment(config)
        row = {"experiment": name, **artifacts.metrics, "output_dir": str(artifacts.output_dir)}
        rows.append(row)
        print(f"Completed {name}: {artifacts.metrics}")

    results = pd.DataFrame(rows)
    results = results.sort_values(by=[col for col in results.columns if col.startswith("ndcg@")] or ["experiment"], ascending=False)
    results.to_csv(output_dir / "benchmark_results.csv", index=False)

    display_columns = [col for col in results.columns if col in {"experiment", "recall@1", "recall@3", "recall@5", "mrr@10", "map@10", "ndcg@10"}]
    summary = results[display_columns].copy() if display_columns else results.copy()
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    with (output_dir / "benchmark_table.tex").open("w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    print(f"Saved benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
