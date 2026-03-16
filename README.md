# Retrieval and Reranking for Course-Document QA

Research-style benchmark for course-document question answering retrieval.

This project is designed as an **empirical study** rather than a product app. It compares classical lexical retrieval, lightweight semantic retrieval, dense retrieval, and cross-encoder reranking on question-to-passage matching for lecture notes, course handouts, and study materials.

## What is included

- **BM25** lexical retriever implemented in pure Python
- **TF-IDF** baseline using scikit-learn
- **Dense retrieval** using Sentence-Transformers
- **Hybrid retrieval** that combines lexical and dense scores
- **Cross-encoder reranking** for top retrieved passages
- Evaluation with **Recall@K**, **Precision@K**, **Hit@K**, **MRR@K**, **MAP@K**, and **NDCG@K**
- Export of `metrics.json`, `rankings.csv`, `per_query_metrics.csv`, `query_report.csv`, and a histogram plot
- Sample course corpus and sample query set
- Benchmark script that aggregates results into **CSV** and **LaTeX** table format
- Utility script that chunks `.txt` and `.md` course materials into a JSONL corpus

## Repository structure

```text
course_doc_qa_retrieval_benchmark_full_source/
├── configs/
│   ├── benchmark_sample.yaml
│   ├── bm25_baseline.yaml
│   ├── tfidf_baseline.yaml
│   ├── dense_minilm.yaml
│   ├── hybrid_lexical.yaml
│   ├── hybrid_rerank_minilm.yaml
│   └── benchmark_lite.yaml
├── data/
│   ├── sample_course_docs.jsonl
│   └── sample_queries.jsonl
├── scripts/
│   ├── build_corpus_from_folder.py
│   ├── run_benchmark.py
│   ├── run_experiment.py
│   └── smoke_test.py
├── src/course_doc_qa_retrieval/
│   ├── data.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── rerankers.py
│   ├── retrievers.py
│   └── utils.py
├── pyproject.toml
├── requirements-lite.txt
└── requirements.txt
```

## Installation

### Lightweight mode (BM25 + TF-IDF only)

```bash
pip install -r requirements-lite.txt
pip install -e .
```

### Full mode (dense retrieval + reranking)

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

### 1) Run a lexical baseline

```bash
python scripts/run_experiment.py --config configs/bm25_baseline.yaml
```

### 2) Run a TF-IDF baseline

```bash
python scripts/run_experiment.py --config configs/tfidf_baseline.yaml
```

### 3) Run dense retrieval

```bash
python scripts/run_experiment.py --config configs/dense_minilm.yaml
```

### 4) Run hybrid retrieval + reranking

```bash
python scripts/run_experiment.py --config configs/hybrid_rerank_minilm.yaml
```

### 5) Run the benchmark suite

```bash
python scripts/run_benchmark.py --config configs/benchmark_sample.yaml
```

### 6) Run the lightweight benchmark (no dense models required)

```bash
python scripts/run_benchmark.py --config configs/benchmark_lite.yaml
```

### 7) Smoke test

```bash
python scripts/smoke_test.py
```

## Input data format

### Corpus JSONL
Each line is one passage or chunk.

```json
{"doc_id":"doc_001","title":"Lecture 3","text":"...","course":"Machine Learning","section":"Lecture 3","source_path":"ml/lecture3.md"}
```

### Query JSONL
Each line is one evaluation question.

```json
{"query_id":"q1","question":"What is overfitting?","relevant_doc_ids":["doc_023"]}
```

## Build your own corpus from text or markdown notes

```bash
python scripts/build_corpus_from_folder.py \
  --input-dir ./my_course_notes \
  --output-path ./data/my_course_corpus.jsonl \
  --max-words 180 \
  --stride-words 140
```

This script recursively reads `.txt` and `.md` files, chunks them into overlapping passages, and writes a JSONL corpus suitable for retrieval experiments.

## Output files

Each experiment writes to its configured `output_dir`.

- `metrics.json`: aggregate retrieval metrics
- `per_query_metrics.csv`: per-query Recall/Precision/Hit/MRR/MAP/NDCG
- `rankings.csv`: ranked documents for each query
- `query_report.csv`: compact analysis table per query
- `first_hit_histogram.png`: diagnostic figure showing the rank of the first relevant hit

The benchmark script additionally writes:

- `benchmark_results.csv`
- `benchmark_summary.csv`
- `benchmark_table.tex`
