#!/usr/bin/env python3
from __future__ import annotations

import argparse

from course_doc_qa_retrieval.data import build_corpus_from_folder



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunked course corpus JSONL from .txt/.md files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing course notes")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save corpus JSONL")
    parser.add_argument("--glob-pattern", type=str, default="**/*", help="Glob for input files")
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--stride-words", type=int, default=140)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    df = build_corpus_from_folder(
        input_dir=args.input_dir,
        output_path=args.output_path,
        glob_pattern=args.glob_pattern,
        max_words=args.max_words,
        stride_words=args.stride_words,
    )
    print(f"Built {len(df)} chunks -> {args.output_path}")


if __name__ == "__main__":
    main()
