from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    course: str | None = None
    section: str | None = None
    source_path: str | None = None


@dataclass(slots=True)
class QueryExample:
    query_id: str
    question: str
    relevant_doc_ids: list[str]
    course: str | None = None



def simple_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())



def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object per line in {path}:{line_number}")
            rows.append(row)
    return rows



def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")



def load_corpus(path: str | Path) -> list[Document]:
    rows = read_jsonl(path)
    documents: list[Document] = []
    for row in rows:
        doc_id = str(row["doc_id"])
        title = str(row.get("title", doc_id))
        text = str(row["text"])
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                text=text,
                course=row.get("course"),
                section=row.get("section"),
                source_path=row.get("source_path"),
            )
        )
    return documents



def load_queries(path: str | Path) -> list[QueryExample]:
    rows = read_jsonl(path)
    queries: list[QueryExample] = []
    for row in rows:
        relevant = row.get("relevant_doc_ids", [])
        if isinstance(relevant, str):
            relevant = [relevant]
        queries.append(
            QueryExample(
                query_id=str(row["query_id"]),
                question=str(row["question"]),
                relevant_doc_ids=[str(doc_id) for doc_id in relevant],
                course=row.get("course"),
            )
        )
    return queries



def chunk_text(text: str, max_words: int = 180, stride_words: int = 140) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [text.strip()]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        stop = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:stop]).strip())
        if stop == len(words):
            break
        start += stride_words
    return chunks



def build_corpus_from_folder(
    input_dir: str | Path,
    output_path: str | Path,
    glob_pattern: str = "**/*",
    max_words: int = 180,
    stride_words: int = 140,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    records: list[dict] = []
    supported_suffixes = {".txt", ".md"}
    file_paths = [path for path in sorted(input_dir.glob(glob_pattern)) if path.is_file() and path.suffix.lower() in supported_suffixes]
    for file_index, file_path in enumerate(file_paths):
        raw_text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(raw_text, max_words=max_words, stride_words=stride_words)
        course = file_path.parent.name if file_path.parent != input_dir else None
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "doc_id": f"doc_{file_index:03d}_{chunk_index:03d}",
                    "title": file_path.stem,
                    "text": chunk,
                    "course": course,
                    "section": f"chunk_{chunk_index}",
                    "source_path": str(file_path.relative_to(input_dir)),
                }
            )
    write_jsonl(records, output_path)
    return pd.DataFrame.from_records(records)
