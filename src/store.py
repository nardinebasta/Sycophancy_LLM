"""Append-only JSONL store with resume-by-id.

Never hold results in memory: a crash at hour six of a run must cost nothing
beyond the conversation in flight.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Iterator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCKS: dict[str, threading.Lock] = {}
_GLOBAL = threading.Lock()


def conversation_id(**parts: Any) -> str:
    """Deterministic id: re-running the same cell skips work already done."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _lock_for(path: str) -> threading.Lock:
    with _GLOBAL:
        if path not in _LOCKS:
            _LOCKS[path] = threading.Lock()
        return _LOCKS[path]


def append(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def read(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def done_ids(path: str) -> set[str]:
    return {r["conversation_id"] for r in read(path) if "conversation_id" in r}


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
