"""Cross-lingual sentence embeddings with a persistent cache.

Backends (all local, nothing is downloaded automatically):
  labse                      sentence-transformers/LaBSE (CPU, made for bitext matching)
  bge-m3                     BAAI/bge-m3 (CPU, slower, strong multilingual)
  ollama:<model>             e.g. ollama:qwen3-embedding:8b (GPU through Ollama)
  none                       disabled (alignment falls back to time + lexical)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_ST_NAMES = {"labse": "sentence-transformers/LaBSE", "bge-m3": "BAAI/bge-m3"}


class Embedder:
    def __init__(self, backend: str, cache_path: Optional[Path] = None, ollama_url: str = "http://localhost:11434") -> None:
        self.backend = backend
        self.ollama_url = ollama_url.rstrip("/")
        self._model = None
        self._db: Optional[sqlite3.Connection] = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(cache_path))
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    @property
    def enabled(self) -> bool:
        return self.backend not in ("", "none", None)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.backend}\x00{text}".encode("utf-8")).hexdigest()

    def _load_st(self):
        if self._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_ST_NAMES[self.backend], device="cpu")
        return self._model

    def _compute(self, texts: list[str]) -> np.ndarray:
        if self.backend in _ST_NAMES:
            model = self._load_st()
            vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        if self.backend.startswith("ollama:"):
            model = self.backend.split(":", 1)[1]
            out = []
            for i in range(0, len(texts), 64):
                body = json.dumps({"model": model, "input": texts[i : i + 64]}).encode("utf-8")
                req = urllib.request.Request(self.ollama_url + "/api/embed", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.loads(r.read().decode("utf-8"))
                out.extend(data["embeddings"])
            arr = np.asarray(out, dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            return arr
        raise ValueError(f"unknown embedding backend {self.backend}")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t.strip() else "。" for t in texts]
        result: list[Optional[np.ndarray]] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            if self._db is not None:
                row = self._db.execute("SELECT v FROM emb WHERE k=?", (self._key(t),)).fetchone()
                if row:
                    result[i] = np.frombuffer(row[0], dtype=np.float32)
                    continue
            missing.setdefault(t, []).append(i)
        if missing:
            uniq = list(missing)
            vecs = self._compute(uniq)
            for t, v in zip(uniq, vecs):
                for i in missing[t]:
                    result[i] = v
                if self._db is not None:
                    self._db.execute("INSERT OR REPLACE INTO emb VALUES (?,?)", (self._key(t), v.astype(np.float32).tobytes()))
            if self._db is not None:
                self._db.commit()
        return np.vstack(result) if result else np.zeros((0, 1), dtype=np.float32)


def resolve_backend(name: str) -> str:
    """'auto' → first available local backend."""
    if name != "auto":
        return name
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            tags = json.loads(r.read().decode("utf-8"))
        names = [m.get("name", "") for m in tags.get("models", [])]
        for want in ("qwen3-embedding:8b", "qwen3-embedding:4b"):
            if any(n.startswith(want) for n in names):
                return "ollama:" + want
    except Exception:
        pass
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if (hub / "models--sentence-transformers--LaBSE").exists():
        return "labse"
    return "none"
