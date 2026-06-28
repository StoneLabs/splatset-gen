"""Fast random access into dataset annotations.jsonl without loading all records."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path


ANNOTATION_FILES = ("annotations.jsonl", "annotations_processed.jsonl")


def find_annotations_path(dataset_dir: Path) -> Path:
    for name in ANNOTATION_FILES:
        path = dataset_dir / name
        if path.is_file():
            return path
    names = ", ".join(ANNOTATION_FILES)
    raise FileNotFoundError(f"Missing {names} in {dataset_dir}")


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    byte_offset: int
    index: int


class DatasetIndex:
    """Index annotations.jsonl by byte offset for O(1) id lookup and sequential access."""

    CACHE_VERSION = 1

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.annotations_path = find_annotations_path(self.dataset_dir)
        self.config_path = self.dataset_dir / "config.yaml"
        self._refs: list[SampleRef] = []
        self._id_to_index: dict[str, int] = {}

    @property
    def count(self) -> int:
        return len(self._refs)

    def build(self, *, use_cache: bool = True) -> None:
        if not self.annotations_path.is_file():
            raise FileNotFoundError(f"Missing annotations: {self.annotations_path}")

        cache_path = self._cache_path()
        if use_cache and self._load_cache(cache_path):
            return

        refs: list[SampleRef] = []
        id_to_index: dict[str, int] = {}

        with self.annotations_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample_id = str(record["id"])
                idx = len(refs)
                refs.append(SampleRef(sample_id=sample_id, byte_offset=offset, index=idx))
                id_to_index[sample_id] = idx

        self._refs = refs
        self._id_to_index = id_to_index
        self._save_cache(cache_path)

    def sample_id_at(self, index: int) -> str:
        return self._refs[index].sample_id

    def resolve_index(self, sample_id: str) -> int:
        try:
            return self._id_to_index[sample_id]
        except KeyError as exc:
            raise KeyError(f"Unknown sample id: {sample_id}") from exc

    def read_record(self, index: int) -> dict:
        if index < 0 or index >= len(self._refs):
            raise IndexError(f"Sample index out of range: {index}")
        ref = self._refs[index]
        with self.annotations_path.open("rb") as handle:
            handle.seek(ref.byte_offset)
            line = handle.readline()
        return json.loads(line)

    def read_record_by_id(self, sample_id: str) -> dict:
        return self.read_record(self.resolve_index(sample_id))

    def page_ids(self, offset: int, limit: int) -> list[dict[str, str | int]]:
        offset = max(0, offset)
        limit = max(1, min(limit, 500))
        end = min(offset + limit, len(self._refs))
        return [
            {"index": ref.index, "id": ref.sample_id}
            for ref in self._refs[offset:end]
        ]

    def read_config_text(self) -> str:
        if not self.config_path.is_file():
            return "# config.yaml not found in dataset directory\n"
        return self.config_path.read_text(encoding="utf-8")

    def resolve_media_path(self, rel_path: str) -> Path:
        candidate = (self.dataset_dir / rel_path).resolve()
        if not str(candidate).startswith(str(self.dataset_dir)):
            raise ValueError("Path escapes dataset directory")
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing file: {rel_path}")
        return candidate

    def _cache_path(self) -> Path:
        stat = self.annotations_path.stat()
        tag = f"v{self.CACHE_VERSION}_{stat.st_size}_{int(stat.st_mtime)}"
        return self.dataset_dir / f".viewer_index_{tag}.pkl"

    def _load_cache(self, cache_path: Path) -> bool:
        if not cache_path.is_file():
            return False
        try:
            payload = pickle.loads(cache_path.read_bytes())
        except Exception:
            return False
        if payload.get("version") != self.CACHE_VERSION:
            return False
        refs = [
            SampleRef(sample_id=r["sample_id"], byte_offset=r["byte_offset"], index=r["index"])
            for r in payload["refs"]
        ]
        self._refs = refs
        self._id_to_index = {ref.sample_id: ref.index for ref in refs}
        return True

    def _save_cache(self, cache_path: Path) -> None:
        payload = {
            "version": self.CACHE_VERSION,
            "built_at": time.time(),
            "refs": [
                {
                    "sample_id": ref.sample_id,
                    "byte_offset": ref.byte_offset,
                    "index": ref.index,
                }
                for ref in self._refs
            ],
        }
        cache_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
