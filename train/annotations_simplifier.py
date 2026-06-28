"""Extract id/image/mask/point fields into annotations_processed.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ANNOTATION_FILES = ("annotations.jsonl", "annotations_processed.jsonl")


def extract_fields(input_path: Path, output_path: Path | None = None) -> int:
    results = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            results.append(
                {
                    "id": entry["id"],
                    "image": entry["image"],
                    "mask": entry["mask"],
                    "point": entry["point"],
                }
            )

    out = output_path or input_path.with_name("annotations_processed.jsonl")
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Extracted {len(results)} entries -> {out}")
    return len(results)


def _default_input(dataset_dir: Path) -> Path:
    for name in ANNOTATION_FILES:
        path = dataset_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No annotations file in {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create annotations_processed.jsonl from annotations.jsonl."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset directories or annotation files. Defaults to all outputs/* with annotations.jsonl.",
    )
    args = parser.parse_args()

    targets: list[Path] = []
    if args.datasets:
        for raw in args.datasets:
            path = Path(raw).expanduser().resolve()
            if path.is_dir():
                targets.append(_default_input(path))
            elif path.is_file():
                targets.append(path)
            else:
                raise FileNotFoundError(f"Not found: {path}")
    else:
        root = Path(__file__).resolve().parent.parent / "outputs"
        if not root.is_dir():
            raise SystemExit(f"No datasets given and {root} does not exist")
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "annotations.jsonl").is_file():
                targets.append(child / "annotations.jsonl")

    if not targets:
        raise SystemExit("No annotation files found")

    for path in targets:
        extract_fields(path)


if __name__ == "__main__":
    main()
