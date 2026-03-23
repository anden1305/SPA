#!/usr/bin/env python3
"""Compress JSON files to .json.xz for space savings.

Examples:
  uv run python scripts/compress_json_xz.py --root results
  uv run python scripts/compress_json_xz.py --root results --delete-original
  uv run python scripts/compress_json_xz.py --root . --dry-run
"""

from __future__ import annotations

import argparse
import lzma
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress .json files to .json.xz")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results"),
        help="Directory to scan recursively (default: results)",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete the original .json file after successful compression",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be compressed without writing anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .json.xz files",
    )
    return parser.parse_args()


def compress_file(json_path: Path, delete_original: bool, dry_run: bool, force: bool) -> tuple[bool, int, int]:
    xz_path = json_path.with_name(f"{json_path.name}.xz")

    if xz_path.exists() and not force:
        return False, 0, 0

    original_size = json_path.stat().st_size
    if dry_run:
        return True, original_size, 0

    data = json_path.read_bytes()
    compressed = lzma.compress(data, preset=6)
    xz_path.write_bytes(compressed)

    if delete_original:
        json_path.unlink()

    return True, original_size, len(compressed)


def main() -> None:
    args = parse_args()
    root = args.root

    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    json_files = sorted(p for p in root.rglob("*.json") if p.is_file())
    if not json_files:
        print("No .json files found.")
        return

    processed = 0
    skipped = 0
    original_total = 0
    compressed_total = 0

    for path in json_files:
        did_process, original_size, compressed_size = compress_file(
            json_path=path,
            delete_original=args.delete_original,
            dry_run=args.dry_run,
            force=args.force,
        )

        if did_process:
            processed += 1
            original_total += original_size
            compressed_total += compressed_size
            if args.dry_run:
                print(f"[DRY-RUN] {path} -> {path}.xz")
            else:
                print(f"[OK] {path} -> {path}.xz")
        else:
            skipped += 1
            print(f"[SKIP] {path} (compressed file exists; use --force to overwrite)")

    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    if not args.dry_run and processed > 0:
        saved = original_total - compressed_total
        ratio = (compressed_total / original_total) if original_total else 0.0
        print(f"Original total bytes:   {original_total}")
        print(f"Compressed total bytes: {compressed_total}")
        print(f"Saved bytes:            {saved}")
        print(f"Compression ratio:      {ratio:.3f}")


if __name__ == "__main__":
    main()
