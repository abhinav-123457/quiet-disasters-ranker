#!/usr/bin/env python3
"""
merge_chunks.py — Reconstruct candidates.jsonl from the split chunks.

GitHub caps single files at 100 MB, so the ~465 MB candidates.jsonl dataset
is committed as 10 line-aligned shards under data_chunks/. This script
concatenates them back into a single candidates.jsonl (or a subset).

Usage
-----
    # Merge ALL chunks -> full 100K dataset (candidates.jsonl)
    python merge_chunks.py

    # Merge specific parts for partial/quick testing (e.g. first 30K)
    python merge_chunks.py --parts 1 2 3 --out candidates_30k.jsonl
"""
import argparse
import glob
import os
import sys

CHUNK_DIR = "data_chunks"
PATTERN = os.path.join(CHUNK_DIR, "candidates_part_*.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Merge candidate chunks into one JSONL file.")
    parser.add_argument(
        "--parts", type=int, nargs="*", default=None,
        help="Specific part numbers to merge (e.g. --parts 1 2 3). Default: all parts.",
    )
    parser.add_argument(
        "--out", type=str, default="candidates.jsonl",
        help="Output file path (default: candidates.jsonl).",
    )
    args = parser.parse_args()

    all_parts = sorted(glob.glob(PATTERN))
    if not all_parts:
        sys.exit(f"ERROR: no chunks found matching '{PATTERN}'. "
                 f"Run this from the repo root where '{CHUNK_DIR}/' lives.")

    if args.parts:
        wanted = set(args.parts)
        selected = [p for p in all_parts
                    if int(os.path.basename(p).split("_")[-1].split(".")[0]) in wanted]
        if not selected:
            sys.exit(f"ERROR: none of the requested parts {sorted(wanted)} were found.")
    else:
        selected = all_parts

    lines = 0
    with open(args.out, "wb") as out:
        for p in selected:
            with open(p, "rb") as f:
                data = f.read()
                out.write(data)
                if not data.endswith(b"\n"):
                    out.write(b"\n")  # guard against a shard missing its trailing newline
                lines += data.count(b"\n")

    print(f"Merged {len(selected)} chunk(s) -> {args.out}  ({lines:,} candidate lines)")


if __name__ == "__main__":
    main()
