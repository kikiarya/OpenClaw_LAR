"""
Prepare JSONL for identify_segments.py from OpenClaw flat llm_input traces.

Input: output of extract_openclaw_tracer_llm_inputs.py (one row per llm_input).

Output: one JSONL line per sample:
  {"system": "<text for mining>", "question_id": ..., "session_id": ...}

By default, truncates system_prompt at the first occurrence of a marker so that
per-session injected workspace context does not dilute frequency statistics.
Markers tried in order (first match wins):
  "# Project Context"
  "## Workspace Files (injected)"

Usage:
  python scripts/prepare_openclaw_system_for_segment_mining.py \\
    --input_jsonl llm_inputs_flat.jsonl \\
    --output_jsonl dataset/triviaqa_openclaw/system_for_segments.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


DEFAULT_MARKERS = [
    "# Project Context",
    "## Workspace Files (injected)",
]


def strip_dynamic_suffix(text: str, markers: List[str]) -> str:
    if not text:
        return ""
    cut = len(text)
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build system-only JSONL for identify_segments.py from OpenClaw llm_input flat file."
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--no_truncate",
        action="store_true",
        help="Use full system_prompt without stripping after Project Context markers.",
    )
    parser.add_argument(
        "--marker",
        action="append",
        default=[],
        help="Additional substring marker to truncate before (repeatable). Applied after defaults.",
    )
    args = parser.parse_args()

    markers = list(DEFAULT_MARKERS)
    for m in args.marker:
        if m and m not in markers:
            markers.append(m)

    n_in = 0
    n_out = 0
    n_empty = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            sp = row.get("system_prompt", "")
            if not isinstance(sp, str):
                sp = ""
            if args.no_truncate:
                system = sp.strip()
            else:
                system = strip_dynamic_suffix(sp, markers).strip()
            if not system:
                n_empty += 1
                continue
            out = {
                "system": system,
                "conversations": [],
            }
            if row.get("question_id") is not None:
                out["question_id"] = row["question_id"]
            if row.get("session_id"):
                out["session_id"] = row["session_id"]
            if row.get("run_id"):
                out["run_id"] = row["run_id"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    summary = {
        "input_lines_parsed": n_in,
        "output_rows": n_out,
        "skipped_empty_system_after_truncate": n_empty,
        "output_jsonl": args.output_jsonl,
        "truncated": not args.no_truncate,
        "markers": markers if not args.no_truncate else [],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
