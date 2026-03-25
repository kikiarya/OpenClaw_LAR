"""
Extract OpenClaw tracer events into flat records for LAR preprocessing.

Typical tracer JSONL mixes multiple phases per run:
  phase=before_prompt_build | llm_input | llm_output | agent_end | ...

This script keeps only phase=="llm_input" rows and writes one JSON object per line with:
  - system_prompt, user_prompt (from "prompt" field), session_id, run_id, question_id, ts

Usage:
  python scripts/extract_openclaw_tracer_llm_inputs.py \\
    --input_jsonl path/to/tracer.jsonl \\
    --output_jsonl path/to/llm_inputs_flat.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional


def safe_load_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract llm_input rows from OpenClaw tracer JSONL.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--require_system_prompt",
        action="store_true",
        help="Skip rows without non-empty system_prompt.",
    )
    args = parser.parse_args()

    kept = 0
    skipped = 0
    bad_lines = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            obj = safe_load_line(raw)
            if obj is None:
                bad_lines += 1
                continue
            if obj.get("phase") != "llm_input":
                skipped += 1
                continue
            system_prompt = obj.get("system_prompt")
            if args.require_system_prompt and not (isinstance(system_prompt, str) and system_prompt.strip()):
                skipped += 1
                continue
            user_prompt = obj.get("prompt")
            if not isinstance(user_prompt, str):
                user_prompt = ""

            out = {
                "ts": obj.get("ts"),
                "phase": "llm_input",
                "session_id": obj.get("session_id"),
                "run_id": obj.get("run_id"),
                "question_id": obj.get("question_id"),
                "session_key": obj.get("session_key"),
                "agent_id": obj.get("agent_id"),
                "provider": obj.get("provider"),
                "model": obj.get("model"),
                "system_prompt": system_prompt if isinstance(system_prompt, str) else "",
                "user_prompt": user_prompt,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(
        json.dumps(
            {
                "kept_llm_input": kept,
                "skipped_non_llm_input_or_filtered": skipped,
                "bad_json_lines": bad_lines,
                "output": args.output_jsonl,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
