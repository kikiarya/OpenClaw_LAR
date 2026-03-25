import argparse
import json
import os
from typing import Any, Dict, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_mapping(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("token_mapping", {})


def replace_with_mapping(text: str, mapping: Dict[str, str]) -> str:
    if not text:
        return text
    out = text
    for src in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(src, mapping[src])
    return out


def choose_assistant_text(result_row: Dict[str, Any]) -> str:
    raw = result_row.get("_raw_response")
    pred = result_row.get("prediction", "")
    if isinstance(raw, str) and raw.strip():
        return raw
    return pred if isinstance(pred, str) else ""


def build_index(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        v = r.get(key)
        if isinstance(v, str) and v and v not in out:
            out[v] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge OpenClaw llm_input traces with TriviaQA results and build dual-view SFT JSONL."
    )
    parser.add_argument("--llm_input_jsonl", type=str, required=True, help="analysis/llm_inputs_flat.jsonl")
    parser.add_argument("--results_jsonl", type=str, required=True, help="triviaqa_5000.jsonl")
    parser.add_argument("--mapping_json", type=str, required=True, help="token mapping file")
    parser.add_argument("--output_dir", type=str, required=True, help="where train/validation JSONL are saved")
    parser.add_argument("--split_ratio", type=float, default=0.9)
    args = parser.parse_args()

    llm_rows = load_jsonl(args.llm_input_jsonl)
    result_rows = load_jsonl(args.results_jsonl)
    mapping = load_mapping(args.mapping_json)

    by_session = build_index(result_rows, "session_id")
    by_qid = build_index(result_rows, "question_id")

    merged: List[Dict[str, Any]] = []
    matched_by_session = 0
    matched_by_qid = 0
    dropped = 0

    for row in llm_rows:
        res: Optional[Dict[str, Any]] = None
        sid = row.get("session_id")
        qid = row.get("question_id")
        if isinstance(sid, str) and sid in by_session:
            res = by_session[sid]
            matched_by_session += 1
        elif isinstance(qid, str) and qid in by_qid:
            res = by_qid[qid]
            matched_by_qid += 1
        if res is None:
            dropped += 1
            continue

        system_text = row.get("system_prompt", "")
        question_text = res.get("question", "")
        assistant_text = choose_assistant_text(res)

        if not isinstance(system_text, str):
            system_text = ""
        if not isinstance(question_text, str):
            question_text = ""
        if not isinstance(assistant_text, str):
            assistant_text = ""
        if not system_text or not question_text or not assistant_text:
            dropped += 1
            continue

        compressed_system = replace_with_mapping(system_text, mapping)

        sample = {
            "id": res.get("question_id", row.get("run_id", "")),
            "original_messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "compressed_messages": [
                {"role": "system", "content": compressed_system},
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "meta": {
                "session_id": sid,
                "run_id": row.get("run_id"),
                "question_id": res.get("question_id"),
            },
        }
        merged.append(sample)

    split_idx = int(len(merged) * args.split_ratio)
    train_rows = merged[:split_idx]
    val_rows = merged[split_idx:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "validation.jsonl")
    ex_path = os.path.join(args.output_dir, "examples.json")
    stat_path = os.path.join(args.output_dir, "build_stats.json")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ex_path, "w", encoding="utf-8") as f:
        json.dump(train_rows[:3], f, ensure_ascii=False, indent=2)

    stats = {
        "llm_input_rows": len(llm_rows),
        "result_rows": len(result_rows),
        "merged_rows": len(merged),
        "matched_by_session": matched_by_session,
        "matched_by_question_id": matched_by_qid,
        "dropped_rows": dropped,
        "split_ratio": args.split_ratio,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "mapping_size": len(mapping),
    }
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved: {train_path}")
    print(f"saved: {val_path}")


if __name__ == "__main__":
    main()
