import argparse
import json
import os
from typing import Any, Dict, List


def load_mapping(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("token_mapping", {})


def replace_text(text: str, mapping: Dict[str, str]) -> str:
    if not text:
        return text
    out = text
    for src in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(src, mapping[src])
    return out


def to_role(val: str) -> str:
    v = (val or "").strip().lower()
    if v in ("human", "user"):
        return "user"
    if v in ("assistant", "gpt", "model"):
        return "assistant"
    if v == "system":
        return "system"
    return "user"


def parse_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(sample.get("messages"), list):
        msgs = []
        for m in sample["messages"]:
            role = to_role(m.get("role", m.get("from", "")))
            content = m.get("content", m.get("value", ""))
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
        return msgs

    if isinstance(sample.get("conversations"), list):
        msgs = []
        for m in sample["conversations"]:
            role = to_role(m.get("from", m.get("role", "")))
            content = m.get("value", m.get("content", ""))
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
        return msgs

    return []


def build_dual(sample: Dict[str, Any], mapping: Dict[str, str], compress_conversations: bool) -> Dict[str, Any]:
    system_text = sample.get("system_prompt", sample.get("system", "")) or ""
    messages = parse_messages(sample)

    original_messages = []
    compressed_messages = []

    if system_text:
        original_messages.append({"role": "system", "content": system_text})
        compressed_messages.append({"role": "system", "content": replace_text(system_text, mapping)})

    for m in messages:
        content = m["content"]
        original_messages.append(m)
        if compress_conversations:
            compressed_messages.append({"role": m["role"], "content": replace_text(content, mapping)})
        else:
            compressed_messages.append({"role": m["role"], "content": content})

    out = {
        "original_messages": original_messages,
        "compressed_messages": compressed_messages,
    }
    if "id" in sample:
        out["id"] = sample["id"]
    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenClaw dual-view dataset for train.py")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--compress_conversations", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    mapping = load_mapping(args.mapping_json)

    dual_rows = [build_dual(r, mapping, args.compress_conversations) for r in rows]
    split_idx = int(len(dual_rows) * args.split_ratio)
    train_rows = dual_rows[:split_idx]
    val_rows = dual_rows[split_idx:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "validation.jsonl")
    ex_path = os.path.join(args.output_dir, "examples.json")

    dump_jsonl(train_path, train_rows)
    dump_jsonl(val_path, val_rows)
    with open(ex_path, "w", encoding="utf-8") as f:
        json.dump(train_rows[:3], f, ensure_ascii=False, indent=2)

    print(f"total={len(dual_rows)} train={len(train_rows)} val={len(val_rows)}")
    print(f"saved: {train_path}")
    print(f"saved: {val_path}")


if __name__ == "__main__":
    main()
