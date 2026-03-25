import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict


def normalize_text(text: str) -> str:
    if not text:
        return ""
    x = text.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    # Mask simple version-like strings to reduce false variants.
    x = re.sub(r"\bv?\d+\.\d+(?:\.\d+)?\b", "<VER>", x)
    return x.strip()


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def approx_token_len(text: str) -> int:
    return len(text.split())


def extract_fields(sample: dict) -> dict:
    # Keep aliases broad for OpenClaw variants.
    field_aliases = {
        "system_prompt": ["system_prompt", "system", "prompt_system"],
        "tool_usage_prompt": ["tool_usage_prompt", "tool_prompt", "prompt_tool"],
    }
    out = {
        "system_prompt": "",
        "tool_usage_prompt": "",
        "system_prompt_chars": sample.get("system_prompt_chars", None),
    }
    for logical, keys in field_aliases.items():
        for key in keys:
            val = sample.get(key)
            if isinstance(val, str) and val.strip():
                out[logical] = val
                break
    return out


def field_stats(samples: list, field: str) -> dict:
    raw_counts = Counter()
    norm_counts = Counter()
    norm_to_examples = defaultdict(list)
    present = 0

    for row in samples:
        text = row.get(field, "")
        if not text:
            continue
        present += 1
        raw_counts[text] += 1
        norm = normalize_text(text)
        norm_counts[norm] += 1
        if len(norm_to_examples[norm]) < 2:
            norm_to_examples[norm].append(text)

    if present == 0:
        return {
            "present": 0,
            "top_norm_coverage": 0.0,
            "top_raw_coverage": 0.0,
            "tier": "Tier-3",
            "variants": [],
        }

    top_raw, top_raw_n = raw_counts.most_common(1)[0]
    top_norm, top_norm_n = norm_counts.most_common(1)[0]
    raw_cov = top_raw_n / present
    norm_cov = top_norm_n / present

    if norm_cov >= 0.99:
        tier = "Tier-1"
    elif norm_cov >= 0.95:
        tier = "Tier-2"
    else:
        tier = "Tier-3"

    variants = []
    for text, n in norm_counts.most_common(10):
        variants.append(
            {
                "norm_hash": short_hash(text),
                "count": n,
                "coverage": round(n / present, 6),
                "approx_tokens": approx_token_len(text),
                "example": norm_to_examples[text][0][:400],
            }
        )

    return {
        "present": present,
        "top_norm_coverage": round(norm_cov, 6),
        "top_raw_coverage": round(raw_cov, 6),
        "top_raw_hash": short_hash(top_raw),
        "top_norm_hash": short_hash(top_norm),
        "top_norm_approx_tokens": approx_token_len(top_norm),
        "tier": tier,
        "variants": variants,
    }


def load_jsonl(path: str, limit: int) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out.append(extract_fields(item))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze repeated prompt spans in OpenClaw trajectories.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl, args.limit)
    missing_system_text = sum(1 for r in rows if not r.get("system_prompt"))
    has_only_char_count = sum(
        1
        for r in rows
        if (not r.get("system_prompt")) and (r.get("system_prompt_chars") is not None)
    )

    result = {
        "num_samples": len(rows),
        "data_health": {
            "missing_system_prompt_text": missing_system_text,
            "has_only_system_prompt_chars": has_only_char_count,
        },
        "system_prompt": field_stats(rows, "system_prompt"),
        "tool_usage_prompt": field_stats(rows, "tool_usage_prompt"),
        "selection_policy": {
            "tier_1": "coverage >= 0.99",
            "tier_2": "0.95 <= coverage < 0.99",
            "tier_3": "coverage < 0.95",
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Analyzed {len(rows)} samples.")
    print(f"Saved: {args.output_json}")
    if has_only_char_count > 0:
        print(
            "Warning: input appears to be evaluation logs with system_prompt_chars only. "
            "Prompt text is missing, so segment mining/replacement target discovery is not possible "
            "from this file alone."
        )


if __name__ == "__main__":
    main()
