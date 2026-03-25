import argparse
import json
from typing import Dict, List, Tuple


def load_analysis(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_manual_candidates(path: str) -> List[Tuple[str, str]]:
    """
    JSON format:
    [
      {"name": "system_prompt", "text": "..."},
      {"name": "tool_usage_prompt", "text": "..."}
    ]
    """
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for item in raw:
        name = item.get("name", "")
        text = item.get("text", "")
        if name and text:
            out.append((name, text))
    return out


def select_candidates(analysis: dict, min_coverage: float) -> List[Tuple[str, str, float]]:
    selected = []
    for field in ("system_prompt", "tool_usage_prompt"):
        node = analysis.get(field, {})
        cov = float(node.get("top_norm_coverage", 0.0))
        if cov < min_coverage:
            continue
        variants = node.get("variants", [])
        if not variants:
            continue
        # The variant example is normalized text, suited for stable replacement.
        text = variants[0].get("example", "")
        if text:
            selected.append((field, text, cov))
    return selected


def build_mapping(entries: List[Tuple[str, str]], start_index: int = 0) -> Dict[str, str]:
    mapping = {}
    idx = start_index
    # Longer text first helps replacement robustness.
    for _, text in sorted(entries, key=lambda x: len(x[1]), reverse=True):
        if text in mapping:
            continue
        mapping[text] = f"<seg_{idx}>"
        idx += 1
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Build token_mapping.json from prompt-repeat analysis.")
    parser.add_argument("--analysis_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--min_coverage", type=float, default=0.95)
    parser.add_argument("--manual_candidates_json", type=str, default="")
    args = parser.parse_args()

    analysis = load_analysis(args.analysis_json)
    auto_selected = select_candidates(analysis, args.min_coverage)
    manual_selected = parse_manual_candidates(args.manual_candidates_json)

    entries = [(name, text) for name, text, _ in auto_selected]
    for name, text in manual_selected:
        entries.append((name, text))

    token_mapping = build_mapping(entries)
    special_tokens = list(token_mapping.values())
    result = {
        "special_tokens": special_tokens,
        "token_mapping": token_mapping,
        "metadata": {
            "min_coverage": args.min_coverage,
            "auto_selected_fields": [
                {"name": name, "coverage": cov} for name, _, cov in auto_selected
            ],
            "manual_count": len(manual_selected),
            "total_tokens": len(special_tokens),
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output_json}")
    print(f"special_tokens: {len(special_tokens)}")


if __name__ == "__main__":
    main()
