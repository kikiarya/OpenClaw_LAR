import json
import argparse
from tqdm import tqdm
import os
from typing import Dict, List, Any, Optional, Tuple


def load_token_mapping(mapping_file: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    与训练脚本一致的mapping读取方式:
    mapping_file 期望格式:
    {
      "special_tokens": ["<seg_0>", "<seg_1>", ...],
      "token_to_text": {
        "<seg_0>": "some long text segment ...",
        "<seg_1>": "..."
      }
    }

    Returns:
      special_tokens: List[str]
      token_to_text: Dict[token -> text]
      text_to_token: Dict[text -> token]  (由token_to_text反转得到，用于压缩)
    """
    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    special_tokens = data["special_tokens"]
    text_to_token = data.get("token_mapping", {})
    # print(token_to_text)

    # 反转得到 text -> token，用于压缩
    # 注意：如果存在重复text映射到多个token，这里后者会覆盖前者（一般不应发生）
    token_to_text = {v: k for k, v in text_to_token.items()}

    return special_tokens, token_to_text, text_to_token


def replace_text_with_tokens(text: str, text_to_token: Dict[str, str]) -> str:
    """将文本中的segments替换为特殊tokens（按segment长度降序，长的优先）"""
    if not text:
        return text

    # 按segment长度排序（长的优先）
    sorted_segments = sorted(text_to_token.keys(), key=len, reverse=True)

    # print(sorted_segments)

    replaced_text = text
    for segment in sorted_segments:
        if segment and segment in replaced_text:
            replaced_text = replaced_text.replace(segment + ' ', text_to_token[segment] + ' ')

    return replaced_text


def from_to_role(src: str) -> str:
    """
    将原始数据的 from 字段映射到 chat template 的 role:
    - human/user -> user
    - gpt/assistant -> assistant
    - system -> system
    """
    s = (src or "").strip().lower()
    if s in ["human", "user"]:
        return "user"
    if s in ["gpt", "assistant", "model"]:
        return "assistant"
    if s == "system":
        return "system"
    # 兜底：不认识就当 user（比 assistant 更安全）
    return "user"


def build_dual_messages(
    system_prompt: str,
    conversations: List[Dict[str, Any]],
    text_to_token: Dict[str, str],
    compress_system: bool = True,
    compress_conversations: bool = False,
) -> Dict[str, Any]:
    """
    构造训练脚本需要的 dual format:
      {
        "original_messages": [...],
        "compressed_messages": [...]
      }

    - original_messages: system使用原始system；对话使用原始value
    - compressed_messages: system/对话按开关选择压缩
    """
    system_prompt = system_prompt or ""

    compressed_system = (
        replace_text_with_tokens(system_prompt, text_to_token)
        if compress_system and system_prompt
        else system_prompt
    )

    original_messages = []
    compressed_messages = []

    # system 消息（训练脚本用 apply_chat_template，需要 system role 更稳）
    if system_prompt != "" or compressed_system != "":
        original_messages.append({"role": "system", "content": system_prompt})
        compressed_messages.append({"role": "system", "content": compressed_system})

    # 多轮对话
    for conv in conversations:
        role = from_to_role(conv.get("from", conv.get("role", "")))
        value = conv.get("value", conv.get("content", ""))

        original_messages.append({"role": role, "content": value})

        if compress_conversations and value:
            c_value = replace_text_with_tokens(value, text_to_token)
        else:
            c_value = value

        compressed_messages.append({"role": role, "content": c_value})

    return {
        "original_messages": original_messages,
        "compressed_messages": compressed_messages,
    }


def load_jsonl_dataset(jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """从JSONL文件加载数据集"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def compute_compression_stats(processed_data: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    统计字符压缩比：仅按 content 字符数粗略统计（足够做 sanity check）
    """
    orig_chars = 0
    comp_chars = 0

    for item in processed_data:
        orig_msgs = item.get("original_messages", [])
        comp_msgs = item.get("compressed_messages", [])
        for m in orig_msgs:
            orig_chars += len(m.get("content", "") or "")
        for m in comp_msgs:
            comp_chars += len(m.get("content", "") or "")

    return orig_chars, comp_chars


def build_sft_dataset_dual(
    input_jsonl: str,
    text_to_token: Dict[str, str],
    output_dir: str,
    compress_system: bool = True,
    compress_conversations: bool = False,
    max_samples: Optional[int] = None,
    split_ratio: float = 0.9,
):
    print("=" * 80)
    print("🏗️ Building Dual-Conversation SFT Dataset (original_messages + compressed_messages)")
    print("=" * 80)
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_dir}")
    print(f"Compress system: {compress_system}")
    print(f"Compress conversations: {compress_conversations}\n")

    # 加载原始数据
    print("📚 Loading dataset from JSONL...")
    raw_data = load_jsonl_dataset(input_jsonl, max_samples)
    print(f"✅ Loaded {len(raw_data)} samples\n")

    # 显示 system 压缩示例
    if raw_data and compress_system:
        sample_system = raw_data[0].get("system", "") or ""
        if sample_system:
            compressed_sample = replace_text_with_tokens(sample_system, text_to_token)
            print("🔄 System Prompt Compression Example:")
            print(f"Original length: {len(sample_system)} chars, {len(sample_system.split())} words")
            print(f"Compressed length: {len(compressed_sample)} chars, {len(compressed_sample.split())} words")
            ratio = len(sample_system) / max(len(compressed_sample), 1)
            print(f"Compression ratio: {ratio:.2f}x")
            print("\nOriginal (first 200 chars):")
            print(sample_system[:200] + "...")
            print("\nCompressed (first 200 chars):")
            print(compressed_sample[:200] + "...\n")

    # 处理所有数据
    print("🔄 Converting to dual format...")
    processed_data = []

    for item in tqdm(raw_data, desc="Processing"):
        system_prompt = item.get("system", "") or ""
        conversations = item.get("conversations", None)

        if conversations is None or not isinstance(conversations, list):
            raise ValueError("Each JSONL item must contain a list field 'conversations'.")

        dual = build_dual_messages(
            system_prompt=system_prompt,
            conversations=conversations,
            text_to_token=text_to_token,
            compress_system=compress_system,
            compress_conversations=compress_conversations,
        )

        # 保留原始ID（如果有）
        if "id" in item:
            dual["id"] = item["id"]

        processed_data.append(dual)

    # 划分训练/验证
    split_idx = int(len(processed_data) * split_ratio)
    train_data = processed_data[:split_idx]
    eval_data = processed_data[split_idx:]

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(processed_data)}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(eval_data)}")

    # 压缩统计
    orig_chars, comp_chars = compute_compression_stats(processed_data)
    if orig_chars > 0:
        compression_ratio = orig_chars / max(comp_chars, 1)
        savings_pct = (1 - comp_chars / orig_chars) * 100
        print(f"\n💾 Compression Statistics:")
        print(f"   Total original chars: {orig_chars:,}")
        print(f"   Total compressed chars: {comp_chars:,}")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
        print(f"   Space savings: {savings_pct:.1f}%")

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    train_jsonl = os.path.join(output_dir, "train.jsonl")
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    eval_jsonl = os.path.join(output_dir, "validation.jsonl")
    with open(eval_jsonl, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n💾 Saved datasets:")
    print(f"   Training: {train_jsonl}")
    print(f"   Validation: {eval_jsonl}")

    examples_file = os.path.join(output_dir, "examples.json")
    with open(examples_file, "w", encoding="utf-8") as f:
        json.dump(train_data[:3], f, indent=2, ensure_ascii=False)
    print(f"   Examples: {examples_file}")

    # 打印示例
    if train_data:
        print(f"\n📝 Example (first sample):")
        ex = train_data[0]
        print("Original messages:")
        for m in ex["original_messages"][:6]:
            print(f"  [{m['role']}]: {m['content'][:120]}{'...' if len(m['content'])>120 else ''}")
        print("Compressed messages:")
        for m in ex["compressed_messages"][:6]:
            print(f"  [{m['role']}]: {m['content'][:120]}{'...' if len(m['content'])>120 else ''}")

    print("\n✅ Dataset construction complete!")
    print(f"\n📝 Next steps:")
    print(f"   1. Put train.jsonl / validation.jsonl under: {output_dir}")
    print(f"   2. Your training script will auto-detect dual format and use it directly.")


def main():
    parser = argparse.ArgumentParser(description="Build dual SFT dataset for KL distillation training script")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL with {system, conversations}")
    parser.add_argument("--mapping", type=str, default="token_mapping.json", help="Token mapping file")
    parser.add_argument("--output_dir", type=str, default="sft_dataset", help="Output directory")
    parser.add_argument("--compress_system", action="store_true", default=True, help="Compress system prompt (default: True)")
    parser.add_argument("--no_compress_system", action="store_false", dest="compress_system", help="Don't compress system prompt")
    parser.add_argument("--compress_conversations", action="store_true", help="Also compress conversation content")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process (for testing)")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/validation split ratio (default: 0.9)")
    args = parser.parse_args()

    print("🔤 Loading token mapping (training-script compatible)...")
    special_tokens, token_to_text, text_to_token = load_token_mapping(args.mapping)
    print(f"✅ Loaded {len(special_tokens)} special tokens")
    print(f"✅ Loaded {len(token_to_text)} token_to_text entries")
    # 打印几个token示例
    if special_tokens:
        print(f"   Example tokens: {special_tokens[:5]}")
    print()

    build_sft_dataset_dual(
        input_jsonl=args.input_jsonl,
        text_to_token=text_to_token,
        output_dir=args.output_dir,
        compress_system=args.compress_system,
        compress_conversations=args.compress_conversations,
        max_samples=args.max_samples,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
