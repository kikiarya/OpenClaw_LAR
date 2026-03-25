import json
import argparse
from collections import OrderedDict

def load_segments(segments_file):
    """加载segments文件"""
    with open(segments_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 兼容两种格式
    if 'segments' in data:
        # 新格式：包含metadata
        return data['segments'], data.get('metadata', {}), data.get('compression_stats', {})
    else:
        # 旧格式：直接是segments列表
        return data, {}, {}

def create_token_mapping(segments_file, output_file, token_prefix="<seg_", token_suffix=">"):
    """
    为每个segment分配一个特殊token
    
    Args:
        segments_file: segments.json文件路径
        output_file: 输出的mapping文件路径
        token_prefix: 特殊token前缀
        token_suffix: 特殊token后缀
    
    Returns:
        dict: {segment: special_token}
    """
    # 加载segments
    segments, metadata, compression_stats = load_segments(segments_file)
    
    print(f"📚 Loaded {len(segments)} segments from {segments_file}")
    
    if metadata:
        print(f"\n📊 Metadata:")
        print(f"   Samples analyzed: {metadata.get('num_samples', 'N/A')}")
        print(f"   Min frequency: {metadata.get('min_freq', 'N/A')}")
        print(f"   Max entropy: {metadata.get('max_entropy', 'N/A')}")
        print(f"   N-gram range: {metadata.get('n_range', 'N/A')}")
    
    if compression_stats:
        print(f"\n💾 Expected Compression:")
        print(f"   Character ratio: {compression_stats.get('char_compression_ratio', 0):.2f}x")
        print(f"   Word ratio: {compression_stats.get('word_compression_ratio', 0):.2f}x")
        print(f"   Space savings: {compression_stats.get('char_savings_pct', 0):.1f}%")
    
    print()
    
    # 创建映射
    token_mapping = OrderedDict()
    special_tokens = []
    
    for idx, seg_info in enumerate(segments):
        segment = seg_info['segment']
        special_token = f"{token_prefix}{idx}{token_suffix}"
        
        token_mapping[segment] = special_token
        special_tokens.append(special_token)
    
    # 打印示例映射
    print("🗺️ Token Mapping Examples:")
    print(f"{'Segment':<60} {'Special Token':<15} {'Freq':<8} {'Words':<6}")
    print("-" * 95)
    for idx, (seg, token) in enumerate(list(token_mapping.items())[:15]):
        seg_info = segments[idx]
        freq = seg_info.get('frequency', 'N/A')
        num_words = seg_info.get('num_words', len(seg.split()))
        
        # 截断过长的segment
        display_seg = seg if len(seg) <= 60 else seg[:57] + "..."
        print(f"{display_seg:<60} {token:<15} {freq:<8} {num_words:<6}")
    
    if len(token_mapping) > 15:
        print(f"... and {len(token_mapping) - 15} more mappings")
    
    # 统计信息
    total_original_words = sum(len(seg.split()) for seg in token_mapping.keys())
    avg_words_per_segment = total_original_words / len(token_mapping)
    theoretical_compression = total_original_words / len(token_mapping)
    
    print(f"\n📊 Mapping Statistics:")
    print(f"   Total segments: {len(token_mapping)}")
    print(f"   Total special tokens: {len(special_tokens)}")
    print(f"   Original total words: {total_original_words}")
    print(f"   Avg words per segment: {avg_words_per_segment:.2f}")
    print(f"   Theoretical compression: {theoretical_compression:.2f}x")
    
    # 保存映射
    output_data = {
        'token_mapping': token_mapping,
        'special_tokens': special_tokens,
        'metadata': {
            'num_segments': len(segments),
            'total_original_words': total_original_words,
            'avg_words_per_segment': avg_words_per_segment,
            'theoretical_compression_ratio': theoretical_compression,
            'token_prefix': token_prefix,
            'token_suffix': token_suffix
        },
        'source_metadata': metadata,
        'compression_stats': compression_stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved token mapping to {output_file}")
    
    return token_mapping, special_tokens

def replace_text_with_tokens(text, token_mapping):
    """
    将文本中的segments替换为特殊tokens
    按照从长到短的顺序替换，避免部分匹配问题
    """
    # 按segment长度排序（长的优先），避免短segment先匹配导致长segment无法匹配
    sorted_segments = sorted(token_mapping.keys(), key=lambda x: len(x), reverse=True)
    
    replaced_text = text
    replacements = []
    
    for segment in sorted_segments:
        if segment in replaced_text:
            special_token = token_mapping[segment]
            count = replaced_text.count(segment)
            replaced_text = replaced_text.replace(segment, special_token)
            
            if count > 0:
                replacements.append({
                    'segment': segment,
                    'token': special_token,
                    'count': count
                })
    
    return replaced_text, replacements

def test_replacement_on_jsonl(token_mapping, jsonl_path, num_samples=3):
    """
    在实际JSONL数据上测试替换效果
    
    Args:
        token_mapping: token映射
        jsonl_path: JSONL文件路径
        num_samples: 测试样本数
    """
    print("\n🧪 Testing replacement on actual data...")
    
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    
    total_original_chars = 0
    total_compressed_chars = 0
    total_original_words = 0
    total_compressed_words = 0
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}:")
        print(f"{'='*80}")
        
        if 'system' in sample and sample['system']:
            original = sample['system']
            replaced, replacements = replace_text_with_tokens(original, token_mapping)
            
            total_original_chars += len(original)
            total_compressed_chars += len(replaced)
            total_original_words += len(original.split())
            total_compressed_words += len(replaced.split())
            
            print(f"\n📝 Original System Prompt ({len(original)} chars, {len(original.split())} words):")
            print(original[:200] + ("..." if len(original) > 200 else ""))
            
            print(f"\n✨ Compressed System Prompt ({len(replaced)} chars, {len(replaced.split())} words):")
            print(replaced)
            
            if replacements:
                print(f"\n🔄 Replacements made:")
                for rep in replacements[:5]:  # 只显示前5个
                    print(f"   '{rep['segment'][:50]}...' → {rep['token']} ({rep['count']}x)")
                if len(replacements) > 5:
                    print(f"   ... and {len(replacements) - 5} more replacements")
            
            compression_ratio = len(original) / max(len(replaced), 1)
            word_ratio = len(original.split()) / max(len(replaced.split()), 1)
            print(f"\n📉 Compression: {compression_ratio:.2f}x (chars), {word_ratio:.2f}x (words)")
    
    if total_original_chars > 0:
        print(f"\n{'='*80}")
        print(f"📊 Overall Test Statistics:")
        print(f"{'='*80}")
        print(f"   Total original: {total_original_chars} chars, {total_original_words} words")
        print(f"   Total compressed: {total_compressed_chars} chars, {total_compressed_words} words")
        print(f"   Overall compression: {total_original_chars/total_compressed_chars:.2f}x (chars)")
        print(f"   Overall compression: {total_original_words/total_compressed_words:.2f}x (words)")
        print(f"   Space savings: {(1-total_compressed_chars/total_original_chars)*100:.1f}% (chars)")

def main():
    parser = argparse.ArgumentParser(description="Create token mapping for segments")
    parser.add_argument("--segments", type=str, default="segments.json", 
                        help="Input segments file")
    parser.add_argument("--output", type=str, default="token_mapping.json", 
                        help="Output mapping file")
    parser.add_argument("--token_prefix", type=str, default="<seg_", 
                        help="Special token prefix")
    parser.add_argument("--token_suffix", type=str, default=">",
                        help="Special token suffix")
    parser.add_argument("--test_jsonl", type=str, default=None,
                        help="Test replacement on this JSONL file")
    parser.add_argument("--test_samples", type=int, default=3,
                        help="Number of samples to test")
    args = parser.parse_args()
    
    print("="*80)
    print("🗺️ Creating Token Mapping for Segments")
    print("="*80)
    print(f"Input: {args.segments}")
    print(f"Output: {args.output}")
    print(f"Token format: {args.token_prefix}N{args.token_suffix}\n")
    
    token_mapping, special_tokens = create_token_mapping(
        args.segments,
        args.output,
        args.token_prefix,
        args.token_suffix
    )
    
    # 测试替换（如果提供了测试文件）
    if args.test_jsonl:
        test_replacement_on_jsonl(token_mapping, args.test_jsonl, args.test_samples)
    
    print("\n✅ Token mapping creation complete!")
    print(f"\n📝 Next steps:")
    print(f"   1. Use this mapping in build_sft_dataset.py")
    print(f"   2. The {len(special_tokens)} special tokens will be added to tokenizer")
    print(f"   3. Train with train_lora_sft.py to learn token semantics")

if __name__ == "__main__":
    main()