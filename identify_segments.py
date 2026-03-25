import re
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import json
import argparse

def calculate_entropy(token_distribution):
    """计算token分布的熵"""
    probs = np.array(list(token_distribution.values()))
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy

def tokenize_with_html(text):
    """
    将文本分词，保持HTML标记完整
    
    例如：
    "Please <strong>click here</strong> to continue"
    -> ["Please", "<strong>", "click", "here", "</strong>", "to", "continue"]
    """
    # 匹配HTML标记或普通单词
    # HTML标记: <...> 或 </...>
    # 普通单词: 连续的非空白、非<>字符
    pattern = r'</?[^>]+>|\S+'
    tokens = re.findall(pattern, text)
    return tokens

def split_into_sentences(text):
    """
    将文本分割成句子
    支持常见的句子分隔符：. ! ? ; \n
    """
    # 使用正则表达式分割句子
    # 匹配句号、问号、感叹号、分号、换行符
    sentence_delimiters = r'[.!?;\n]+'
    sentences = re.split(sentence_delimiters, text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def extract_ngrams(text, n_range=(2, 6)):
    """
    提取n-gram segments（不跨句子边界）
    保持HTML标记完整
    
    Args:
        text: 输入文本
        n_range: n-gram范围 (min, max)
    
    Returns:
        list of n-grams
    """
    ngrams = []
    
    # 先分割成句子
    sentences = split_into_sentences(text)
    
    # 对每个句子分别提取n-grams
    for sentence in sentences:
        # 使用新的tokenize函数，保持HTML标记完整
        words = tokenize_with_html(sentence)
        
        # 只有当句子长度足够时才提取n-grams
        if len(words) < n_range[0]:
            continue
        
        for n in range(n_range[0], n_range[1] + 1):
            # 确保不会超出句子边界
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
    
    return ngrams

def extract_html_patterns(text, min_length=1, max_length=2):
    """
    专门提取HTML标记序列pattern
    
    例如: </div> </div> <div> <div>
    
    Args:
        text: 输入文本
        min_length: 最小标记数量
        max_length: 最大标记数量
    
    Returns:
        list of HTML tag sequences
    """
    # 提取所有HTML标记（保留顺序）
    html_tag_pattern = r'</?[^>]+>'
    
    # 找到所有HTML标记
    tags = re.findall(html_tag_pattern, text)
    
    if len(tags) < min_length:
        return []
    
    # 提取连续的HTML标记序列
    patterns = []
    for i in range(len(tags)):
        for length in range(min_length, min(max_length + 1, len(tags) - i + 1)):
            pattern = ' '.join(tags[i:i+length])
            patterns.append(pattern)
    
    return patterns

def analyze_segment_entropy(segment, contexts):
    """
    分析segment在不同上下文中后续token的熵
    contexts: list of texts containing this segment
    
    注意：只在句子内部分析后续token，不跨句子边界
    """
    next_tokens = []
    
    for context in contexts:
        # 先分割成句子
        sentences = split_into_sentences(context)
        
        for sentence in sentences:
            # 使用tokenize_with_html来处理
            words = tokenize_with_html(sentence)
            sentence_text = ' '.join(words)
            
            # 在句子中查找segment后的token
            # 转义特殊字符
            escaped_segment = re.escape(segment)
            # 匹配segment后面的任何token（包括HTML标记）
            pattern = escaped_segment + r'\s+(\S+)'
            matches = re.findall(pattern, sentence_text)
            next_tokens.extend(matches)
    
    if not next_tokens:
        return float('inf')  # 无法计算熵，返回无穷大
    
    # 统计next token的分布
    token_dist = Counter(next_tokens)
    entropy = calculate_entropy(token_dist)
    
    return entropy

def analyze_html_pattern_entropy(pattern, contexts):
    """
    专门分析HTML pattern的后续标记熵
    
    Args:
        pattern: HTML标记序列，如 "</div> </div> <div>"
        contexts: 包含该pattern的文本列表
    
    Returns:
        float: 熵值
    """
    next_tags = []
    
    for context in contexts:
        # 提取所有HTML标记
        html_tag_pattern = r'</?[^>]+>'
        tags = re.findall(html_tag_pattern, context)
        
        if len(tags) < 2:
            continue
        
        # 将tags组成字符串进行匹配
        tags_text = ' '.join(tags)
        
        # 查找pattern后的下一个标记
        escaped_pattern = re.escape(pattern)
        pattern_regex = escaped_pattern + r'\s+(</?[^>]+>)'
        matches = re.findall(pattern_regex, tags_text)
        next_tags.extend(matches)
    
    if not next_tags:
        return float('inf')
    
    # 统计分布
    tag_dist = Counter(next_tags)
    entropy = calculate_entropy(tag_dist)
    
    return entropy

def calculate_overlap_ratio(seg1, seg2):
    """
    计算两个segment的重叠比例
    返回较短segment被包含的比例
    """
    words1 = set(seg1.split())
    words2 = set(seg2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    min_len = min(len(words1), len(words2))
    
    return len(intersection) / min_len

def is_substring(short_seg, long_seg):
    """检查short_seg是否是long_seg的子串"""
    # 使用词边界匹配，避免部分词匹配
    pattern = r'\b' + re.escape(short_seg) + r'\b'
    return bool(re.search(pattern, long_seg))

def build_word_index(segments):
    """
    构建单词到segments的倒排索引
    用于快速查找包含特定单词的segments
    
    Returns:
        dict: {word: list of (segment_idx, segment_info)}
    """
    word_index = defaultdict(list)
    for idx, seg_info in enumerate(segments):
        words = seg_info['segment'].split()
        for word in words:
            word_index[word].append((idx, seg_info))
    return word_index

def filter_redundant_segments(segment_stats, overlap_threshold=0.7):
    """
    高效过滤冗余的segments
    
    优化策略：
    1. 使用倒排索引加速候选查找（从O(n²)降至O(n*k)，k为候选数）
    2. 预计算单词集合，避免重复split
    3. 提前终止不必要的比较
    
    Args:
        segment_stats: 已排序的segment列表（按score降序）
        overlap_threshold: 重叠度阈值（0-1）
    
    Returns:
        过滤后的segment列表
    """
    print(f"\n🔧 Filtering redundant segments (overlap threshold: {overlap_threshold})...")
    
    if not segment_stats:
        return []
    
    # 预计算所有segments的单词集合
    print(f"   Preprocessing {len(segment_stats)} segments...")
    segment_word_sets = [set(seg['segment'].split()) for seg in segment_stats]
    
    filtered = []
    filtered_word_sets = []
    
    removed_count = 0
    substring_removed = 0
    overlap_removed = 0
    replacement_count = 0
    
    print(f"   Filtering duplicates...")
    
    for i, current in enumerate(tqdm(segment_stats, desc="Processing")):
        current_seg = current['segment']
        current_words = segment_word_sets[i]
        is_redundant = False
        removal_reason = ""
        replace_indices = []
        
        # 只与已保留的segments比较
        for j, (kept, kept_words) in enumerate(zip(filtered, filtered_word_sets)):
            kept_seg = kept['segment']
            
            # 快速检查：如果没有共同词，跳过
            if not (current_words & kept_words):
                continue
            
            # 检查1: 当前segment是已保留segment的子串
            if is_substring(current_seg, kept_seg):
                is_redundant = True
                removal_reason = f"substring of '{kept_seg[:50]}...'"
                substring_removed += 1
                break
            
            # 检查2: 已保留segment是当前segment的子串
            # 标记要替换（当前更长，分数更高）
            if is_substring(kept_seg, current_seg):
                replace_indices.append(j)
                continue
            
            # 检查3: 计算重叠度（使用预计算的word sets）
            intersection = current_words & kept_words
            min_len = min(len(current_words), len(kept_words))
            overlap = len(intersection) / min_len
            
            if overlap >= overlap_threshold:
                is_redundant = True
                removal_reason = f"overlap {overlap:.2f} with '{kept_seg[:50]}...'"
                overlap_removed += 1
                break
        
        # 处理替换
        if replace_indices:
            # 从后往前删除，避免索引变化
            for j in sorted(replace_indices, reverse=True):
                del filtered[j]
                del filtered_word_sets[j]
            replacement_count += len(replace_indices)
            # 添加当前（更长的）segment
            filtered.append(current)
            filtered_word_sets.append(current_words)
        elif not is_redundant:
            # 添加新的segment
            filtered.append(current)
            filtered_word_sets.append(current_words)
        else:
            # 被过滤掉
            removed_count += 1
            if removed_count <= 5:
                print(f"   ❌ Removed: '{current_seg[:60]}...' - {removal_reason}")
    
    print(f"\n📊 Filtering results:")
    print(f"   Original segments: {len(segment_stats)}")
    print(f"   Kept segments: {len(filtered)}")
    print(f"   Removed total: {removed_count}")
    print(f"   - Substring matches: {substring_removed}")
    print(f"   - High overlap: {overlap_removed}")
    print(f"   - Replaced shorter segments: {replacement_count}")
    
    return filtered

def load_jsonl_dataset(jsonl_path, max_samples=None):
    """
    从JSONL文件加载数据集
    
    期望格式：
    {
        "system": "system prompt (optional)",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Failed to parse line {i+1}")
                continue
    
    return data

def extract_all_texts(dataset, include_system=True, include_conversations=True):
    """
    从数据集中提取所有文本
    
    Args:
        dataset: JSONL格式的数据集
        include_system: 是否包含system prompt
        include_conversations: 是否包含对话内容
    
    Returns:
        list of texts
    """
    all_texts = []
    
    for item in dataset:
        # 提取system prompt
        if include_system and "system" in item and item["system"]:
            all_texts.append(item["system"])
        
        # 提取conversations
        if include_conversations and "conversations" in item:
            for conv in item["conversations"]:
                if "value" in conv and conv["value"]:
                    all_texts.append(conv["value"])
                if "content" in conv and conv["content"]:
                    all_texts.append(conv["content"])
    
    return all_texts

def identify_all_segments(dataset, 
                          min_freq=10, 
                          max_entropy=2.0,
                          n_range=(2, 6),
                          max_segments=100,
                          overlap_threshold=0.7,
                          include_system=True,
                          include_conversations=False,
                          html_pattern_range=(3, 8),
                          analyze_html_patterns=True):
    """
    一次性识别所有高频低熵的segments（包括HTML patterns）
    
    Args:
        dataset: JSONL格式的数据集
        min_freq: 最小出现频率
        max_entropy: 最大熵阈值
        n_range: n-gram范围
        max_segments: 最大segment数量
        overlap_threshold: 重叠度阈值（用于过滤冗余segments）
        include_system: 是否分析system prompt
        include_conversations: 是否分析对话内容
        html_pattern_range: HTML pattern的长度范围（标记数量）
        analyze_html_patterns: 是否单独分析HTML patterns
    
    Returns:
        dict: {
            'text_segments': list of text segment statistics,
            'html_patterns': list of HTML pattern statistics
        }
    """
    print("📊 Step 1: Extracting texts from dataset...")
    all_texts = extract_all_texts(
        dataset, 
        include_system=include_system,
        include_conversations=include_conversations
    )
    print(f"📝 Extracted {len(all_texts)} text segments\n")
    
    # 统计各类文本数量
    system_count = sum(1 for item in dataset if item.get("system"))
    conv_count = sum(len(item.get("conversations", [])) for item in dataset)
    print(f"   System prompts: {system_count}")
    print(f"   Conversation turns: {conv_count}\n")
    
    # ============ 处理普通文本segments ============
    print("🔍 Step 2: Extracting n-grams and counting frequency...")
    segment_freq = Counter()
    segment_contexts = defaultdict(list)
    
    for text in tqdm(all_texts, desc="Processing texts"):
        ngrams = extract_ngrams(text, n_range)
        for ngram in ngrams:
            segment_freq[ngram] += 1
            # 保存包含该segment的上下文（用于计算熵）
            if segment_freq[ngram] <= 100:  # 限制保存的上下文数量
                segment_contexts[ngram].append(text)
    
    print(f"📊 Found {len(segment_freq)} unique segments\n")
    
    print("🧮 Step 3: Filtering by frequency...")
    high_freq_segments = {seg: freq for seg, freq in segment_freq.items() 
                          if freq >= min_freq}
    print(f"✅ {len(high_freq_segments)} segments with freq >= {min_freq}\n")
    
    print("🌡️ Step 4: Calculating entropy for high-frequency segments...")
    segment_stats = []
    
    for segment in tqdm(high_freq_segments.keys(), desc="Computing entropy"):
        freq = high_freq_segments[segment]
        entropy = analyze_segment_entropy(segment, segment_contexts[segment])
        
        if entropy <= max_entropy and 'id=' not in segment and '<think>' not in segment and '</think>' not in segment:
            segment_stats.append({
                'segment': segment,
                'frequency': freq,
                'entropy': entropy,
                'score': freq / (entropy + 1),  # 综合分数：高频低熵
                'num_words': len(segment.split()),
                'type': 'text'
            })
    
    print(f"✅ {len(segment_stats)} text segments with entropy <= {max_entropy}\n")
    
    # 按综合分数排序
    segment_stats.sort(key=lambda x: x['score'], reverse=True)
    
    # 过滤冗余的segments
    filtered_segments = filter_redundant_segments(segment_stats, overlap_threshold)
    
    # ============ 处理HTML patterns ============
    html_pattern_stats = []
    
    if analyze_html_patterns:
        print("\n🏷️  Step 5: Extracting HTML patterns...")
        html_pattern_freq = Counter()
        html_pattern_contexts = defaultdict(list)
        
        for text in tqdm(all_texts, desc="Extracting HTML"):
            patterns = extract_html_patterns(text, 
                                            min_length=html_pattern_range[0],
                                            max_length=html_pattern_range[1])
            for pattern in patterns:
                html_pattern_freq[pattern] += 1
                if html_pattern_freq[pattern] <= 100:
                    html_pattern_contexts[pattern].append(text)
        
        # print(html_pattern_freq.most_common(100))
        print(f"📊 Found {len(html_pattern_freq)} unique HTML patterns\n")
        
        print("🧮 Step 6: Filtering HTML patterns by frequency...")
        high_freq_html = {pat: freq for pat, freq in html_pattern_freq.items() 
                          if freq >= min_freq}
        print(f"✅ {len(high_freq_html)} HTML patterns with freq >= {min_freq}\n")
        
        print("🌡️ Step 7: Calculating entropy for HTML patterns...")
        for pattern in tqdm(high_freq_html.keys(), desc="Computing HTML entropy"):
            freq = high_freq_html[pattern]
            entropy = analyze_html_pattern_entropy(pattern, html_pattern_contexts[pattern])
            print(pattern, freq, entropy)
            if entropy <= max_entropy and 'id=' not in pattern and '<think>' not in pattern and '</think>' not in pattern:
                html_pattern_stats.append({
                    'segment': pattern,
                    'frequency': freq,
                    'entropy': entropy,
                    'score': freq / (entropy + 1),
                    'num_tags': len(pattern.split()),
                    'type': 'html'
                })
        
        print(f"✅ {len(html_pattern_stats)} HTML patterns with entropy <= {max_entropy}\n")
        
        # 按分数排序
        html_pattern_stats.sort(key=lambda x: x['score'], reverse=True)
    
    # 取前N个
    top_text_segments = filtered_segments[:max_segments]
    top_html_patterns = html_pattern_stats[:max_segments]
    
    print("\n🏆 Top text segments identified:")
    print(f"{'Segment':<50} {'Freq':<8} {'Entropy':<10} {'Score':<10} {'Words':<6}")
    print("-" * 90)
    for item in top_text_segments[:20]:
        print(f"{item['segment']:<50} {item['frequency']:<8} {item['entropy']:<10.2f} {item['score']:<10.2f} {item['num_words']:<6}")
    
    if len(top_text_segments) > 20:
        print(f"... and {len(top_text_segments) - 20} more text segments")
    
    if analyze_html_patterns and top_html_patterns:
        print("\n🏆 Top HTML patterns identified:")
        print(f"{'Pattern':<60} {'Freq':<8} {'Entropy':<10} {'Score':<10} {'Tags':<6}")
        print("-" * 100)
        for item in top_html_patterns[:20]:
            print(f"{item['segment']:<60} {item['frequency']:<8} {item['entropy']:<10.2f} {item['score']:<10.2f} {item['num_tags']:<6}")
        
        if len(top_html_patterns) > 20:
            print(f"... and {len(top_html_patterns) - 20} more HTML patterns")
    
    return {
        'text_segments': top_text_segments,
        'html_patterns': top_html_patterns
    }

def analyze_compression_potential(segments, dataset):
    """
    分析压缩潜力
    
    Args:
        segments: 识别出的segments字典 {'text_segments': [...], 'html_patterns': [...]}
        dataset: 数据集
    
    Returns:
        dict: 压缩统计信息
    """
    print("\n📈 Analyzing compression potential...")
    
    # 合并所有segments
    all_segments = segments['text_segments'] + segments['html_patterns']
    
    # 按长度降序排序segments，确保先替换长的
    sorted_segments = sorted(all_segments, key=lambda x: len(x['segment']), reverse=True)
    segment_to_token = {seg['segment']: f"<seg_{i}>" for i, seg in enumerate(sorted_segments)}
    
    total_original_chars = 0
    total_compressed_chars = 0
    total_original_words = 0
    total_compressed_words = 0
    
    for item in dataset:
        # 处理system prompt
        if item.get("system"):
            text = item["system"]
            total_original_chars += len(text)
            total_original_words += len(text.split())
            
            # 模拟压缩（按长度降序替换）
            compressed = text
            for segment, token in segment_to_token.items():
                compressed = compressed.replace(segment, token)
            
            total_compressed_chars += len(compressed)
            total_compressed_words += len(compressed.split())
    
    if total_original_chars > 0:
        char_compression = total_original_chars / total_compressed_chars
        word_compression = total_original_words / total_compressed_words
        char_savings = (1 - total_compressed_chars / total_original_chars) * 100
        word_savings = (1 - total_compressed_words / total_original_words) * 100
        
        print(f"   Original: {total_original_chars:,} chars, {total_original_words:,} words")
        print(f"   Compressed: {total_compressed_chars:,} chars, {total_compressed_words:,} words")
        print(f"   Compression ratio: {char_compression:.2f}x (chars), {word_compression:.2f}x (words)")
        print(f"   Space savings: {char_savings:.1f}% (chars), {word_savings:.1f}% (words)")
        
        return {
            'original_chars': total_original_chars,
            'compressed_chars': total_compressed_chars,
            'original_words': total_original_words,
            'compressed_words': total_compressed_words,
            'char_compression_ratio': char_compression,
            'word_compression_ratio': word_compression,
            'char_savings_pct': char_savings,
            'word_savings_pct': word_savings
        }
    
    return {}

def main():
    parser = argparse.ArgumentParser(description="Identify high-frequency low-entropy segments (including HTML patterns)")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Input JSONL file with conversations")
    parser.add_argument("--min_freq", type=int, default=10, 
                        help="Minimum frequency")
    parser.add_argument("--max_entropy", type=float, default=10.0, 
                        help="Maximum entropy")
    parser.add_argument("--max_segments", type=int, default=100, 
                        help="Maximum segments to keep (per type)")
    parser.add_argument("--overlap_threshold", type=float, default=0.7,
                        help="Overlap threshold for filtering redundant segments (0-1)")
    parser.add_argument("--n_min", type=int, default=4,
                        help="Minimum n-gram size")
    parser.add_argument("--n_max", type=int, default=6,
                        help="Maximum n-gram size")
    parser.add_argument("--html_min", type=int, default=3,
                        help="Minimum HTML pattern length (number of tags)")
    parser.add_argument("--html_max", type=int, default=8,
                        help="Maximum HTML pattern length (number of tags)")
    parser.add_argument("--output", type=str, default="segments.json", 
                        help="Output file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (for testing)")
    parser.add_argument("--include_conversations", action="store_true",
                        help="Also analyze conversation content (not just system prompts)")
    parser.add_argument("--no_html", action="store_true",
                        help="Disable HTML pattern analysis")
    args = parser.parse_args()
    
    print("="*90)
    print("🔬 Identifying High-Frequency Low-Entropy Segments (with HTML support)")
    print("="*90)
    print(f"Input: {args.input_jsonl}")
    print(f"Min Frequency: {args.min_freq}")
    print(f"Max Entropy: {args.max_entropy}")
    print(f"Max Segments: {args.max_segments} (per type)")
    print(f"Overlap Threshold: {args.overlap_threshold}")
    print(f"Text N-gram range: ({args.n_min}, {args.n_max})")
    print(f"HTML pattern range: ({args.html_min}, {args.html_max}) tags")
    print(f"Include conversations: {args.include_conversations}")
    print(f"Analyze HTML patterns: {not args.no_html}")
    print(f"⚠️  N-grams will NOT cross sentence boundaries\n")
    
    # 加载数据集
    print("📚 Loading dataset...")
    dataset = load_jsonl_dataset(args.input_jsonl, args.max_samples)
    print(f"✅ Loaded {len(dataset)} samples\n")
    
    # 识别segments
    segments = identify_all_segments(
        dataset,
        min_freq=args.min_freq,
        max_entropy=args.max_entropy,
        n_range=(args.n_min, args.n_max),
        max_segments=args.max_segments,
        overlap_threshold=args.overlap_threshold,
        include_system=True,
        include_conversations=args.include_conversations,
        html_pattern_range=(args.html_min, args.html_max),
        analyze_html_patterns=not args.no_html
    )
    
    # 分析压缩潜力
    compression_stats = analyze_compression_potential(segments, dataset)
    
    # 保存结果
    output_data = {
        # 'text_segments': segments['text_segments'],
        # 'html_patterns': segments['html_patterns'],
        'segments': segments['text_segments'] + segments['html_patterns'],
        'metadata': {
            'num_samples': len(dataset),
            'min_freq': args.min_freq,
            'max_entropy': args.max_entropy,
            'overlap_threshold': args.overlap_threshold,
            'n_range': [args.n_min, args.n_max],
            'html_pattern_range': [args.html_min, args.html_max],
            'include_conversations': args.include_conversations,
            'analyze_html_patterns': not args.no_html
        },
        'compression_stats': compression_stats
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    total_segments = len(segments['text_segments']) + len(segments['html_patterns'])
    print(f"\n💾 Saved {total_segments} segments to {args.output}")
    print(f"   - Text segments: {len(segments['text_segments'])}")
    print(f"   - HTML patterns: {len(segments['html_patterns'])}")
    print(f"\n✅ Segment identification complete!")

if __name__ == "__main__":
    main()