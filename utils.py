import re
from typing import List, Tuple, Dict, Any

SEG_RE = re.compile(r"<seg_\d+>")

def split_into_shared_chunks(compressed_text: str, min_len: int = 1) -> List[str]:
    """
    Split compressed_text by <seg_n> and return literal chunks that remain (shared content).
    """
    if not compressed_text:
        return []
    parts = SEG_RE.split(compressed_text)
    # keep chunks that have at least min_len non-space chars
    chunks = []
    for p in parts:
        if p and len(p.strip()) >= min_len:
            chunks.append(p)
    return chunks

def find_chunk_spans_in_order(rendered_text: str, chunks: List[str], start_cursor: int = 0) -> List[Tuple[int, int]]:
    """
    使用更宽松的匹配规则
    """
    spans = []
    cursor = start_cursor
    
    for ch in chunks:
        # 去除首尾空格
        ch_stripped = ch.strip()
        if not ch_stripped:
            continue
        
        # 方法1：直接查找去除空格的版本
        s = rendered_text.find(ch_stripped, cursor)
        
        if s == -1:
            # 方法2：尝试压缩所有空格后查找
            ch_compressed = ' '.join(ch_stripped.split())
            text_window = rendered_text[cursor:]
            
            # 在窗口中查找
            for i in range(len(text_window) - len(ch_compressed) + 1):
                window_compressed = ' '.join(text_window[i:i+len(ch_compressed)].split())
                if window_compressed == ch_compressed:
                    s = cursor + i
                    break
        
        if s == -1:
            print(f"   ⚠️ Chunk not found: {repr(ch_stripped[:50])}")
            continue
        
        e = s + len(ch_stripped)
        spans.append((s, e))
        cursor = e
    
    return spans

def mask_from_spans(offset_mapping: List[Tuple[int, int]], spans: List[Tuple[int, int]]) -> List[int]:
    """
    Token mask=1 if token overlaps any (char_start,char_end) span.
    """
    mask = [0] * len(offset_mapping)
    if not spans:
        return mask

    for i, (a, b) in enumerate(offset_mapping):
        # Many special tokens have (0,0) in fast tokenizers; ignore them.
        if a == 0 and b == 0:
            continue
        for (s, e) in spans:
            if b > s and a < e:  # overlap
                mask[i] = 1
                break
    return mask

def exclude_special_and_seg_ids(input_ids: List[int], tokenizer, mask: List[int], seg_token_ids: set) -> List[int]:
    """
    Remove all special tokens + <seg_n> tokens from mask.
    """
    special_ids = set(tokenizer.all_special_ids)
    out = mask[:]
    for i, tid in enumerate(input_ids):
        if out[i] == 0:
            continue
        if tid in seg_token_ids:
            out[i] = 0
    return out

def build_shared_content_kl_masks(
    teacher_tokenizer,
    student_tokenizer,
    teacher_text: str,
    student_text: str,
    original_messages: List[Dict[str, Any]],
    compressed_messages: List[Dict[str, Any]],
    teacher_enc: Dict[str, Any],
    student_enc: Dict[str, Any],
    roles_for_kl: set,
    seg_token_ids: set,
    min_chunk_len: int = 2,
) -> Tuple[List[int], List[int]]:
    """
    Return:
      teacher_kl_mask, student_kl_mask (list[int] aligned with their input_ids)
    
    逻辑：
    - 如果 assistant 内容没有 <seg_n>，标记整个内容
    - 如果 assistant 内容有 <seg_n>，只标记共享的非 seg 部分
    """
    # init empty masks
    teacher_mask = [0] * len(teacher_enc["input_ids"])
    student_mask = [0] * len(student_enc["input_ids"])

    t_cursor = 0
    s_cursor = 0

    
    for om, cm in zip(original_messages, compressed_messages):
        role = cm.get("role")
        if role not in roles_for_kl:
            continue

        content = cm.get("content", "")
        if not content:
            continue
        
        # 检查是否包含 seg token
        has_seg = bool(SEG_RE.search(content))
        if has_seg:
            # print(content)
            # assert False
            # 有 seg：只标记共享的字面 chunk
            chunks = split_into_shared_chunks(content, min_len=min_chunk_len)
            # print(chunks)
            if not chunks:
                continue
            
            t_spans = find_chunk_spans_in_order(teacher_text, chunks, start_cursor=t_cursor)
            s_spans = find_chunk_spans_in_order(student_text, chunks, start_cursor=s_cursor)
            # print(t_spans)
            # print([teacher_text[i:j] for i,j in t_spans])
            # print(teacher_enc["offset_mapping"][:100])
            # print(s_spans)
            # print([student_text[i:j] for i,j in s_spans])
            # print(student_enc["offset_mapping"][:100])
            if t_spans:
                t_cursor = t_spans[-1][1]
            if s_spans:
                s_cursor = s_spans[-1][1]
            
            t_part = mask_from_spans(teacher_enc["offset_mapping"], t_spans)
            s_part = mask_from_spans(student_enc["offset_mapping"], s_spans)
            # print(t_part)
            # print(s_part)
            
        else:
            # 没有 seg：标记整个 content
            # 把整个 content 当作一个完整的 chunk
            chunks = [content]
            
            t_spans = find_chunk_spans_in_order(teacher_text, chunks, start_cursor=t_cursor)
            s_spans = find_chunk_spans_in_order(student_text, chunks, start_cursor=s_cursor)
            
            if t_spans:
                t_cursor = t_spans[-1][1]
            if s_spans:
                s_cursor = s_spans[-1][1]
            
            t_part = mask_from_spans(teacher_enc["offset_mapping"], t_spans)
            s_part = mask_from_spans(student_enc["offset_mapping"], s_spans)

        # OR into global masks
        teacher_mask = [a or b for a, b in zip(teacher_mask, t_part)]
        student_mask = [a or b for a, b in zip(student_mask, s_part)]

    # exclude specials + seg tokens
    teacher_mask = exclude_special_and_seg_ids(teacher_enc["input_ids"], teacher_tokenizer, teacher_mask, seg_token_ids)
    student_mask = exclude_special_and_seg_ids(student_enc["input_ids"], student_tokenizer, student_mask, seg_token_ids)
    
    return teacher_mask, student_mask


def find_subsequence_ignore_prefix(haystack, needle, tokenizer, start=0):
    # Remove leading special tokens from needle
    special_ids = set(tokenizer.all_special_ids)
    # print(special_ids)

    k = 0
    while k < len(needle) and needle[k] in special_ids:
        k += 1
    core = needle[k:]
    if not core:
        return -1

    n, m = len(haystack), len(core)
    for i in range(start, n - m + 1):
        if haystack[i:i+m] == core:
            return i
    return -1


def build_ce_labels_mark_all_assistant(
    tokenizer,
    full_input_ids,
    messages,
    ignore_special_tokens=False,
):
    """
    Mark ALL assistant content tokens as labels (not -100).
    Works by tokenizing each assistant 'content' and locating it as a subsequence in full_input_ids.
    """
    labels = [-100] * len(full_input_ids)

    # print(full_input_ids)
    # print(messages)
    # print(tokenizer.decode(full_input_ids))
    cursor = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not content:
            continue

        # Tokenize only the assistant content (no chat template)
        content_ids = tokenizer(
            content,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if not content_ids:
            continue

        pos = find_subsequence_ignore_prefix(full_input_ids, content_ids, tokenizer, start=cursor)
        if pos == -1:
            # Fallback: try from beginning (sometimes template inserts stuff before)
            pos = find_subsequence_ignore_prefix(full_input_ids, content_ids, tokenizer, start=0)
        if pos == -1:
            # If still not found, skip this assistant segment (minimal behavior)
            continue
        
        # print(pos)
        # print('='*100)
        end = pos + len(content_ids)

        if ignore_special_tokens:
            special_ids = set(tokenizer.all_special_ids)
            for i in range(pos, end):
                if full_input_ids[i] not in special_ids:
                    labels[i] = full_input_ids[i]
        else:
            labels[pos:end] = full_input_ids[pos:end]

        cursor = end  # move cursor forward to keep order

    return labels

def build_ce_labels_robust(
    tokenizer,
    messages,
    max_length,
):
    """
    利用 chat template 的角色标记来定位 assistant 内容
    支持多种模型格式：Qwen, Llama, ChatML 等
    """
    full_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    labels = [-100] * len(input_ids)
    
    # 检测模型类型并使用对应的 pattern
    model_name = getattr(tokenizer, 'name_or_path', '').lower()
    
    # 定义不同模型的 assistant 内容 pattern
    patterns = []
    
    # Qwen / ChatML 格式: <|im_start|>assistant\n ... <|im_end|>
    if 'qwen' in model_name or hasattr(tokenizer, 'im_start_id'):
        patterns.append(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>')
    
    # Llama 3/3.1 格式: <|start_header_id|>assistant<|end_header_id|>\n\n ... <|eot_id|>
    if 'llama' in model_name or 'meta-llama' in model_name:
        patterns.append(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>')
    
    # Llama 2 格式: [/INST] ... </s>
    if 'llama-2' in model_name:
        patterns.append(r'\[/INST\]\s*(.*?)(?:</s>|$)')
    
    # Mistral 格式: [/INST] ... </s>
    if 'mistral' in model_name or 'mixtral' in model_name:
        patterns.append(r'\[/INST\]\s*(.*?)(?:</s>|$)')
    
    # 通用回退：尝试从 chat template 中推断
    if not patterns:
        # 检查 full_text 中的实际格式
        if '<|im_start|>assistant' in full_text:
            patterns.append(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>')
        elif '<|start_header_id|>assistant' in full_text:
            patterns.append(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>')
        elif '[/INST]' in full_text:
            patterns.append(r'\[/INST\]\s*(.*?)(?:</s>|$)')
        else:
            # 最后的回退：查找 "assistant" 后的内容
            patterns.append(r'assistant[:\n\s]+(.*?)(?:<\||</s>|\[/INST\]|$)')
    
    # 使用所有匹配的 pattern
    matched_any = False
    for pattern in patterns:
        for match in re.finditer(pattern, full_text, re.DOTALL):
            matched_any = True
            content_start = match.start(1)  # group 1 是内容部分
            content_end = match.end(1)
            
            # 标记这个范围内的所有 tokens
            for idx, (tok_start, tok_end) in enumerate(offsets):
                # 跳过特殊 token (offset 为 (0,0))
                if tok_start == 0 and tok_end == 0:
                    continue
                
                # 检查 token 是否在 assistant 内容范围内
                if tok_start >= content_start and tok_start < content_end:
                    labels[idx] = input_ids[idx]
    
    # 调试信息
    if not matched_any:
        print(f"⚠️ Warning: No assistant content matched!")
        print(f"   Model: {model_name}")
        print(f"   Patterns tried: {patterns}")
        print(f"   Full text preview: {full_text[:200]}...")
    
    return input_ids, labels

def print_kl_mask_tokens(dataset, tokenizer, num_samples=3):
    """
    打印 student_kl_mask 标记的 token 内容
    
    Args:
        dataset: 处理后的数据集
        tokenizer: student tokenizer
        num_samples: 打印多少个样本
    """
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        
        student_input_ids = sample["student_input_ids"]
        student_kl_mask = sample["student_kl_mask"]
        
        print(f"\n{'='*80}")
        print(f"Sample {idx}")
        print(f"{'='*80}")
        
        # 方法1: 打印所有被 mask 标记的 token
        masked_token_ids = [tid for tid, mask in zip(student_input_ids, student_kl_mask) if mask == 1]
        masked_tokens = [tokenizer.decode([tid]) for tid in masked_token_ids]
        
        print(f"\n标记的 token 数量: {len(masked_token_ids)} / {len(student_input_ids)}")
        print(f"标记比例: {len(masked_token_ids) / len(student_input_ids) * 100:.2f}%")
        
        print(f"\n标记的 token:")
        print(masked_tokens)
        
        # 方法2: 打印连续的被标记文本片段
        print(f"\n被标记的连续文本片段:")
        
        in_masked_region = False
        current_chunk = []
        chunk_count = 0
        
        for tid, mask in zip(student_input_ids, student_kl_mask):
            if mask == 1:
                if not in_masked_region:
                    in_masked_region = True
                    chunk_count += 1
                current_chunk.append(tid)
            else:
                if in_masked_region:
                    # 输出这个chunk
                    chunk_text = tokenizer.decode(current_chunk)
                    print(f"\nChunk {chunk_count}:")
                    print(f"  Token IDs: {current_chunk[:10]}{'...' if len(current_chunk) > 10 else ''}")
                    print(f"  Length: {len(current_chunk)} tokens")
                    print(f"  Text: {repr(chunk_text[:200])}")
                    
                    current_chunk = []
                    in_masked_region = False
        
        # 处理最后一个chunk
        if current_chunk:
            chunk_text = tokenizer.decode(current_chunk)
            print(f"\nChunk {chunk_count}:")
            print(f"  Length: {len(current_chunk)} tokens")
            print(f"  Text: {repr(chunk_text[:200])}")
        
        # 方法3: 可视化 mask（用颜色或符号标记）
        print(f"\n可视化 (M=masked, U=unmasked):")
        mask_pattern = ''.join(['M' if m == 1 else 'U' for m in student_kl_mask])
        # 每100个字符换行
        for i in range(0, len(mask_pattern), 100):
            print(mask_pattern[i:i+100])

        print(tokenizer.decode(student_input_ids))

