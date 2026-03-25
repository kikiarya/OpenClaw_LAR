import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
from typing import List, Dict, Tuple, Optional
import traceback
from datetime import datetime
import os
import re
import random
import numpy as np

# Import Mind2Web dataloader
from mind2web_utils import get_data_split, format_input_generation, format_input_multichoice

def set_seed(seed: int = 42):
    """
    固定所有随机性来源的种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量以确保完全的可复现性
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"✅ Random seed set to {seed}")

# System prompt for Mind2Web
SYSTEM_PROMPT = """You are a web navigation assistant. 
Your job is to help complete the user's task by interacting with the webpage step by step.

You are given:
- A natural language task
- The current webpage (Cleaned HTML DOM)

Think step by step about what to do next, then output the next action in the required format.

Valid actions:
- CLICK[id]
- TYPE[id](text)
- SELECT[id](option)
- HOVER[id]
- PRESS[key]

Rules:
- Think carefully about the best action to take to accomplish the task.
- Only output one action per step.
- Do not explain in natural language.
- Do not hallucinate elements that do not exist in the DOM.
- Choose the element that best matches the intent of the task.
- Output action exactly two to three line in this format:

Element: <ELEMENT_ID>
Action: <ACTION_TYPE>
Value: <VALUE> # If ACTION_TYPE is TYPE or SELECT, otherwise VALUE is null.

Where:
- ELEMENT_ID is integer or null
- ACTION_TYPE is in {CLICK, TYPE, SELECT, HOVER, PRESS}
- VALUE is a string or null

No other tokens are allowed before or after this line.

OUTPUT FORMAT:
Your response must strictly follow this sequence:
1. <think> [Your thought process] </think>
2. [Output action strictly follow the rules] """

# SYSTEM_PROMPT = """You are a web navigation assistant. 
# Your job is to help complete the user's task by interacting with the webpage step by step.

# You are given:
# - A natural language task
# - The current webpage (Cleaned HTML DOM)

# Valid actions:
# - CLICK[id]
# - TYPE[id](text)
# - SELECT[id](option)
# - HOVER[id]
# - PRESS[key]

# Rules:
# - Only output one action per step.
# - Do not explain in natural language.
# - Do not hallucinate elements that do not exist in the DOM.
# - Choose the element that best matches the intent of the task.
# - Output action exactly two to three line in this format:

# Element: <ELEMENT_ID>
# Action: <ACTION_TYPE>
# Value: <VALUE> # If ACTION_TYPE is TYPE or SELECT, otherwise VALUE is null.

# Where:
# - ELEMENT_ID is integer or null
# - ACTION_TYPE is in {CLICK, TYPE, SELECT, HOVER, PRESS}
# - VALUE is a string or null

# No other tokens are allowed before or after this line.

# OUTPUT FORMAT:
# Your response must strictly follow this sequence:
# 1. [Output action strictly follow the rules] """


def load_token_mapping(mapping_file):
    """加载token映射"""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['token_mapping'], data['special_tokens']


def replace_text_with_tokens(text, token_mapping, tokenizer=None, equal_length=False):
    """将文本中的segments替换为特殊tokens，可选补齐到等长token数。"""
    if not token_mapping:
        return text

    if equal_length and tokenizer is None:
        raise ValueError("equal_length=True requires tokenizer.")

    pad_token = None
    if equal_length:
        pad_token = tokenizer.pad_token
        if pad_token is None:
            raise ValueError("Tokenizer has no pad_token. Please set tokenizer.pad_token first.")

    original_total_tokens = None
    if equal_length:
        original_total_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    sorted_segments = sorted(token_mapping.keys(), key=lambda x: len(x), reverse=True)
    replaced_text = text
    for segment in sorted_segments:
        if segment not in replaced_text:
            continue

        special_token = token_mapping[segment]
        replacement = special_token
        if equal_length:
            original_len = len(tokenizer.encode(segment, add_special_tokens=False))
            replaced_len = len(tokenizer.encode(special_token, add_special_tokens=False))
            if original_len > replaced_len:
                pad_count = original_len - replaced_len
                replacement = special_token + "".join([pad_token] * pad_count)

        replaced_text = replaced_text.replace(' ' + segment + ' ', replacement + ' ').replace(segment + ' ', replacement + ' ')

    if equal_length:
        replaced_total_tokens = len(tokenizer.encode(replaced_text, add_special_tokens=False))
        if replaced_total_tokens < original_total_tokens:
            pad_count = original_total_tokens - replaced_total_tokens
            replaced_text = replaced_text + "".join([pad_token] * pad_count)

    return replaced_text


def build_attention_mask(input_ids: torch.LongTensor, pad_token_id: Optional[int], mask_padding_tokens: bool) -> torch.LongTensor:
    if mask_padding_tokens and pad_token_id is not None:
        return (input_ids != pad_token_id).long()
    return torch.ones_like(input_ids, dtype=torch.long)


def resolve_generation_eos_token_id(tokenizer):
    """Include eot token ids as eos when available in tokenizer."""
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    unk_id = tokenizer.unk_token_id
    for token in ("<|eot_id|>", "<eot_id>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            if unk_id is not None and token_id == unk_id:
                continue
            if token_id not in eos_ids:
                eos_ids.append(token_id)

    if not eos_ids:
        return None
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


def strip_eot_markers(text: str) -> str:
    if not text:
        return text
    for marker in ("<|eot_id|>", "<eot_id>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _normalize_action(s: Optional[str]) -> str:
    """规范化action字符串"""
    return (s or "").strip().upper()


def _normalize_value(v: Optional[str]) -> str:
    """规范化value字符串"""
    if v is None:
        return ""
    v = str(v).strip()
    if v.lower() in {"null", "none", ""}:
        return ""
    return v


def _extract_id_from_element_text(elem_text: str) -> Optional[str]:
    """
    从element文本中提取ID
    支持格式:
      - "<svg id=1 />"
      - "svg#1"
      - "Element: <div id=\"7\">"
      - "id=12" or "id='12'"
    """
    t = elem_text.strip()

    # svg#1 or div#12 etc.
    m = re.search(r"#\s*(\d+)\b", t)
    if m:
        return m.group(1)

    # id=1, id="1", id='1'
    m = re.search(r"\bid\s*=\s*['\"]?(\d+)['\"]?\b", t)
    if m:
        return m.group(1)

    return None


def _parse_model_answer(ans: str) -> Tuple[Optional[str], str, str, str]:
    """
    解析模型输出
    Expected format:
      <think>...</think>
      Element: svg#1
      Action: CLICK
      Value: null
    """
    elem = ""
    action = ""
    value = ""

    for line in ans.splitlines():
        line_stripped = line.strip()
        if re.match(r"^element\s*:", line_stripped, flags=re.IGNORECASE):
            elem = line_stripped.split(":", 1)[1].strip()
        elif re.match(r"^action\s*:", line_stripped, flags=re.IGNORECASE):
            action = line_stripped.split(":", 1)[1].strip()
        elif re.match(r"^value\s*:", line_stripped, flags=re.IGNORECASE):
            value = line_stripped.split(":", 1)[1].strip()

    pred_id = elem if elem.isdigit() else _extract_id_from_element_text(elem)
    pred_action = _normalize_action(action)
    pred_value = _normalize_value(value)

    return pred_id, pred_action, pred_value, elem


def _parse_ground_truth(gt: str) -> Tuple[Optional[str], str, str]:
    """
    Ground truth format example:
      Element: <svg id=1 />
      Action: CLICK
    Value may be absent; treat as empty.
    """
    elem_line = ""
    action_line = ""
    value_line = ""

    for line in gt.splitlines():
        line = line.strip()
        if line.lower().startswith("element:"):
            elem_line = line.split(":", 1)[1].strip()
        elif line.lower().startswith("action:"):
            action_line = line.split(":", 1)[1].strip()
        elif line.lower().startswith("value:"):
            value_line = line.split(":", 1)[1].strip()

    gt_id = _extract_id_from_element_text(elem_line) if elem_line else None
    gt_action = _normalize_action(action_line)
    gt_value = _normalize_value(value_line)

    return gt_id, gt_action, gt_value


def build_sample(sample, top_k=-1, neg_ratio=0.1, num_candidates=10, mode='generation', seed=42, sample_idx=0):
    """构建单个样本的输入输出"""

    random.seed(seed + sample_idx)

    if top_k > 0:
        top_negatives = [c for c in sample["neg_candidates"] if c["rank"] < top_k]
        other_negatives = [c for c in sample["neg_candidates"] if c["rank"] >= top_k]
    else:
        top_negatives = []
        other_negatives = sample["neg_candidates"]
    
    if random.random() < 0.8 and len(top_negatives) > 0:
        neg_candidates = top_negatives
    else:
        neg_candidates = other_negatives

    if len(sample["pos_candidates"]) != 0 and (
        random.random() > neg_ratio or len(neg_candidates) == 0
    ):
        pos_candidate = random.choice(sample["pos_candidates"])
        neg_candidate = random.sample(
            neg_candidates,
            min(len(neg_candidates), num_candidates - 1),
        )
        gt = pos_candidate["backend_node_id"]
        candidate_ids = [gt] + [c["backend_node_id"] for c in neg_candidate]
        if mode == "multichoice":
            seq_context, seq_in, seq_out, _ = format_input_multichoice(
                sample, candidate_ids, gt, keep_html_brackets=True
            )
        else:
            seq_context, seq_in, seq_out, _ = format_input_generation(
                sample, candidate_ids, gt, keep_html_brackets=True
            )
    else:
        neg_candidate = random.sample(
            neg_candidates,
            min(len(neg_candidates), num_candidates),
        )
        gt = -1
        candidate_ids = [c["backend_node_id"] for c in neg_candidate]
        if mode == "multichoice":
            seq_context, seq_in, seq_out, _ = format_input_multichoice(
                sample, candidate_ids, gt, keep_html_brackets=True
            )
        else:
            seq_context, seq_in, seq_out, _ = format_input_generation(
                sample, candidate_ids, gt, keep_html_brackets=True
            )
    
    return seq_context, seq_in, seq_out


def evaluate_single_sample(
    model,
    tokenizer,
    sample: Dict,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    token_mapping: Dict = None,
    equal_length: bool = False,
    mode: str = 'generation',
    verbose: bool = False,
    max_retries: int = 3,
    sample_idx: int = 0
) -> Tuple[bool, Dict, str]:
    """
    评估单个Mind2Web样本
    
    Returns:
        success: 预测是否正确
        result_dict: 包含详细结果的字典
        error_msg: 错误信息（如果有）
    """
    error_msg = ""
    
    try:
        # 构建样本输入
        seq_context, seq_in, seq_out = build_sample(
            sample, 
            top_k=-1, 
            neg_ratio=0.1, 
            num_candidates=10, 
            mode=mode,
            sample_idx=sample_idx
        )
        
        # 解析ground truth
        gt_id, gt_action, gt_value = _parse_ground_truth(seq_out)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Context: {seq_context[:200]}...")
            print(f"Query: {seq_in[:200]}...")
            print(f"Ground Truth: {seq_out}")
            print(f"GT parsed - ID: {gt_id}, Action: {gt_action}, Value: {gt_value}")
        
        # 应用token压缩
        if token_mapping:
            seq_context = replace_text_with_tokens(
                seq_context,
                token_mapping,
                tokenizer=tokenizer,
                equal_length=equal_length
            )
            seq_in = replace_text_with_tokens(
                seq_in,
                token_mapping,
                tokenizer=tokenizer,
                equal_length=equal_length
            )
        
        # 构建messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{seq_context}\n{seq_in}"}
        ]
        
        # 应用chat template
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        model_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)
        pad_token_id = tokenizer.pad_token_id
        mask_padding_tokens = (
            equal_length
            and pad_token_id is not None
            and pad_token_id != tokenizer.eos_token_id
        )
        attention_mask = build_attention_mask(
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            mask_padding_tokens=mask_padding_tokens
        ).to(model_device)

        prompt_tokens = int(input_ids.shape[1])
        
        # 尝试多次生成
        pred_id = None
        pred_action = None
        pred_value = None
        model_answer = None
        generated_tokens = 0
        total_tokens = prompt_tokens
        generation_eos_token_id = resolve_generation_eos_token_id(tokenizer)
        
        for attempt in range(max_retries):
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": generation_eos_token_id
            }
            if temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            else:
                generation_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = model.generate(**generation_kwargs)

            generated_ids = output_ids[0, prompt_tokens:]
            model_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            model_answer = strip_eot_markers(model_answer)
            generated_tokens = int(generated_ids.shape[0])
            total_tokens = prompt_tokens + generated_tokens
            
            if verbose:
                print(f"\nAttempt {attempt + 1}:")
                print(f"Model response: {model_answer}")
            
            # 解析模型输出
            pred_id, pred_action, pred_value, elem_raw = _parse_model_answer(model_answer)
            
            if pred_id:
                break
            
            if verbose:
                print(f"⚠️ Failed to parse element ID, retrying...")
        
        # 检查预测结果
        if not pred_id:
            error_msg = f"Failed to parse element ID after {max_retries} attempts"
            success = False
            id_match = False
            action_match = False
            value_match = False
        else:
            # 比较预测和ground truth
            id_match = (pred_id == gt_id)
            action_match = (pred_action == gt_action)
            value_match = (pred_value == gt_value) if gt_value else True
            
            success = id_match and action_match and value_match
            
            if verbose:
                print(f"\nComparison:")
                print(f"  ID: {pred_id} vs {gt_id} - {'✓' if id_match else '✗'}")
                print(f"  Action: {pred_action} vs {gt_action} - {'✓' if action_match else '✗'}")
                print(f"  Value: {pred_value} vs {gt_value} - {'✓' if value_match else '✗'}")
                print(f"  Overall: {'✓ CORRECT' if success else '✗ WRONG'}")
        
        result_dict = {
            "context": seq_context,
            "query": seq_in,
            "ground_truth": seq_out,
            "gt_parsed": {
                "id": gt_id,
                "action": gt_action,
                "value": gt_value
            },
            "prediction": {
                "id": pred_id,
                "action": pred_action,
                "value": pred_value,
                "raw_answer": model_answer
            },
            "tokens": {
                "prompt": prompt_tokens,
                "generated": generated_tokens,
                "total": total_tokens
            },
            "correct": success,
            "element_correct": id_match,
            "action_correct": action_match,
            "value_correct": value_match
        }
        
        return success, result_dict, error_msg
    
    except Exception as e:
        error_msg = f"Error processing sample: {str(e)}\n{traceback.format_exc()}"
        return False, {}, error_msg

def evaluate_model(
    model,
    tokenizer,
    dataset,
    token_mapping: Dict,
    use_compression: bool = True,
    max_samples: int = None,
    verbose: bool = False,
    output_dir: str = "eval_output_mind2web",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.95,
    equal_length: bool = False,
    mode: str = 'generation'
):
    """
    评估模型在Mind2Web数据集上的性能
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备system prompt
    if use_compression and token_mapping:
        system_prompt = replace_text_with_tokens(
            SYSTEM_PROMPT,
            token_mapping,
            tokenizer=tokenizer,
            equal_length=equal_length
        )
        prompt_type = "Compressed_equal_length" if equal_length else "Compressed"
    else:
        system_prompt = SYSTEM_PROMPT
        prompt_type = "Original"
    
    print(f"\n{'='*80}")
    print(f"📊 Evaluating with {prompt_type} Prompt")
    print(f"{'='*80}")
    print(f"Prompt length: {len(system_prompt)} chars")
    original_prompt_tokens = len(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False))
    current_prompt_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    print(
        f"Prompt tokens: {current_prompt_tokens} "
        f"(original: {original_prompt_tokens}, delta: {current_prompt_tokens - original_prompt_tokens})"
    )
    print(f"Dataset size: {len(dataset)}")
    if max_samples:
        print(f"Evaluating on: {max_samples} samples")
    print()
    
    correct = 0
    element_correct = 0
    action_correct = 0
    value_correct = 0
    element_and_action_correct = 0
    total = 0
    total_tokens = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    
    results = []
    detailed_logs = []
    errors = []
    
    # 限制样本数
    eval_indices = range(min(max_samples or len(dataset), len(dataset)))
    
    for idx in tqdm(eval_indices, desc=f"Evaluating ({prompt_type})"):
        try:
            sample = dataset[idx]
            
            success, result_dict, error_msg = evaluate_single_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                token_mapping=token_mapping if use_compression else None,
                equal_length=equal_length if use_compression else False,
                mode=mode,
                verbose=verbose,
                max_retries=3,
                sample_idx=idx
            )
            
            if success:
                correct += 1
            
            # 统计各个维度的正确性
            if result_dict.get('element_correct', False):
                element_correct += 1
            if result_dict.get('action_correct', False):
                action_correct += 1
            if result_dict.get('value_correct', False):
                value_correct += 1
            if result_dict.get('element_correct', False) and result_dict.get('action_correct', False):
                element_and_action_correct += 1
            
            total += 1
            
            if result_dict and 'tokens' in result_dict:
                total_tokens += result_dict['tokens']['total']
                total_prompt_tokens += result_dict['tokens']['prompt']
                total_generated_tokens += result_dict['tokens']['generated']
            
            # 保存结果
            result = {
                "sample_id": idx,
                "correct": success,
                "error": error_msg if error_msg else None
            }
            results.append(result)
            
            # 保存详细日志
            detailed_log = {
                "sample_id": idx,
                "result": result_dict,
                "error": error_msg if error_msg else None
            }
            detailed_logs.append(detailed_log)
            
            if error_msg:
                errors.append({
                    "sample_id": idx,
                    "error": error_msg
                })
            
            # 每50个样本打印进度
            if total % 50 == 0:
                acc = correct / total * 100
                elem_acc = element_correct / total * 100
                action_acc = action_correct / total * 100
                avg_tokens = total_tokens / total
                print(f"Progress: {total} samples | Overall: {acc:.2f}% | Element: {elem_acc:.2f}% | Action: {action_acc:.2f}% | Avg tokens: {avg_tokens:.1f}")
                
        except Exception as e:
            error_msg = f"Fatal error on sample {idx}: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            errors.append({
                "sample_id": idx,
                "error": error_msg
            })
            continue
    
    # 计算最终指标
    overall_accuracy = correct / total * 100 if total > 0 else 0
    element_accuracy = element_correct / total * 100 if total > 0 else 0
    action_accuracy = action_correct / total * 100 if total > 0 else 0
    value_accuracy = value_correct / total * 100 if total > 0 else 0
    element_action_accuracy = element_and_action_correct / total * 100 if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_prompt_tokens = total_prompt_tokens / total if total > 0 else 0
    avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
    
    # 保存详细日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"detailed_logs_{prompt_type.lower()}_{timestamp}.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed logs saved to: {log_file}")
    
    # 保存错误日志
    if errors:
        error_file = os.path.join(output_dir, f"errors_{prompt_type.lower()}_{timestamp}.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"⚠️  Errors saved to: {error_file}")
    
    return {
        "overall_accuracy": overall_accuracy,
        "element_accuracy": element_accuracy,
        "action_accuracy": action_accuracy,
        "value_accuracy": value_accuracy,
        "element_action_accuracy": element_action_accuracy,
        "correct": correct,
        "element_correct": element_correct,
        "action_correct": action_correct,
        "value_correct": value_correct,
        "element_action_correct": element_and_action_correct,
        "total": total,
        "prompt_type": prompt_type,
        "avg_tokens": avg_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_generated_tokens": avg_generated_tokens,
        "results": results,
        "errors": errors,
        "log_file": log_file
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mind2Web model with transformers.generate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mapping", type=str, default="token_mapping.json", help="Token mapping file")
    parser.add_argument("--baseline", action="store_true", help="Also evaluate baseline")
    parser.add_argument("--samples", type=int, default=500, help="Number of test samples")
    # parser.add_argument("--data_path", type=str, default="../../Mind2Web/data/test_*/*.json", help="Path to test data")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output_dir", type=str, default="eval_output_mind2web", help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--model_name", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct", 
                        help="Base model name")
    parser.add_argument("--test_split", type=str, default="all", choices=["all", "task", "website", "domain"], help="Data split to evaluate")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Legacy arg (ignored in transformers backend)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Legacy arg (ignored in transformers backend)")
    parser.add_argument(
        "--equal_length",
        action="store_true",
        help="Pad replaced spans with tokenizer pad_token so compressed prompt keeps similar token length."
    )
    parser.add_argument("--mode", type=str, default="generation", choices=["generation", "multichoice"],
                        help="Evaluation mode")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("🧪 Evaluating Mind2Web Model (using transformers)")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    # print(f"Data path: {args.data_path}")
    print(f"Samples: {args.samples}")
    print(f"Mode: {args.mode}")
    print("Backend: transformers.generate")
    print(f"Equal length mode: {args.equal_length}")
    print(f"Output directory: {args.output_dir}\n")
    
    # 加载数据集
    print(f"📚 Loading Mind2Web test set from ./../Mind2Web/data...")
    try:
        data_path = ""
        if args.test_split == "all":
            data_path = "../github/Mind2Web/data/test_*/*0.json"
        elif args.test_split == "task":
            data_path = "../github/Mind2Web/data/test_task/*.json"
        elif args.test_split == "website":
            data_path = "../github/Mind2Web/data/test_website/*.json"
        elif args.test_split == "domain":
            data_path = "../github/Mind2Web/data/test_domain/*.json"
        dataset = get_data_split('json', data_path, is_train=False)
        print(f"✅ Loaded {len(dataset)} samples\n")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(traceback.format_exc())
        return
    
    # 加载token映射
    token_mapping = None
    special_tokens = []
    if os.path.exists(args.mapping):
        print("🔤 Loading token mapping...")
        try:
            token_mapping, special_tokens = load_token_mapping(args.mapping)
            print(f"✅ Loaded {len(token_mapping)} token mappings\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not load token mapping: {e}")
            print("Proceeding without compression...\n")
    
    # 加载tokenizer
    print("🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            vocab = tokenizer.get_vocab()
            if args.equal_length and "<|finetune_right_pad_id|>" in vocab:
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
                print("   Using built-in <|finetune_right_pad_id|> as pad_token for equal_length mode.")
            elif "<pad>" in vocab:
                tokenizer.pad_token = "<pad>"
                print("   Using built-in <pad> as pad_token.")
            else:
                tokenizer.pad_token = tokenizer.eos_token
                print("   pad_token not found in tokenizer, fallback to eos_token.")

        # 添加特殊tokens（压缩映射）
        if special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        print(f"   Vocab size: {len(tokenizer)}\n")

        if args.equal_length and tokenizer.pad_token_id == tokenizer.eos_token_id:
            raise ValueError(
                "equal_length requires a dedicated pad_token distinct from eos_token, "
                "otherwise attention mask cannot isolate only the supplemented padding tokens."
            )
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print(traceback.format_exc())
        return

    def load_model(model_path: str):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto"
        )
        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        return model
    
    # 评估baseline（可选）
    baseline_results = None
    if args.baseline:
        print("📏 Loading baseline model with transformers...")
        try:
            baseline_model = load_model(args.model_name)

            baseline_results = evaluate_model(
                baseline_model,
                tokenizer,
                dataset,
                token_mapping={},
                use_compression=False,
                max_samples=args.samples,
                verbose=args.verbose,
                output_dir=args.output_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                equal_length=False,
                mode=args.mode
            )
            
            print(f"\n✅ Baseline Results:")
            print(f"   Overall Accuracy: {baseline_results['overall_accuracy']:.2f}% ({baseline_results['correct']}/{baseline_results['total']})")
            print(f"   Element Accuracy: {baseline_results['element_accuracy']:.2f}% ({baseline_results['element_correct']}/{baseline_results['total']})")
            print(f"   Action Accuracy: {baseline_results['action_accuracy']:.2f}% ({baseline_results['action_correct']}/{baseline_results['total']})")
            print(f"   Element+Action Accuracy: {baseline_results['element_action_accuracy']:.2f}% ({baseline_results['element_action_correct']}/{baseline_results['total']})")
            print(f"   Value Accuracy: {baseline_results['value_accuracy']:.2f}% ({baseline_results['value_correct']}/{baseline_results['total']})")
            
            del baseline_model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error evaluating baseline: {e}")
            print(traceback.format_exc())
    
    # 评估压缩模型
    print(f"\n🤖 Loading compressed model with transformers...")
    try:
        compressed_model = load_model(args.checkpoint)

        compressed_results = evaluate_model(
            compressed_model,
            tokenizer,
            dataset,
            token_mapping=token_mapping or {},
            use_compression=bool(token_mapping),
            max_samples=args.samples,
            verbose=args.verbose,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            equal_length=args.equal_length,
            mode=args.mode
        )

        del compressed_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error evaluating compressed model: {e}")
        print(traceback.format_exc())
        return
    
    # 打印最终结果
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    
    if baseline_results:
        print(f"\n📏 Baseline (Original Prompt):")
        print(f"   Overall Accuracy: {baseline_results['overall_accuracy']:.2f}% ({baseline_results['correct']}/{baseline_results['total']})")
        print(f"   ├─ Element Accuracy: {baseline_results['element_accuracy']:.2f}% ({baseline_results['element_correct']}/{baseline_results['total']})")
        print(f"   ├─ Action Accuracy: {baseline_results['action_accuracy']:.2f}% ({baseline_results['action_correct']}/{baseline_results['total']})")
        print(f"   ├─ Element+Action: {baseline_results['element_action_accuracy']:.2f}% ({baseline_results['element_action_correct']}/{baseline_results['total']})")
        print(f"   └─ Value Accuracy: {baseline_results['value_accuracy']:.2f}% ({baseline_results['value_correct']}/{baseline_results['total']})")
        print(f"   Avg tokens: {baseline_results['avg_tokens']:.1f}")
        if baseline_results['errors']:
            print(f"   ⚠️  Errors: {len(baseline_results['errors'])}")
    
    print(f"\n🎯 Compressed Model:")
    print(f"   Overall Accuracy: {compressed_results['overall_accuracy']:.2f}% ({compressed_results['correct']}/{compressed_results['total']})")
    print(f"   ├─ Element Accuracy: {compressed_results['element_accuracy']:.2f}% ({compressed_results['element_correct']}/{compressed_results['total']})")
    print(f"   ├─ Action Accuracy: {compressed_results['action_accuracy']:.2f}% ({compressed_results['action_correct']}/{compressed_results['total']})")
    print(f"   ├─ Element+Action: {compressed_results['element_action_accuracy']:.2f}% ({compressed_results['element_action_correct']}/{compressed_results['total']})")
    print(f"   └─ Value Accuracy: {compressed_results['value_accuracy']:.2f}% ({compressed_results['value_correct']}/{compressed_results['total']})")
    print(f"   Avg tokens: {compressed_results['avg_tokens']:.1f}")
    if compressed_results['errors']:
        print(f"   ⚠️  Errors: {len(compressed_results['errors'])}")
    
    if baseline_results:
        print(f"\n📊 Performance Comparison:")
        overall_gap = baseline_results['overall_accuracy'] - compressed_results['overall_accuracy']
        element_gap = baseline_results['element_accuracy'] - compressed_results['element_accuracy']
        action_gap = baseline_results['action_accuracy'] - compressed_results['action_accuracy']
        elem_action_gap = baseline_results['element_action_accuracy'] - compressed_results['element_action_accuracy']
        
        print(f"   Overall Accuracy Gap: {overall_gap:+.2f}%")
        print(f"   Element Accuracy Gap: {element_gap:+.2f}%")
        print(f"   Action Accuracy Gap: {action_gap:+.2f}%")
        print(f"   Element+Action Gap: {elem_action_gap:+.2f}%")
        
        if baseline_results['avg_tokens'] > 0:
            token_reduction = (1 - compressed_results['avg_tokens'] / baseline_results['avg_tokens']) * 100
            print(f"   Token Reduction: {token_reduction:.1f}%")
            print(f"   💡 Trade-off: {token_reduction:.1f}% token reduction for {overall_gap:+.2f}% overall accuracy change")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"evaluation_results_mind2web_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline_results,
            'compressed': compressed_results,
            'timestamp': timestamp,
            'args': vars(args)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
