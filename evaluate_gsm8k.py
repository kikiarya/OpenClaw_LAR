import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
import json
import argparse
from typing import List, Dict, Tuple, Optional
import traceback
from datetime import datetime
import os


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

        replaced_text = replaced_text.replace(segment, replacement)

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
    """Include eot token ids as eos when tokenizer supports them."""
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


def extract_answer(text: str) -> str:
    """
    从生成的文本中提取最终答案
    支持多种格式：
    - #### 1234
    - The answer is 1234
    - <answer>1234</answer>
    """
    text = text.strip()
    
    # 格式1: #### 后面的数字 (GSM8K标准格式)
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # 格式2: <answer>标签
    match = re.search(r'<answer>\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # 格式3: "The answer is"
    match = re.search(r'(?:the answer is|answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # 格式4: 最后一行的数字
    lines = text.strip().split('\n')
    for line in reversed(lines):
        match = re.search(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if match:
            return match.group(1).replace(',', '')
    
    return ""


def check_answer(predicted: str, ground_truth: str) -> bool:
    """检查答案是否正确"""
    try:
        # 移除逗号并转换为数字
        pred_clean = predicted.replace(',', '').strip()
        gt_clean = ground_truth.replace(',', '').strip()
        
        # 尝试作为整数比较
        try:
            return int(float(pred_clean)) == int(float(gt_clean))
        except:
            # 如果不是整数，作为浮点数比较
            return abs(float(pred_clean) - float(gt_clean)) < 1e-6
    except:
        return False


def run_gsm8k_inference(
    model,
    tokenizer,
    question: str,
    ground_truth: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    token_mapping: Dict[str, str] = None,
    equal_length: bool = False,
    verbose: bool = False
) -> Tuple[bool, str, str, int, int]:
    """
    对单个GSM8K问题进行推理
    
    Returns:
        success: 是否正确
        predicted_answer: 预测的答案
        full_response: 完整的生成文本
        total_tokens: 总token数
        prompt_tokens: prompt token数
    """
    try:
        if token_mapping:
            question = replace_text_with_tokens(
                question,
                token_mapping,
                tokenizer=tokenizer,
                equal_length=equal_length
            )

        # 构建prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            text,
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
        generation_eos_token_id = resolve_generation_eos_token_id(tokenizer)
        
        # 生成响应
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": generation_eos_token_id
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)

        generated_ids = output_ids[0, prompt_tokens:]
        generated_tokens = int(generated_ids.shape[0])
        total_tokens = prompt_tokens + generated_tokens

        # 解码响应
        full_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        full_response = strip_eot_markers(full_response)
        
        # 提取答案
        predicted_answer = extract_answer(full_response)
        
        # 检查正确性
        success = check_answer(predicted_answer, ground_truth)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Full Response:\n{full_response}")
            print(f"Extracted Answer: {predicted_answer}")
            print(f"Correct: {success}")
            print(f"{'='*60}\n")
        
        return success, predicted_answer, full_response, total_tokens, prompt_tokens
        
    except Exception as e:
        error_msg = f"Error in inference: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return False, "", "", 0, 0


def evaluate_model(
    model, 
    tokenizer, 
    dataset, 
    token_mapping, 
    use_compression=True, 
    max_samples=None, 
    verbose=False, 
    output_dir="eval_output",
    max_new_tokens=512,
    temperature=0.7,
    equal_length=False
):
    """
    评估模型在GSM8K上的性能
    
    Args:
        model: 模型
        tokenizer: tokenizer
        dataset: GSM8K测试数据集
        token_mapping: token映射（用于压缩）
        use_compression: 是否使用压缩的prompt
        max_samples: 最大样本数
        verbose: 是否打印详细信息
        output_dir: 输出目录
        max_new_tokens: 最大生成token数
        temperature: 生成温度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # GSM8K System Prompt
    SYSTEM_PROMPT = """You are a mathematical reasoning assistant. Follow these steps:
1. Read the problem carefully
2. Break it down into steps
3. Show your work clearly
4. Provide the final numerical answer

Format your response as:
Reasoning: [your step-by-step solution]
Answer: [numerical answer only]"""
    
    if use_compression:
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
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Prompt preview: {system_prompt[:200]}...\n")
    
    correct = 0
    total = 0
    total_tokens = 0
    total_prompt_tokens = 0
    errors = []
    
    results = []
    detailed_logs = []
    
    # 限制样本数
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating ({prompt_type})")):
        try:
            question = item["question"]
            ground_truth = item["answer"].split("####")[-1].strip()
            
            if verbose:
                print(f"\n{'#'*80}")
                print(f"Sample {idx + 1}/{len(dataset)}")
                print(f"{'#'*80}")
            
            # 运行推理
            success, predicted_answer, full_response, tokens, prompt_tokens = run_gsm8k_inference(
                model=model,
                tokenizer=tokenizer,
                question=question,
                ground_truth=ground_truth,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                token_mapping=token_mapping if use_compression else None,
                equal_length=equal_length if use_compression else False,
                temperature=temperature,
                verbose=verbose
            )
            
            if success:
                correct += 1
            
            total += 1
            total_tokens += tokens
            total_prompt_tokens += prompt_tokens
            
            result = {
                "sample_id": idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "correct": success,
                "total_tokens": tokens,
                "prompt_tokens": prompt_tokens
            }
            results.append(result)
            
            # 存储详细日志
            detailed_log = {
                "sample_id": idx,
                "question": question,
                "ground_truth": ground_truth,
                "full_response": full_response,
                "predicted_answer": predicted_answer,
                "correct": success,
                "total_tokens": tokens,
                "prompt_tokens": prompt_tokens
            }
            detailed_logs.append(detailed_log)
            
            # 每10个样本打印一次进度
            if total % 10 == 0:
                print(f"Progress: {total} samples, Accuracy: {correct/total*100:.2f}%")
                
        except Exception as e:
            error_msg = f"Error processing sample {idx}: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            errors.append({
                "sample_id": idx,
                "question": item.get("question", ""),
                "error": error_msg
            })
            continue
    
    accuracy = correct / total * 100 if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_prompt_tokens = total_prompt_tokens / total if total > 0 else 0
    
    # 保存详细日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"detailed_logs_{prompt_type.lower()}_{timestamp}.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed logs saved to: {log_file}")
    
    # 保存错误
    if errors:
        error_file = os.path.join(output_dir, f"errors_{prompt_type.lower()}_{timestamp}.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"⚠️  Errors saved to: {error_file}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "prompt_type": prompt_type,
        "avg_tokens": avg_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "results": results,
        "errors": errors,
        "log_file": log_file
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument("--mapping", type=str, default="token_mapping.json", help="Token mapping file")
    parser.add_argument("--baseline", action="store_true", help="Also evaluate baseline")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument(
        "--equal_length",
        action="store_true",
        help="Pad replaced spans with tokenizer pad_token so compressed prompt keeps similar token length."
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="Output directory for logs")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("🧪 Evaluating Compressed Model on GSM8K")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mapping: {args.mapping}")
    print(f"Samples: {args.samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Equal length mode: {args.equal_length}")
    print(f"Output directory: {args.output_dir}")
    print(f"Verbose: {args.verbose}\n")
    
    # 加载数据集
    print("📚 Loading GSM8K test set...")
    try:
        dataset = load_dataset("gsm8k", "main")
        test_data = dataset["test"]
        print(f"✅ Loaded {len(test_data)} samples\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(traceback.format_exc())
        return
    
    # 加载token映射
    print("🔤 Loading token mapping...")
    try:
        token_mapping, special_tokens = load_token_mapping(args.mapping)
        print(f"✅ Loaded {len(token_mapping)} token mappings\n")
    except Exception as e:
        print(f"❌ Error loading token mapping: {e}")
        print(traceback.format_exc())
        return
    
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
        
        # 添加特殊tokens
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
    
    # 评估baseline（可选）
    baseline_results = None
    if args.baseline:
        print("📏 Loading baseline model...")
        try:
            baseline_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            baseline_model.eval()
            
            baseline_results = evaluate_model(
                baseline_model,
                tokenizer,
                test_data,
                token_mapping,
                use_compression=False,
                max_samples=args.samples,
                verbose=args.verbose,
                output_dir=args.output_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                equal_length=False
            )
            
            print(f"\n✅ Baseline Results:")
            print(f"   Accuracy: {baseline_results['accuracy']:.2f}%")
            print(f"   Correct: {baseline_results['correct']}/{baseline_results['total']}")
            
            del baseline_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error evaluating baseline: {e}")
            print(traceback.format_exc())
    
    # 评估压缩模型
    print(f"\n🤖 Loading compressed model (LoRA)...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 调整embedding大小
        base_model.resize_token_embeddings(len(tokenizer))
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(
            base_model,
            args.checkpoint,
            is_trainable=False
        )
        model.eval()
        
        compressed_results = evaluate_model(
            model,
            tokenizer,
            test_data,
            token_mapping,
            use_compression=True,
            max_samples=args.samples,
            verbose=args.verbose,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            equal_length=args.equal_length
        )
        
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
        print(f"   Accuracy: {baseline_results['accuracy']:.2f}%")
        print(f"   Correct: {baseline_results['correct']}/{baseline_results['total']}")
        print(f"   Avg prompt tokens: {baseline_results['avg_prompt_tokens']:.1f}")
        if baseline_results['errors']:
            print(f"   ⚠️  Errors: {len(baseline_results['errors'])}")
    
    print(f"\n🎯 Compressed Model (LoRA + Special Tokens):")
    print(f"   Accuracy: {compressed_results['accuracy']:.2f}%")
    print(f"   Correct: {compressed_results['correct']}/{compressed_results['total']}")
    print(f"   Avg prompt tokens: {compressed_results['avg_prompt_tokens']:.1f}")
    if compressed_results['errors']:
        print(f"   ⚠️  Errors: {len(compressed_results['errors'])}")
    
    if baseline_results:
        accuracy_gap = baseline_results['accuracy'] - compressed_results['accuracy']
        if baseline_results['avg_prompt_tokens'] > 0:
            token_reduction = (1 - compressed_results['avg_prompt_tokens'] / baseline_results['avg_prompt_tokens']) * 100
            print(f"\n📉 Performance Gap: {accuracy_gap:.2f}%")
            print(f"💡 Token reduction: {token_reduction:.1f}%")
            print(f"💡 Trade-off: {token_reduction:.1f}% token reduction for {accuracy_gap:.2f}% accuracy change")
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"evaluation_results_gsm8k_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline_results,
            'compressed': compressed_results,
            'timestamp': timestamp,
            'args': vars(args)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Summary results saved to: {output_file}")
    print(f"📁 All outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
