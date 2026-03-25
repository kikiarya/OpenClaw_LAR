import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from typing import List, Dict, Tuple, Optional
import traceback
from datetime import datetime

# Import KodCode environment
from env.kodcode import KodCodeReActEnv

def load_token_mapping(mapping_file):
    """加载 token 映射"""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['token_mapping'], data['special_tokens']


def replace_text_with_tokens(text, token_mapping, tokenizer=None, equal_length=False):
    """将文本中的 segments 替换为特殊 tokens，可选补齐到等长 token 数。"""
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

        replaced_text = (
            replaced_text
            .replace(' ' + segment + ' ', replacement + ' ')
            .replace(segment + ' ', replacement + ' ')
        )

    if equal_length:
        replaced_total_tokens = len(tokenizer.encode(replaced_text, add_special_tokens=False))
        if replaced_total_tokens < original_total_tokens:
            pad_count = original_total_tokens - replaced_total_tokens
            replaced_text = replaced_text + "".join([pad_token] * pad_count)

    return replaced_text


# ---------------------------------------------------------------------------
# Generation helpers (ported from evaluate_triviaqa.py)
# ---------------------------------------------------------------------------

class StopOnTokenSequences(StoppingCriteria):
    """Stop generation when any stop token sequence appears as suffix."""

    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = [torch.tensor(seq, dtype=torch.long) for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_sequences:
            return False
        current = input_ids[0]
        for seq in self.stop_sequences:
            seq_len = seq.size(0)
            if current.size(0) >= seq_len:
                if torch.equal(current[-seq_len:], seq.to(current.device)):
                    return True
        return False


def build_attention_mask(
    input_ids: torch.LongTensor,
    pad_token_id: Optional[int],
    mask_padding_tokens: bool,
) -> torch.LongTensor:
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


# ---------------------------------------------------------------------------
# Episode runner  (transformers backend)
# ---------------------------------------------------------------------------

def run_kodcode_episode(
    model,
    tokenizer,
    task_config: Dict,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    stop_strings: List[str],
    max_turns: int = 10,
    token_mapping: Dict = None,
    equal_length: bool = False,
    verbose: bool = False,
) -> Tuple[bool, float, int, int, int, List[Dict], str]:
    """
    运行一个完整的 KodCode 交互回合（使用 transformers.generate）

    Returns:
        success, reward, turns, total_tokens, prompt_tokens,
        conversation_history, error_msg
    """
    error_msg = ""
    conversation_history = []

    pad_token_id = tokenizer.pad_token_id
    model_device = next(model.parameters()).device
    mask_padding_tokens = (
        equal_length
        and pad_token_id is not None
        and pad_token_id != tokenizer.eos_token_id
    )
    generation_eos_token_id = resolve_generation_eos_token_id(tokenizer)
    stop_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
    stopping_criteria = StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])

    try:
        # 初始化环境
        env = KodCodeReActEnv()
        user_prompt = env.set_env(task_config)

        messages = [{"role": "user", "content": user_prompt}]
        conversation_history.append({"turn": 0, "role": "user", "content": user_prompt})

        done = False
        reward = 0.0
        turn = 0
        total_tokens = 0
        prompt_tokens = 0

        while not done and turn < max_turns:
            turn += 1
            try:
                chat = [{"role": "system", "content": system_prompt}]
                chat.extend(messages)

                prompt_text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs["input_ids"].to(model_device)
                attention_mask = build_attention_mask(
                    input_ids=input_ids,
                    pad_token_id=pad_token_id,
                    mask_padding_tokens=mask_padding_tokens,
                ).to(model_device)

                prompt_len = input_ids.shape[1]
                prompt_tokens += prompt_len

                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": generation_eos_token_id,
                    "stopping_criteria": stopping_criteria,
                }
                if temperature > 0:
                    generation_kwargs["do_sample"] = True
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["top_p"] = 1.0
                else:
                    generation_kwargs["do_sample"] = False

                with torch.no_grad():
                    output_ids = model.generate(**generation_kwargs)

                generated_ids = output_ids[0, prompt_len:]
                llm_response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
                llm_response = strip_eot_markers(llm_response)

                generated_tokens = generated_ids.shape[0]
                total_tokens += prompt_len + generated_tokens

                # 环境执行动作（先 step，再替换展示用文本）
                observation, reward, done = env.step(llm_response)

                # 替换为特殊 tokens（用于后续对话 context 压缩）
                llm_response_compressed = (
                    replace_text_with_tokens(
                        llm_response,
                        token_mapping,
                        tokenizer=tokenizer,
                        equal_length=equal_length,
                    )
                    if token_mapping else llm_response
                )

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Turn {turn}:")
                    print(f"{'='*60}")
                    print(f"LLM Response:\n{llm_response_compressed}\n")

                messages.append({"role": "assistant", "content": llm_response_compressed})
                conversation_history.append({
                    "turn": turn,
                    "role": "assistant",
                    "content": llm_response_compressed,
                    "tokens_used": int(generated_tokens),
                })

                if verbose:
                    print(f"Observation: {observation if observation else '(empty)'}")
                    print(f"Done: {done}, Reward: {reward}\n")

                if not done and observation:
                    messages.append({"role": "user", "content": observation})
                    conversation_history.append({
                        "turn": turn,
                        "role": "observation",
                        "content": observation,
                    })

            except Exception as e:
                error_msg = f"Error in turn {turn}: {str(e)}\n{traceback.format_exc()}"
                print(f"❌ {error_msg}")
                break

        success = reward > 0.0
        return success, reward, turn, total_tokens, prompt_tokens, conversation_history, error_msg

    except Exception as e:
        error_msg = f"Fatal error in episode: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return False, 0.0, 0, 0, 0, conversation_history, error_msg


# ---------------------------------------------------------------------------
# Model-level evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    dataset,
    token_mapping,
    use_compression=True,
    max_samples=None,
    verbose=False,
    output_dir="eval_output",
    temperature=0.0,
    max_tokens=8192,
    equal_length=False,
):
    """
    评估模型性能（使用 transformers）

    Args:
        model: transformers CausalLM 实例
        tokenizer: tokenizer
        dataset: 测试数据集（KodCode 格式）
        token_mapping: token 映射（用于压缩）
        use_compression: 是否使用压缩的 prompt
        max_samples: 最大样本数
        verbose: 是否打印详细信息
        output_dir: 输出目录
        temperature: 采样温度
        max_tokens: 每轮最大生成 token 数
        equal_length: 替换后是否用 pad_token 补齐 token 长度
    """
    os.makedirs(output_dir, exist_ok=True)

    SYSTEM_PROMPT = """You are a coding assistant. Solve the user's problem using the provided Python execution tool.

TOOL SPECIFICATION:
You can execute Python code using the following tags:
<execute>
# Code to execute
</execute>

The system will return the output in an <observation> tag.

OUTPUT FORMAT:
Your response must strictly follow this sequence:
1. <think> [Your thought process] </think>
2. <execute> [Code snippet] </execute>
3. ... Wait for <observation> ...
4. Repeat as needed.
5. <answer> [Final code solution only] </answer>

CONSTRAINTS:
- Use the provided format strictly.
- The <answer> tag must contain the final function definition.
"""

    if use_compression:
        system_prompt = replace_text_with_tokens(
            SYSTEM_PROMPT,
            token_mapping,
            tokenizer=tokenizer,
            equal_length=equal_length,
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
    print(f"Prompt preview: {system_prompt[:300]}...\n")
    print(tokenizer.tokenize(SYSTEM_PROMPT, add_special_tokens=False))
    print('=' * 100)
    print(tokenizer.tokenize(system_prompt, add_special_tokens=False))

    stop_strings = ["</execute>", "</answer>"]

    correct = 0
    total = 0
    total_turns = 0
    total_tokens = 0
    total_prompt_tokens = 0
    errors = []
    results = []
    detailed_logs = []

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating ({prompt_type})")):
        try:
            task_id = item.get("question_id", idx)
            question = (item.get("question") or "").strip()
            solution = (item.get("solution") or "").strip()
            test = (item.get("test") or "").strip()
            test_info = item.get("test_info")

            task_config = {
                "prompt": question,
                "solution": solution,
                "test": test,
                "test_info": test_info,
            }

            if verbose:
                print(f"\n{'#'*80}")
                print(f"Sample {idx + 1}/{len(dataset)} (Task ID: {task_id})")
                print(f"Problem: {question}")
                print(f"Test cases: {len(test_info) if test_info else 'N/A'}")
                print(f"{'#'*80}")

            success, reward, turns, tokens, p_tokens, conversation_history, error_msg = run_kodcode_episode(
                model=model,
                tokenizer=tokenizer,
                task_config=task_config,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stop_strings=stop_strings,
                max_turns=10,
                token_mapping=token_mapping if use_compression else None,
                equal_length=equal_length if use_compression else False,
                verbose=verbose,
            )

            if success:
                correct += 1

            total += 1
            total_turns += turns
            total_tokens += tokens
            total_prompt_tokens += p_tokens

            result = {
                "sample_id": idx,
                "task_id": task_id,
                "prompt": question,
                "test": test,
                "test_info": test_info,
                "correct": success,
                "turns": turns,
                "reward": reward,
                "total_tokens": tokens,
                "prompt_tokens": p_tokens,
                "error": error_msg if error_msg else None,
            }
            results.append(result)

            detailed_log = {
                "sample_id": idx,
                "task_id": task_id,
                "prompt": question,
                "test": test,
                "test_info": test_info,
                "reference_code": solution,
                "system_prompt": system_prompt,
                "conversation_history": conversation_history,
                "result": {
                    "success": success,
                    "reward": reward,
                    "turns": turns,
                    "total_tokens": tokens,
                    "prompt_tokens": p_tokens,
                },
                "error": error_msg if error_msg else None,
            }
            detailed_logs.append(detailed_log)

            if error_msg:
                errors.append({"sample_id": idx, "task_id": task_id, "prompt": question, "error": error_msg})

            if total % 10 == 0:
                print(
                    f"Progress: {total} samples, "
                    f"Accuracy: {correct/total*100:.2f}%, "
                    f"Avg turns: {total_turns/total:.1f}"
                )

        except Exception as e:
            error_msg = f"Error processing sample {idx}: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            errors.append({
                "sample_id": idx,
                "task_id": item.get("question_id", idx),
                "prompt": item.get("question", ""),
                "error": error_msg,
            })
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    avg_turns = total_turns / total if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_prompt_tokens = total_prompt_tokens / total if total > 0 else 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"detailed_logs_{prompt_type.lower()}_{timestamp}.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed logs saved to: {log_file}")

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
        "avg_turns": avg_turns,
        "avg_tokens_per_episode": avg_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "results": results,
        "errors": errors,
        "log_file": log_file,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to merged model (or LoRA-merged checkpoint)")
    parser.add_argument("--mapping", type=str, default="token_mapping.json",
                        help="Token mapping file")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate baseline (original model without compression)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of test samples")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-turn information")
    parser.add_argument("--output_dir", type=str, default="eval_output_kodcode",
                        help="Output directory for logs")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Max new tokens to generate per turn")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="Base model name (used for baseline and tokenizer)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Legacy arg (ignored, kept for CLI compatibility)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Legacy arg (ignored, kept for CLI compatibility)")
    parser.add_argument(
        "--equal_length",
        action="store_true",
        help=(
            "Pad replaced spans with tokenizer pad_token so the compressed "
            "prompt keeps a similar token length to the original."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🧪 Evaluating Compressed Model on KodCode (using transformers)")
    print("=" * 80)
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Mapping:             {args.mapping}")
    print(f"Samples:             {args.samples}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Backend:             transformers.generate")
    print(f"Equal length mode:   {args.equal_length}")
    print(f"Verbose:             {args.verbose}\n")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("📚 Loading KodCode test set...")
    question_ids = json.load(
        open('testset_split/kodcode_test_set_question_ids.json', 'r')
    )["test_set_question_ids"]
    try:
        dataset = load_dataset("KodCode/KodCode-V1")["train"]
        question_ids_set = set(question_ids)
        dataset = dataset.filter(lambda x: x['question_id'] in question_ids_set)
        print(f"✅ Loaded {len(dataset)} samples\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # Load token mapping
    # ------------------------------------------------------------------
    print("🔤 Loading token mapping...")
    try:
        token_mapping, special_tokens = load_token_mapping(args.mapping)
        print(f"✅ Loaded {len(token_mapping)} token mappings\n")
    except Exception as e:
        print(f"❌ Error loading token mapping: {e}")
        print(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print("🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

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

        # Add compression special tokens
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f"   Vocab size: {len(tokenizer)}\n")

        if args.equal_length and tokenizer.pad_token_id == tokenizer.eos_token_id:
            raise ValueError(
                "equal_length requires a dedicated pad_token distinct from eos_token, "
                "otherwise the attention mask cannot isolate supplemented padding tokens."
            )
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # Helper: load model
    # ------------------------------------------------------------------
    def load_model(model_path: str):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Baseline evaluation (optional)
    # ------------------------------------------------------------------
    baseline_results = None
    if args.baseline:
        print("📏 Loading baseline model with transformers...")
        try:
            baseline_model = load_model(args.model_name)

            baseline_results = evaluate_model(
                baseline_model,
                tokenizer,
                dataset,
                token_mapping,
                use_compression=False,
                max_samples=args.samples,
                verbose=args.verbose,
                output_dir=args.output_dir,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                equal_length=False,
            )

            print(f"\n✅ Baseline Results:")
            print(f"   Accuracy:         {baseline_results['accuracy']:.2f}%")
            print(f"   Correct:          {baseline_results['correct']}/{baseline_results['total']}")
            print(f"   Avg turns:        {baseline_results['avg_turns']:.1f}")
            print(f"   Avg prompt tokens:{baseline_results['avg_prompt_tokens']:.1f}")

            del baseline_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error evaluating baseline: {e}")
            print(traceback.format_exc())

    # ------------------------------------------------------------------
    # Compressed model evaluation
    # ------------------------------------------------------------------
    print(f"\n🤖 Loading compressed model with transformers...")
    try:
        compressed_model = load_model(args.checkpoint)

        compressed_results = evaluate_model(
            compressed_model,
            tokenizer,
            dataset,
            token_mapping,
            use_compression=True,
            max_samples=args.samples,
            verbose=args.verbose,
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            equal_length=args.equal_length,
        )

        del compressed_model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error evaluating compressed model: {e}")
        print(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # Print final results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)

    if baseline_results:
        print(f"\n📏 Baseline (Original Prompt):")
        print(f"   Accuracy:          {baseline_results['accuracy']:.2f}%")
        print(f"   Correct:           {baseline_results['correct']}/{baseline_results['total']}")
        print(f"   Avg turns:         {baseline_results['avg_turns']:.1f}")
        print(f"   Avg prompt tokens: {baseline_results['avg_prompt_tokens']:.1f}")
        if baseline_results['errors']:
            print(f"   ⚠️  Errors: {len(baseline_results['errors'])}")

    print(f"\n🎯 Compressed Model (LoRA + Special Tokens):")
    print(f"   Accuracy:          {compressed_results['accuracy']:.2f}%")
    print(f"   Correct:           {compressed_results['correct']}/{compressed_results['total']}")
    print(f"   Avg turns:         {compressed_results['avg_turns']:.1f}")
    print(f"   Avg prompt tokens: {compressed_results['avg_prompt_tokens']:.1f}")
    if compressed_results['errors']:
        print(f"   ⚠️  Errors: {len(compressed_results['errors'])}")

    if baseline_results:
        accuracy_gap = baseline_results['accuracy'] - compressed_results['accuracy']
        if baseline_results['avg_prompt_tokens'] > 0:
            token_reduction = (
                1 - compressed_results['avg_prompt_tokens'] / baseline_results['avg_prompt_tokens']
            ) * 100
            print(f"\n📉 Performance Gap:  {accuracy_gap:.2f}%")
            print(f"💡 Token reduction:  {token_reduction:.1f}%")
            print(
                f"💡 Trade-off: {token_reduction:.1f}% token reduction "
                f"for {accuracy_gap:.2f}% accuracy change"
            )

    # ------------------------------------------------------------------
    # Save summary results
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"evaluation_results_kodcode_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'baseline': baseline_results,
                'compressed': compressed_results,
                'timestamp': timestamp,
                'args': vars(args),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n💾 Summary results saved to: {output_file}")
    print(f"📁 All outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()