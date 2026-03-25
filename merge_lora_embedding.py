import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import json
import os


def load_token_mapping(mapping_file):
    """加载token映射"""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['token_mapping'], data['special_tokens']


def merge_lora_with_expanded_embeddings(
    base_model_name: str,
    lora_checkpoint: str,
    token_mapping_file: str,
    output_dir: str,
    device: str = "cuda"
):
    """
    合并LoRA模型和扩展的Embedding
    
    Args:
        base_model_name: 基础模型名称
        lora_checkpoint: LoRA checkpoint路径
        token_mapping_file: token映射文件
        output_dir: 输出目录
        device: 设备
    """
    print("="*80)
    print("🔧 Merging LoRA with Expanded Embeddings")
    print("="*80)
    
    # 1. 加载token映射
    print(f"\n📋 Loading token mapping from {token_mapping_file}...")
    token_mapping, special_tokens = load_token_mapping(token_mapping_file)
    print(f"   Found {len(special_tokens)} special tokens")
    
    # 2. 加载tokenizer并添加特殊tokens
    print(f"\n🔤 Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    original_vocab_size = len(tokenizer)
    print(f"   Original vocab size: {original_vocab_size}")
    
    # 添加特殊tokens
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    new_vocab_size = len(tokenizer)
    print(f"   New vocab size: {new_vocab_size}")
    print(f"   Added {new_vocab_size - original_vocab_size} tokens")
    
    # 3. 加载基础模型
    print(f"\n🤖 Loading base model from {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    print(f"   Model loaded successfully")
    
    # 4. 扩展embedding层
    print(f"\n📊 Resizing token embeddings...")
    print(f"   Before: {base_model.get_input_embeddings().weight.shape}")
    base_model.resize_token_embeddings(new_vocab_size)
    print(f"   After: {base_model.get_input_embeddings().weight.shape}")
    
    # 5. 加载LoRA权重
    print(f"\n🔗 Loading LoRA weights from {lora_checkpoint}...")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint,
        device_map=device,
        is_trainable=False
    )
    print(f"   LoRA loaded successfully")
    
    # 打印LoRA配置信息
    print(f"\n📝 LoRA Configuration:")
    print(f"   Target modules: {model_with_lora.peft_config['default'].target_modules}")
    print(f"   Rank (r): {model_with_lora.peft_config['default'].r}")
    print(f"   Alpha: {model_with_lora.peft_config['default'].lora_alpha}")
    print(f"   Dropout: {model_with_lora.peft_config['default'].lora_dropout}")
    
    # 6. 合并LoRA权重到基础模型
    print(f"\n🔀 Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()
    print(f"   LoRA merged successfully")
    
    # 7. 验证embedding层
    print(f"\n✅ Verifying merged model:")
    print(f"   Input embeddings shape: {merged_model.get_input_embeddings().weight.shape}")
    print(f"   Output embeddings shape: {merged_model.get_output_embeddings().weight.shape}")
    print(f"   Model dtype: {merged_model.dtype}")
    
    # 8. 保存合并后的模型
    print(f"\n💾 Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,  # 使用safetensors格式
        max_shard_size="5GB"
    )
    print(f"   Model saved")
    
    # 保存tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"   Tokenizer saved")
    
    # 9. 保存额外的配置信息
    lora_config = model_with_lora.peft_config['default']
    merge_info = {
        "base_model": base_model_name,
        "lora_checkpoint": lora_checkpoint,
        "token_mapping_file": token_mapping_file,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "added_tokens": new_vocab_size - original_vocab_size,
        "special_tokens": special_tokens,
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules) if isinstance(lora_config.target_modules, set) else lora_config.target_modules,
            "lora_dropout": lora_config.lora_dropout,
        }
    }
    
    merge_info_path = os.path.join(output_dir, "merge_info.json")
    with open(merge_info_path, 'w', encoding='utf-8') as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)
    print(f"   Merge info saved to {merge_info_path}")
    
    print(f"\n✅ Merge completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    return merged_model, tokenizer


def verify_merged_model(model_path: str):
    """
    验证合并后的模型
    
    Args:
        model_path: 合并后的模型路径
    """
    print("\n" + "="*80)
    print("🔍 Verifying Merged Model")
    print("="*80)
    
    # 加载模型
    print(f"\n📂 Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
    
    # 检查是否还有LoRA层
    has_lora = any('lora' in name.lower() for name, _ in model.named_parameters())
    print(f"   Has LoRA layers: {has_lora}")
    
    if has_lora:
        print("   ⚠️  Warning: Model still contains LoRA layers!")
    else:
        print("   ✅ Model is fully merged (no LoRA layers)")
    
    # 读取merge info
    merge_info_path = os.path.join(model_path, "merge_info.json")
    if os.path.exists(merge_info_path):
        with open(merge_info_path, 'r') as f:
            merge_info = json.load(f)
        print(f"\n📋 Merge Information:")
        print(f"   Base model: {merge_info['base_model']}")
        print(f"   Original vocab: {merge_info['original_vocab_size']}")
        print(f"   New vocab: {merge_info['new_vocab_size']}")
        print(f"   Added tokens: {merge_info['added_tokens']}")
    
    # 测试生成
    print(f"\n🧪 Testing generation...")
    test_prompt = "def fibonacci(n):"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Prompt: {test_prompt}")
    print(f"   Generated: {generated_text[:200]}...")
    print(f"\n✅ Verification completed!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA with expanded embeddings")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--token_mapping",
        type=str,
        required=True,
        help="Path to token mapping JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the merged model after merging"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # 执行合并
    merged_model, tokenizer = merge_lora_with_expanded_embeddings(
        base_model_name=args.base_model,
        lora_checkpoint=args.lora_checkpoint,
        token_mapping_file=args.token_mapping,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 可选：验证合并后的模型
    if args.verify:
        verify_merged_model(args.output_dir)
    
    print("\n" + "="*80)
    print("🎉 All done!")
    print("="*80)
    print(f"\nYou can now use the merged model:")
    print(f"  Model path: {args.output_dir}")
    print(f"\nExample usage:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()