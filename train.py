import os
import torch
import torch.nn.functional as F
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import argparse
import glob
import re
from utils import *


# -----------------------------------------------------------------
# DDP device policy:
#   - student 始终使用 LOCAL_RANK 对应的 GPU（标准 DDP 绑卡方式）
#   - teacher 默认每个 rank 各自加载一份，使用该 rank 对应的 teacher GPU
# -----------------------------------------------------------------
local_rank  = int(os.environ.get("LOCAL_RANK", -1))
global_rank = int(os.environ.get("RANK", 0))
world_size  = int(os.environ.get("WORLD_SIZE", 1))
is_main     = global_rank == 0

if torch.cuda.is_available():
    if local_rank >= 0:
        student_device = f"cuda:{local_rank}"
    else:
        student_device = "cuda:0"
else:
    student_device = "cpu"

TEACHER_DEVICE = "cuda:0"


def load_token_mapping(mapping_file):
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    text_to_token = data['token_mapping']
    token_to_text = {v: k for k, v in text_to_token.items()}
    return data['special_tokens'], token_to_text


def load_jsonl_dataset(dataset_dir):
    train_files = glob.glob(os.path.join(dataset_dir, 'train*.jsonl'))
    val_files   = glob.glob(os.path.join(dataset_dir, 'validation*.jsonl'))

    def load_jsonl_file(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    train_data = []
    if train_files:
        for f in train_files:
            if is_main:
                print(f"   Loading {os.path.basename(f)}...")
            train_data.extend(load_jsonl_file(f))
    else:
        raise ValueError(f"No train*.jsonl files found in {dataset_dir}")

    val_data = []
    if val_files:
        for f in val_files:
            if is_main:
                print(f"   Loading {os.path.basename(f)}...")
            val_data.extend(load_jsonl_file(f))

    sample          = train_data[0]
    val_data        = val_data[:1]
    has_dual_format = ('original_messages' in sample and
                       'compressed_messages' in sample)

    if has_dual_format:
        if is_main:
            print("   Detected dual format (original + compressed)")
        train_dataset = Dataset.from_dict({
            'original_messages':   [item['original_messages']   for item in train_data],
            'compressed_messages': [item['compressed_messages'] for item in train_data],
        })
        val_dataset = Dataset.from_dict({
            'original_messages':   [item['original_messages']   for item in val_data],
            'compressed_messages': [item['compressed_messages'] for item in val_data],
        }) if val_data else None
    else:
        if is_main:
            print("   Detected single format (will be converted)")
        train_dataset = Dataset.from_dict({
            'system':   [item.get('system',   '') for item in train_data],
            'question': [item.get('question', '') for item in train_data],
            'answer':   [item.get('answer',   '') for item in train_data],
        })
        val_dataset = Dataset.from_dict({
            'system':   [item.get('system',   '') for item in val_data],
            'question': [item.get('question', '') for item in val_data],
            'answer':   [item.get('answer',   '') for item in val_data],
        }) if val_data else None

    return train_dataset, val_dataset


def restore_compressed_text(compressed_text, token_to_text):
    def replace_token(match):
        return token_to_text.get(match.group(0), match.group(0))
    return re.sub(r'<seg_\d+>', replace_token, compressed_text)


def preprocess_dual_conversation(
    examples,
    tokenizer,
    teacher_tokenizer,
    token_to_text,
    roles_for_kl=("system", "assistant"),
    max_length=4096,
    min_chunk_len=2,
):
    seg_token_ids = set(tokenizer.convert_tokens_to_ids(
        [t for t in tokenizer.additional_special_tokens if SEG_RE.fullmatch(t)]
    ))
    roles_for_kl = set(roles_for_kl)

    teacher_input_ids_list      = []
    student_input_ids_list      = []
    labels_list                 = []
    teacher_attention_mask_list = []
    student_attention_mask_list = []
    teacher_kl_mask_list        = []
    student_kl_mask_list        = []
    kept_original_messages      = []
    kept_compressed_messages    = []

    for i in range(len(examples["original_messages"])):
        original_msgs   = examples["original_messages"][i]
        compressed_msgs = examples["compressed_messages"][i]

        teacher_text = teacher_tokenizer.apply_chat_template(
            original_msgs,   tokenize=False, add_generation_prompt=False)
        student_text = tokenizer.apply_chat_template(
            compressed_msgs, tokenize=False, add_generation_prompt=False)

        if len(teacher_tokenizer(teacher_text, truncation=False,
                                  padding=False)["input_ids"]) > max_length:
            continue
        if len(tokenizer(student_text, truncation=False,
                          padding=False)["input_ids"]) > max_length:
            continue

        teacher_enc = teacher_tokenizer(
            teacher_text, truncation=True, max_length=max_length,
            padding=False, return_offsets_mapping=True)
        student_enc = tokenizer(
            student_text, truncation=True, max_length=max_length,
            padding=False, return_offsets_mapping=True)

        teacher_input_ids = teacher_enc["input_ids"]
        student_input_ids = student_enc["input_ids"]

        all_input_ids, all_labels = build_ce_labels_robust(
            tokenizer, compressed_msgs, max_length)

        teacher_kl_mask, student_kl_mask = build_shared_content_kl_masks(
            teacher_tokenizer, tokenizer,
            teacher_text, student_text,
            original_msgs, compressed_msgs,
            teacher_enc, student_enc,
            roles_for_kl=roles_for_kl,
            seg_token_ids=seg_token_ids,
            min_chunk_len=min_chunk_len,
        )

        if len(all_input_ids) != len(student_input_ids):
            if is_main:
                print("Warning: CE labels length mismatch, skipping.")
            continue

        if sum(teacher_kl_mask) != sum(student_kl_mask) or sum(teacher_kl_mask) == 8:
            with open(f'test_ori_rank{global_rank}.txt', 'w') as f:
                f.write(teacher_tokenizer.decode(
                    [x for x, m in zip(teacher_input_ids, teacher_kl_mask) if m == 1]))
                f.write('\n' + '=' * 100 + '\n' + teacher_text)
            with open(f'test_com_rank{global_rank}.txt', 'w') as f:
                f.write(tokenizer.decode(
                    [x for x, m in zip(student_input_ids, student_kl_mask) if m == 1]))
                f.write('\n' + '=' * 100 + '\n' + student_text)
            continue

        teacher_input_ids_list.append(teacher_input_ids)
        student_input_ids_list.append(student_input_ids)
        labels_list.append(all_labels)
        teacher_attention_mask_list.append([1] * len(teacher_input_ids))
        student_attention_mask_list.append([1] * len(student_input_ids))
        teacher_kl_mask_list.append(teacher_kl_mask)
        student_kl_mask_list.append(student_kl_mask)
        kept_original_messages.append(original_msgs)
        kept_compressed_messages.append(compressed_msgs)

    return {
        "teacher_text":           kept_original_messages,
        "student_text":           kept_compressed_messages,
        "teacher_input_ids":      teacher_input_ids_list,
        "student_input_ids":      student_input_ids_list,
        "labels":                 labels_list,
        "teacher_attention_mask": teacher_attention_mask_list,
        "student_attention_mask": student_attention_mask_list,
        "teacher_kl_mask":        teacher_kl_mask_list,
        "student_kl_mask":        student_kl_mask_list,
    }


def filter_dual_conversation_by_maxlen(
        example, tokenizer, teacher_tokenizer, max_length):
    original_msgs   = example["original_messages"]
    compressed_msgs = example["compressed_messages"]
    teacher_text = teacher_tokenizer.apply_chat_template(
        original_msgs,   tokenize=False, add_generation_prompt=False)
    student_text = tokenizer.apply_chat_template(
        compressed_msgs, tokenize=False, add_generation_prompt=False)
    if len(teacher_tokenizer(teacher_text,  truncation=False,
                              padding=False)["input_ids"]) > max_length:
        return False
    if len(tokenizer(student_text, truncation=False,
                     padding=False)["input_ids"]) > max_length:
        return False
    return True


class DualInputDataCollator:
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        teacher_input_ids      = [f['teacher_input_ids']      for f in features]
        student_input_ids      = [f['student_input_ids']      for f in features]
        labels                 = [f['labels']                 for f in features]
        teacher_kl_masks       = [f['teacher_kl_mask']        for f in features]
        student_kl_masks       = [f['student_kl_mask']        for f in features]
        teacher_attention_mask = [f['teacher_attention_mask'] for f in features]
        student_attention_mask = [f['student_attention_mask'] for f in features]

        max_teacher_len = max(len(ids) for ids in teacher_input_ids)
        max_student_len = max(len(ids) for ids in student_input_ids)
        pad_id          = self.tokenizer.pad_token_id

        batch_teacher_input_ids      = []
        batch_teacher_attention_mask = []
        batch_teacher_kl_mask        = []
        for ids, mask, kl in zip(teacher_input_ids,
                                  teacher_attention_mask, teacher_kl_masks):
            pad = max_teacher_len - len(ids)
            batch_teacher_input_ids.append(ids  + [pad_id] * pad)
            batch_teacher_attention_mask.append(mask + [0]  * pad)
            batch_teacher_kl_mask.append(kl   + [0]  * pad)

        batch_student_input_ids      = []
        batch_student_attention_mask = []
        batch_student_kl_mask        = []
        batch_labels                 = []
        for ids, mask, lbls, kl in zip(student_input_ids, student_attention_mask,
                                        labels, student_kl_masks):
            pad = max_student_len - len(ids)
            batch_student_input_ids.append(ids  + [pad_id] * pad)
            batch_student_attention_mask.append(mask + [0]  * pad)
            batch_labels.append(lbls + [-100]   * pad)
            batch_student_kl_mask.append(kl   + [0]  * pad)

        return {
            'teacher_input_ids':      torch.tensor(batch_teacher_input_ids,      dtype=torch.long),
            'teacher_attention_mask': torch.tensor(batch_teacher_attention_mask, dtype=torch.long),
            'teacher_kl_mask':        torch.tensor(batch_teacher_kl_mask,        dtype=torch.long),
            'input_ids':              torch.tensor(batch_student_input_ids,      dtype=torch.long),
            'attention_mask':         torch.tensor(batch_student_attention_mask, dtype=torch.long),
            'labels':                 torch.tensor(batch_labels,                 dtype=torch.long),
            'student_kl_mask':        torch.tensor(batch_student_kl_mask,        dtype=torch.long),
        }


def format_dataset_for_dual_conversation(dataset, token_to_text):
    def convert(example):
        original_system = restore_compressed_text(example['system'], token_to_text)
        return {
            'original_messages': [
                {"role": "system",    "content": original_system},
                {"role": "user",      "content": example['question']},
                {"role": "assistant", "content": example['answer']},
            ],
            'compressed_messages': [
                {"role": "system",    "content": example['system']},
                {"role": "user",      "content": example['question']},
                {"role": "assistant", "content": example['answer']},
            ],
        }
    return dataset.map(convert, remove_columns=dataset.column_names)


def _align_and_pool(s_sliced, t_sliced):
    """如果序列长度不同，用 avg-pool 对齐到较短的那个。"""
    if s_sliced.size(0) == t_sliced.size(0):
        return s_sliced, t_sliced
    min_len = min(s_sliced.size(0), t_sliced.size(0))
    if s_sliced.size(0) > min_len:
        chunks   = torch.chunk(s_sliced, min_len, dim=0)
        s_sliced = torch.stack([c.mean(0) for c in chunks])
    if t_sliced.size(0) > min_len:
        chunks   = torch.chunk(t_sliced, min_len, dim=0)
        t_sliced = torch.stack([c.mean(0) for c in chunks])
    return s_sliced, t_sliced


class KLDistillationTrainer(Trainer):
    """
    每个 rank 都持有 teacher，并对自己的 local batch 计算 KL。
    这样 teacher/student 始终是同一批样本，避免跨 rank 广播造成错配。
    """
    def __init__(self, teacher_model=None, kl_weight=0.5, temperature=2.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model   # use_kl_distillation 时每个 rank 非 None
        self.kl_weight     = kl_weight
        self.temperature   = temperature

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_input_ids      = inputs.pop('teacher_input_ids',      None)
        teacher_attention_mask = inputs.pop('teacher_attention_mask', None)
        teacher_kl_mask        = inputs.pop('teacher_kl_mask',        None)
        student_kl_mask        = inputs.pop('student_kl_mask',        None)

        # ---- student forward ----
        outputs        = model(**inputs)
        student_logits = outputs.logits    # on student_device
        lm_loss        = outputs.loss
        sdev           = student_logits.device

        if self.kl_weight == 0.0 or teacher_input_ids is None:
            return (lm_loss, outputs) if return_outputs else lm_loss

        # ---- teacher forward（local batch）----
        teacher_sliced_list = self._get_teacher_slices(
            teacher_input_ids,
            teacher_attention_mask,
            teacher_kl_mask,
            sdev,
        )
        # teacher_sliced_list: List[Tensor]，每个元素 shape [n_masked_i, V_t] on sdev

        # ---- student 切片（在 sdev 上，不需要跨设备）----
        student_kl_mask = student_kl_mask.to(sdev)
        kl_loss = self._compute_kl(
            student_logits, student_kl_mask,
            teacher_sliced_list, sdev,
        )

        total_loss = (1 - self.kl_weight) * lm_loss + self.kl_weight * kl_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'lm_loss':    lm_loss.item(),
                'kl_loss':    kl_loss.item(),
                'total_loss': total_loss.item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    def _get_teacher_slices(self,
                            teacher_input_ids,
                            teacher_attention_mask,
                            teacher_kl_mask,
                            target_device):
        """
        每个 rank 在本地跑 teacher forward，并按 mask 切片。
        返回：List[Tensor]，每个 [n_masked_i, V_teacher]，在 target_device 上。
        """
        batch_size = teacher_input_ids.size(0)
        if self.teacher_model is None:
            return [None] * batch_size

        t_ids  = teacher_input_ids.to(TEACHER_DEVICE)
        t_mask = teacher_attention_mask.to(TEACHER_DEVICE)
        t_kl   = teacher_kl_mask.to(TEACHER_DEVICE)

        with torch.no_grad():
            t_logits = self.teacher_model(
                input_ids=t_ids,
                attention_mask=t_mask,
            ).logits

        result = []
        for b in range(batch_size):
            seq_len = min(t_logits[b].size(0), t_kl[b].size(0))
            mask_b = t_kl[b][:seq_len].bool()
            sliced = t_logits[b][:seq_len][mask_b]
            result.append(sliced.to(target_device, dtype=torch.float16))
        return result

    # ------------------------------------------------------------------
    def _compute_kl(self, student_logits, student_kl_mask,
                    teacher_sliced_list, sdev):
        batch_size = student_logits.size(0)
        kl_losses  = []

        for b in range(batch_size):
            s_sliced = student_logits[b][student_kl_mask[b].bool()]
            t_sliced = teacher_sliced_list[b]

            if t_sliced is None or s_sliced.size(0) == 0 or t_sliced.size(0) == 0:
                continue

            s_sliced, t_sliced = _align_and_pool(s_sliced, t_sliced)

            s_t = s_sliced / self.temperature
            t_t = t_sliced / self.temperature

            if t_t.size(-1) != s_t.size(-1):
                minv = min(t_t.size(-1), s_t.size(-1))
                t_t  = t_t[..., :minv]
                s_t  = s_t[..., :minv]

            kl_losses.append(F.kl_div(
                F.log_softmax(s_t, dim=-1),
                F.softmax(t_t,    dim=-1),
                reduction='batchmean',
            ))

        if not kl_losses:
            return torch.tensor(0.0, device=sdev)
        return torch.stack(kl_losses).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name",        type=str,   default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model_name",                type=str,   default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_dir",               type=str,   default="sft_dataset")
    parser.add_argument("--mapping",                   type=str,   default="token_mapping.json")
    parser.add_argument("--output_dir",                type=str,   default="lora_sft_output")
    parser.add_argument("--epochs",                    type=int,   default=3)
    parser.add_argument("--batch_size",                type=int,   default=2)
    parser.add_argument("--learning_rate",             type=float, default=1e-4)
    parser.add_argument("--lora_r",                    type=int,   default=8)
    parser.add_argument("--lora_alpha",                type=int,   default=16)
    parser.add_argument("--max_length",                type=int,   default=4096)
    parser.add_argument("--use_kl_distillation",       action="store_true")
    parser.add_argument("--kl_weight",                 type=float, default=0.5)
    parser.add_argument("--temperature",               type=float, default=2.0)
    parser.add_argument("--train_new_embeddings_only", action="store_true")
    parser.add_argument("--teacher_device",            type=str,   default=None)
    parser.add_argument("--teacher_student_ratio",     type=str,   default="2:2")
    args = parser.parse_args()

    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    global TEACHER_DEVICE
    if args.teacher_device is not None:
        TEACHER_DEVICE = args.teacher_device
    elif torch.cuda.is_available():
        n = torch.cuda.device_count()
        if args.teacher_student_ratio == "2:2" and local_rank >= 0 and n >= (2 * world_size):
            teacher_idx = local_rank + world_size
        elif args.teacher_student_ratio == "1:3":
            raise ValueError(
                "teacher_student_ratio=1:3 requires shared teacher service, "
                "which is not enabled in this script. Use 2:2 or set --teacher_device per rank launch."
            )
        else:
            teacher_idx = local_rank if local_rank >= 0 else 0
        TEACHER_DEVICE = f"cuda:{teacher_idx}"
    else:
        TEACHER_DEVICE = "cpu"

    if is_main:
        print("=" * 80)
        print("🚀 KL Distillation LoRA Training")
        print()
        print(f"  world_size={world_size}, local_rank={local_rank}, rank={global_rank}")
        print(f"  student_device(按LOCAL_RANK绑定) = {student_device}")
        print(f"  teacher_student_ratio            = {args.teacher_student_ratio}")
        print(f"  teacher_device(this rank)        = {TEACHER_DEVICE}")
        print()
        print("  每步流程：")
        print(f"    each rank: teacher fwd on {TEACHER_DEVICE} for local batch")
        print("    all:    student fwd + KL loss on 各自 student device")
        print("=" * 80)
        print(f"  model:      {args.model_name}")
        print(f"  max_length: {args.max_length}")
        print(f"  batch_size: {args.batch_size}\n")

    # ------------------------------------------------------------------
    # ALL RANKS: tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True)
    if 'Qwen' in args.model_name:
        custom_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        tokenizer.chat_template = custom_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ------------------------------------------------------------------
    # ALL RANKS: special tokens
    # ------------------------------------------------------------------
    special_tokens, token_to_text = load_token_mapping(args.mapping)
    num_added = tokenizer.add_special_tokens(
        {'additional_special_tokens': special_tokens})
    original_vocab_size = len(tokenizer) - num_added

    if is_main:
        print(f"   Added {num_added} tokens  "
              f"(vocab: {original_vocab_size} → {len(tokenizer)})\n")

    # ------------------------------------------------------------------
    # ALL RANKS: teacher model（每个 rank 一份，确保与 local batch 对齐）
    # ------------------------------------------------------------------
    teacher_model     = None
    teacher_tokenizer = None

    if args.use_kl_distillation:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        if 'Qwen' in args.model_name:
            teacher_tokenizer.chat_template = custom_template
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

        print(f"👨‍🏫 [rank {global_rank}] Loading teacher on {TEACHER_DEVICE} ...")
        teacher_device_idx = 0 if TEACHER_DEVICE == "cpu" else int(TEACHER_DEVICE.split(":")[-1])
        teacher_dtype = torch.float16 if TEACHER_DEVICE != "cpu" else torch.float32
        teacher_attn_impl = "flash_attention_2" if TEACHER_DEVICE != "cpu" else None
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_name,
            trust_remote_code=True,
            torch_dtype=teacher_dtype,
            attn_implementation=teacher_attn_impl,
            local_files_only=True,
        )
        teacher_model.to(TEACHER_DEVICE)
        teacher_model.resize_token_embeddings(original_vocab_size)
        teacher_model.config.use_cache = False
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(f"   Teacher ready on {TEACHER_DEVICE}\n")

    # ------------------------------------------------------------------
    # ALL RANKS: student model
    # ------------------------------------------------------------------
    if is_main:
        print(f"🤖 Loading student (rank {global_rank}, local_rank {local_rank} → {student_device})...")

    student_device_idx = 0 if local_rank < 0 else local_rank
    student_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    student_attn_impl = "flash_attention_2" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=student_dtype,
        attn_implementation=student_attn_impl,
        local_files_only=True,
    )
    model.to(student_device)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.use_cache = False

    if is_main:
        print(f"   Student vocab: {len(tokenizer)}\n")

    # ------------------------------------------------------------------
    # ALL RANKS: LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.train_new_embeddings_only:
        emb = model.get_input_embeddings()
        emb.weight.requires_grad = False
        emb.weight[original_vocab_size:].requires_grad = True
        lm_head = model.get_output_embeddings()
        lm_head.weight.requires_grad = False
        lm_head.weight[original_vocab_size:].requires_grad = True
        if is_main:
            print(f"   ✅ Only NEW embeddings trainable ({num_added} tokens)")
    else:
        if is_main:
            print("   ✅ ALL embeddings trainable")

    if is_main:
        model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # ALL RANKS: dataset
    # ------------------------------------------------------------------
    if is_main:
        print("\n📊 Loading dataset...")

    train_dataset, eval_dataset = load_jsonl_dataset(args.dataset_dir)

    if 'system' in train_dataset.column_names:
        if is_main:
            print("   Converting single → dual format...")
        train_dataset = format_dataset_for_dual_conversation(
            train_dataset, token_to_text)
        if eval_dataset is not None:
            eval_dataset = format_dataset_for_dual_conversation(
                eval_dataset, token_to_text)

    if eval_dataset is None:
        if is_main:
            print("   No validation file — splitting 90/10...")
        split         = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset  = split['test']

    if is_main:
        print(f"   Train: {len(train_dataset)}, Eval: {len(eval_dataset)}\n")

    # ------------------------------------------------------------------
    # ALL RANKS: filter + preprocess
    # ------------------------------------------------------------------
    if is_main:
        print("🧹 Filtering + preprocessing...")

    kl_teacher_tokenizer = teacher_tokenizer if teacher_tokenizer is not None else tokenizer

    train_dataset = train_dataset.filter(
        lambda x: filter_dual_conversation_by_maxlen(
            x, tokenizer, kl_teacher_tokenizer, max_length=args.max_length),
        desc="Filter train",
    )
    eval_dataset = eval_dataset.filter(
        lambda x: filter_dual_conversation_by_maxlen(
            x, tokenizer, kl_teacher_tokenizer, max_length=args.max_length),
        desc="Filter eval",
    )
    train_dataset = train_dataset.map(
        lambda x: preprocess_dual_conversation(
            x, tokenizer, kl_teacher_tokenizer, token_to_text,
            max_length=args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing train",
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_dual_conversation(
            x, tokenizer, kl_teacher_tokenizer, token_to_text,
            max_length=args.max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Processing eval",
    )

    if is_main:
        print(f"   After filter — Train: {len(train_dataset)}, "
              f"Eval: {len(eval_dataset)}")
        print('teacher kl mask sum:', sum(train_dataset[0]['teacher_kl_mask']))
        print('student kl mask sum:', sum(train_dataset[0]['student_kl_mask']))
        print_kl_mask_tokens(train_dataset, tokenizer, num_samples=1)

    data_collator = DualInputDataCollator(
        tokenizer=tokenizer, max_length=args.max_length)

    # ------------------------------------------------------------------
    # ALL RANKS: TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=3,
        fp16=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )

    # ------------------------------------------------------------------
    # ALL RANKS: trainer + train
    # ------------------------------------------------------------------
    trainer = KLDistillationTrainer(
        model=model,
        teacher_model=teacher_model,     # rank 0: 真实模型; rank 1: None
        kl_weight=args.kl_weight if args.use_kl_distillation else 0.0,
        temperature=args.temperature,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if is_main:
        print("\n" + "=" * 80)
        print("🏋️  Starting training...")
        print("=" * 80 + "\n")

    trainer.train()

    # ------------------------------------------------------------------
    # RANK 0 ONLY: save
    # ------------------------------------------------------------------
    if is_main:
        print("\n💾 Saving...")
        final_path = os.path.join(args.output_dir, "final_checkpoint")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"✅ Done → {final_path}")
        # print("\n显存分布：")
        # print("  cuda:0  ~16GB  teacher only（frozen）")
        # print("  cuda:1  ~40GB  student rank 0 + optimizer + activations")
        # print("  cuda:2  ~40GB  student rank 1 + optimizer + activations")


if __name__ == "__main__":
    main()
