# LAR + Distillation (TriviaQA) Architecture and Repro Guide

This document reorganizes the codebase logic around three things:

1. System architecture.
2. Responsibilities of each component.
3. The exact LAR + distillation flow, step by step, with script/function mapping.

---

## 1) System Architecture (Layered View)

### Layer A: Data and Segment Mining (LAR preparation)

- `identify_segments.py`
  - Mines reusable high-frequency, low-entropy segments from trajectory JSONL.
  - Supports both text n-grams and optional HTML tag patterns.
  - Removes redundant segments by substring and overlap filtering.

- `create_token_mapping.py`
  - Converts selected segments into compact symbolic tokens like `<seg_0>`, `<seg_1>`.
  - Exports mapping and `special_tokens` used later by tokenizer/model.

### Layer B: Dual-View Dataset Construction

- `build_sft_dataset_v1.py`
  - Builds paired supervision views:
    - `original_messages` (teacher-facing original text)
    - `compressed_messages` (student-facing LAR-compressed text)
  - Supports toggles:
    - compress system prompt
    - compress conversations

### Layer C: Model Training (LoRA + CE + KL)

- `train.py`
  - Loads tokenizer and adds `<seg_n>` special tokens.
  - Loads student model and wraps with LoRA.
  - Optionally loads frozen teacher model for distillation.
  - Converts dual conversations into:
    - student CE labels
    - teacher/student KL masks for shared content
  - Trains with mixed objective:
    - LM CE loss (student)
    - KL distillation loss (teacher -> student)

- `utils.py`
  - Carries the key token/label/mask logic for distillation-safe alignment.

### Layer D: Checkpoint Finalization

- `merge_lora_embedding.py`
  - Merges LoRA adapters into base model.
  - Preserves expanded embedding for special segment tokens.

### Layer E: Task Evaluation

- `evaluate_triviaqa.py`
  - Runs multi-turn TriviaQA ReAct episodes (`<search>`, `<answer>`).
  - Compares baseline vs compressed prompt behavior.
  - Supports equal-length control mode.
  - Outputs detailed logs and aggregate metrics.

---

## 2) Module Responsibilities (Function-Level Map)

### `identify_segments.py`

- `extract_ngrams(...)`: sentence-bounded n-gram extraction.
- `analyze_segment_entropy(...)`: next-token entropy scoring.
- `filter_redundant_segments(...)`: removes overlapping/subsumed segments.
- `identify_all_segments(...)`: main segment discovery pipeline.
- `analyze_compression_potential(...)`: rough compression gain estimate.

### `build_sft_dataset_v1.py`

- `load_token_mapping(...)`: reads mapping metadata.
- `replace_text_with_tokens(...)`: applies segment replacement.
- `from_to_role(...)`: normalizes role labels.
- `build_dual_messages(...)`: creates original/compressed message pairs.
- `build_sft_dataset_dual(...)`: full dataset transform + split + export.

### `train.py`

- `load_jsonl_dataset(...)`: reads `train*.jsonl` and `validation*.jsonl`.
- `format_dataset_for_dual_conversation(...)`: fallback single->dual conversion.
- `preprocess_dual_conversation(...)`: tokenization + CE labels + KL masks.
- `DualInputDataCollator`: packs teacher/student tensors into one batch.
- `KLDistillationTrainer.compute_loss(...)`: CE/KL joint loss logic.

### `utils.py`

- `build_ce_labels_robust(...)`: assistant-token CE supervision.
- `build_shared_content_kl_masks(...)`: KL masking on shared content only.
- `_align_and_pool(...)` (called by trainer): sequence-length alignment before KL.

### `evaluate_triviaqa.py`

- `SimpleTriviaQAEnv`: search/answer interaction and reward.
- `run_triviaqa_episode(...)`: one sample multi-turn rollout.
- `evaluate_model(...)`: batch evaluation and metric aggregation.

---

## 3) LAR + Distillation Logic (Step-by-Step, With Code References)

This section is the core pipeline you asked to sort out.

### Step 0: Input data shape

Expected trajectory JSONL items include:

- `system` (instruction text)
- `conversations` (turn list, usually `from` + `value`)

Handled in `build_sft_dataset_v1.py`.

### Step 1: Discover compressible patterns (LAR candidate mining)

Code: `identify_segments.py`

Process:

1. Extract all candidate n-grams from selected fields.
2. Keep those above `min_freq`.
3. Score by entropy; keep low-entropy segments.
4. Remove redundant candidates.
5. Save to `segments.json`.

Purpose:

- Find stable repeated text chunks that can be replaced by latent symbolic tokens.

### Step 2: Build latent token mapping

Code: `create_token_mapping.py`

Process:

1. Assign each selected segment to `<seg_n>`.
2. Save:
   - `special_tokens`
   - mapping dictionary (text <-> token forms)

Purpose:

- Define the LAR vocabulary extension used by tokenizer/model.

### Step 3: Build teacher/student dual training views

Code: `build_sft_dataset_v1.py`

Process:

1. For each sample, construct:
   - `original_messages`: uncompressed text
   - `compressed_messages`: text with `<seg_n>` substitutions
2. Split train/validation.
3. Save `train.jsonl`, `validation.jsonl`.

Purpose:

- Preserve semantic equivalence between two views so distillation can transfer knowledge from original to compressed representation.

### Step 4: Extend tokenizer and model embeddings

Code: `train.py` (`main`)

Process:

1. Load tokenizer from `--model_name`.
2. Add `special_tokens` from mapping.
3. Resize student embeddings to include new tokens.
4. Load teacher if `--use_kl_distillation`.

Purpose:

- Ensure `<seg_n>` tokens are represented in student model vocabulary.

### Step 5: Build CE supervision on student side

Code path:

- `train.py` -> `preprocess_dual_conversation(...)`
- `utils.py` -> `build_ce_labels_robust(...)`

Process:

1. Render `compressed_messages` with chat template.
2. Mark assistant content tokens as labels.
3. Non-target tokens are `-100`.

Purpose:

- Train student to generate correct assistant outputs in compressed context.

### Step 6: Build KL supervision masks (shared content only)

Code path:

- `train.py` -> `preprocess_dual_conversation(...)`
- `utils.py` -> `build_shared_content_kl_masks(...)`

Process:

1. Render original (teacher) and compressed (student) texts.
2. Detect shared literal chunks.
3. Create `teacher_kl_mask` and `student_kl_mask`.
4. Exclude `<seg_n>` and special tokens from KL area.

Purpose:

- Distill only on aligned semantic text, avoiding forced one-to-one matching on compressed placeholders.

### Step 7: Forward pass and KL computation

Code path:

- `train.py` -> `KLDistillationTrainer.compute_loss(...)`

Process per batch:

1. Student forward -> `lm_loss`.
2. Teacher forward (frozen, no grad) on local batch.
3. Slice logits by KL masks.
4. If lengths differ, align with `_align_and_pool(...)`.
5. Apply temperature scaling and KL divergence.

Final objective:

- `total_loss = (1 - kl_weight) * lm_loss + kl_weight * kl_loss`

Purpose:

- Keep generation quality (CE) while transferring teacher distributional knowledge under compressed prompts (KL).

### Step 8: Save and deploy

Code: `merge_lora_embedding.py`

Process:

1. Merge LoRA into base model weights.
2. Save merged checkpoint with token-extended embedding.

### Step 9: Evaluate TriviaQA behavior

Code: `evaluate_triviaqa.py`

Process:

1. Load TriviaQA validation split.
2. Run tool-augmented multi-turn QA episodes.
3. Compare original vs compressed prompts.
4. Collect accuracy, turns, token usage, logs.

---

## 4) Minimal Reproducible Path (LAR + Distillation + TriviaQA)

Assume working directory is repository root.
Replace placeholders with your local paths.

### 4.1 Prepare directories

- Input trajectory JSONL: `dataset/triviaqa_qwen/triviaqa_qwen_original.jsonl`
- Segment output: `dataset/triviaqa_qwen/segments.json`
- Mapping output: `dataset/triviaqa_qwen/token_mapping.json`
- SFT output dir: `dataset/triviaqa_qwen/`
- Training output dir: `weight/triviaqa_qwen/full`
- Merged model dir: `weight/triviaqa_qwen/merged_model`

### 4.2 Run LAR data pipeline

```bash
python identify_segments.py \
  --input_jsonl dataset/triviaqa_qwen/triviaqa_qwen_original.jsonl \
  --output dataset/triviaqa_qwen/segments.json \
  --include_conversations \
  --n_min 2 \
  --n_max 6 \
  --max_segments 100 \
  --min_freq 1000
```

```bash
python create_token_mapping.py \
  --segments dataset/triviaqa_qwen/segments.json \
  --output dataset/triviaqa_qwen/token_mapping.json
```

```bash
python build_sft_dataset_v1.py \
  --input_jsonl dataset/triviaqa_qwen/triviaqa_qwen_original.jsonl \
  --mapping dataset/triviaqa_qwen/token_mapping.json \
  --output_dir dataset/triviaqa_qwen \
  --compress_system \
  --compress_conversations \
  --split_ratio 0.99
```

### 4.3 Train with distillation

```bash
python train.py \
  --dataset_dir dataset/triviaqa_qwen \
  --mapping dataset/triviaqa_qwen/token_mapping.json \
  --output_dir weight/triviaqa_qwen/full \
  --model_name Qwen/Qwen3-8B \
  --teacher_model_name Qwen/Qwen3-8B \
  --epochs 3 \
  --batch_size 2 \
  --use_kl_distillation \
  --kl_weight 1.0 \
  --train_new_embeddings_only
```

### 4.4 Merge LoRA and expanded embeddings

```bash
python merge_lora_embedding.py \
  --base_model Qwen/Qwen3-8B \
  --lora_checkpoint weight/triviaqa_qwen/full/final_checkpoint \
  --output_dir weight/triviaqa_qwen/merged_model \
  --token_mapping dataset/triviaqa_qwen/token_mapping.json
```

### 4.5 Evaluate on TriviaQA

```bash
python evaluate_triviaqa.py \
  --checkpoint weight/triviaqa_qwen/merged_model \
  --model_name Qwen/Qwen3-8B \
  --mapping dataset/triviaqa_qwen/token_mapping.json \
  --dataset triviaqa \
  --samples 100 \
  --output_dir eval_output/triviaqa_qwen \
  --baseline
```

Optional controlled comparison:

- add `--equal_length` to evaluate compressed prompt with length control.

---

## 5) Key Flags and Their Effects

- `--compress_system`: apply LAR token substitution to system prompt.
- `--compress_conversations`: apply substitution to dialogue turns too.
- `--use_kl_distillation`: enable teacher-student KL branch.
- `--kl_weight`: CE/KL mixing ratio in final loss.
- `--train_new_embeddings_only`: focus training on newly introduced segment-token embeddings.
- `--equal_length` (evaluation): pad for token-length-controlled comparisons.

---

## 6) Practical Caveats

- Scripts are research-oriented and path-dependent; adjust dataset/model paths.
- Some evaluation flows require external/local services (retrieval endpoint in TriviaQA eval).
- Ensure tokenizer/model availability and GPU memory match chosen base model.

---

## 7) Meeting Core Points (LAR x OpenClaw)

This section captures the meeting-aligned core story and implementation priorities.

### 7.1 Four-step core pipeline

1. `identify_segments.py`
   - Use n-gram frequency + low-entropy filtering (TF-IDF-like intuition) on trajectory text.
   - Mine repeated long prompt fragments as compression candidates.

2. Build `token_mapping`
   - Map selected fragments to special tokens (`<seg_n>`).
   - Mapping size (vocab extension size) is tunable for compression-performance tradeoff.

3. `build_sft_dataset_v1.py`
   - Apply mapping to raw trajectories.
   - Produce dual-view SFT data:
     - `original_messages` (teacher view)
     - `compressed_messages` (student view)

4. `train.py` + `merge_lora_embedding.py`
   - Teacher/student training with KL distillation.
   - Merge LoRA + embedding into deployable checkpoint.

### 7.2 OpenClaw + TriviaQA concrete recommendations

- Task setup:
  - Use TriviaQA single-turn setting.
  - Evaluation size fixed at 5000 validation samples.

- Prompt compression strategy:
  - OpenClaw system prompts are longer and more repetitive than many other settings.
  - Increase n-gram range and allow larger mapping vocabulary to improve compression ratio.

- Segment quality control:
  - Add manual posterior review over auto-mined segments.
  - Remove segments that:
    - carry critical semantic commitments,
    - are discontinuous or brittle under minor template changes.

### 7.3 Inference and deployment plan

1. Train and merge with `merge_lora_embedding.py`.
2. Serve merged model via local vLLM API.
3. Point OpenClaw evaluation pipeline to the new API endpoint.
4. Run side-by-side evaluation against Vanilla baseline.

### 7.4 Engineering adaptation priorities after code handoff

- Use AI-assisted code reading to quickly trace data format and runtime path.
- Adapt `run_episode`-related interface so compressed prompt/model API works in OpenClaw runtime.
- Keep CE+KL core training logic unchanged; focus modifications on:
  - data conversion,
  - token mapping definition,
  - runtime API wiring.
- Execute training on provided GPUs and keep full config logs for reproducibility.

### 7.5 Suggested starting parameters (OpenClaw TriviaQA)

Below is a practical starting grid for segment mining and mapping size.

#### A) Segment mining (`identify_segments.py`)

| Profile | `n_min` | `n_max` | `min_freq` | `max_segments` | `max_entropy` | `include_conversations` | Notes |
|---|---:|---:|---:|---:|---:|---|---|
| Conservative | 3 | 8 | 800 | 256 | 2.0 | false | Prioritize robust fixed system text; lowest semantic risk |
| Balanced (recommended) | 4 | 12 | 500 | 512 | 2.5 | true | Good tradeoff for OpenClaw long prompt + repeated tool instruction |
| Aggressive | 5 | 16 | 200 | 1024 | 3.0 | true | Maximum compression attempt; requires stronger manual filtering |

Recommended first run:

- Start with **Balanced**, then manually inspect top 100 segments.
- If compression is insufficient, increase `max_segments` first, then widen `n_max`.
- If quality drops, tighten `max_entropy` and raise `min_freq`.

#### B) Token mapping size (`create_token_mapping.py`)

| Profile | Effective mapping size | Typical usage |
|---|---:|---|
| Conservative | 64-128 | Replace only system prompt core boilerplate |
| Balanced | 256-512 | System prompt + stable tool usage instructions |
| Aggressive | 800-1200 | Include repeated conversation-side templates and long wrappers |

Implementation note:

- Keep a reserved buffer of special tokens for future prompt variants (for example +32).
- Prefer adding segments in descending token-savings order (longer and higher coverage first).

#### C) Manual posterior filtering checklist

Remove segments if any of the following is true:

- Segment contains dynamic slots (question text, retrieved passages, IDs, timestamps).
- Segment encodes critical task semantics that should remain explicit.
- Segment is discontinuous or depends on fragile formatting boundaries.
- Segment appears in too many near-duplicate variants (low canonical stability).

Keep segments if:

- Coverage is high and stable across the 5000-sample subset.
- String is template-like and instruction boilerplate.
- Token savings per replacement is large and consistent.

#### D) Ready-to-run command templates

Balanced profile:

```bash
python identify_segments.py \
  --input_jsonl dataset/triviaqa_openclaw/original_traj.jsonl \
  --output dataset/triviaqa_openclaw/segments.json \
  --include_conversations \
  --n_min 4 \
  --n_max 12 \
  --max_segments 512 \
  --min_freq 500 \
  --max_entropy 2.5
```

Conservative fallback:

```bash
python identify_segments.py \
  --input_jsonl dataset/triviaqa_openclaw/original_traj.jsonl \
  --output dataset/triviaqa_openclaw/segments_conservative.json \
  --n_min 3 \
  --n_max 8 \
  --max_segments 256 \
  --min_freq 800 \
  --max_entropy 2.0
```

Large-vocab aggressive attempt:

```bash
python identify_segments.py \
  --input_jsonl dataset/triviaqa_openclaw/original_traj.jsonl \
  --output dataset/triviaqa_openclaw/segments_aggressive.json \
  --include_conversations \
  --n_min 5 \
  --n_max 16 \
  --max_segments 1024 \
  --min_freq 200 \
  --max_entropy 3.0
```

